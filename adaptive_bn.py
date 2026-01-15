#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EMD-guided Adaptive Batch Normalization
Adjusts batch normalization based on EMD distance
"""

import torch
import torch.nn as nn
import logging


class EMDGuidedConfig:
    """EMD guidance configuration"""
    
    def __init__(self, emd_threshold=1.5, linear_factor=0.15, max_strength=0.8):
        self.emd_threshold = emd_threshold
        self.linear_factor = linear_factor
        self.max_strength = max_strength
        
        self.layer_weights = {
            'input': 1.0,
            'spectral': 0.6, 
            'cnn': 1.0,
            'spatial': 1.8,
            'pooled': 2.2
        }
    
    def should_use_adabn(self, layer_name, emd_value):
        """Check if AdaBN should be used"""
        layer_type = self._get_layer_type(layer_name)
        weighted_emd = emd_value * self.layer_weights.get(layer_type, 1.0)
        return weighted_emd > self.emd_threshold
    
    def compute_adaptation_strength(self, layer_name, emd_value):
        """Compute adaptation strength"""
        layer_type = self._get_layer_type(layer_name)
        weighted_emd = emd_value * self.layer_weights.get(layer_type, 1.0)
        strength = self.linear_factor * weighted_emd
        return min(self.max_strength, max(0.05, strength))
    
    def _get_layer_type(self, layer_name):
        """Get layer type from name"""
        name_lower = layer_name.lower()
        if 'input' in name_lower:
            return 'input'
        elif 'spectral' in name_lower:
            return 'spectral'
        elif any(x in name_lower for x in ['cnn', 'backbone', 'conv', 'projection']):
            return 'cnn'
        elif 'spatial' in name_lower:
            return 'spatial'
        elif any(x in name_lower for x in ['pooled', 'classifier']):
            return 'pooled'
        return 'cnn'


class EMDAdaBN1d(nn.BatchNorm1d):
    """1D Adaptive Batch Normalization with EMD guidance"""
    
    def __init__(self, num_features, layer_name='', emd_config=None,
                 eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        
        self.layer_name = layer_name
        self.emd_config = emd_config
        
        # Source domain statistics
        self.register_buffer('source_mean', torch.zeros(num_features))
        self.register_buffer('source_var', torch.ones(num_features))
        
        # Adaptation control
        self.register_buffer('adaptation_count', torch.zeros(1))
        self.register_buffer('emd_value', torch.zeros(1))
        self.register_buffer('adaptation_strength', torch.zeros(1))
        self.register_buffer('use_adabn', torch.zeros(1, dtype=torch.bool))
    
    def set_emd_value(self, emd_value):
        """Set EMD value and update strategy"""
        self.emd_value.fill_(emd_value)
        
        if self.emd_config is not None:
            should_use = self.emd_config.should_use_adabn(self.layer_name, emd_value)
            self.use_adabn.fill_(should_use)
            
            if should_use:
                strength = self.emd_config.compute_adaptation_strength(self.layer_name, emd_value)
                self.adaptation_strength.fill_(strength)
        else:
            self.use_adabn.fill_(emd_value > 1.5)
            self.adaptation_strength.fill_(min(0.8, 0.15 * emd_value))
    
    def save_source_stats(self):
        """Save source domain statistics"""
        self.source_mean.copy_(self.running_mean)
        self.source_var.copy_(self.running_var)
    
    def reset_adaptation(self):
        """Reset adaptation state"""
        self.adaptation_count.zero_()
        self.running_mean.copy_(self.source_mean)
        self.running_var.copy_(self.source_var)
    
    def forward(self, x, adapt=False, save_source=False):
        if save_source:
            output = super().forward(x)
            self.save_source_stats()
            return output
        
        if adapt and self.use_adabn.item():
            return self._adaptive_forward(x)
        
        return super().forward(x)
    
    def _adaptive_forward(self, x):
        """Forward with adaptive BN"""
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input, got {x.dim()}D")
        
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        
        # Progressive adaptation
        progress = torch.sigmoid((self.adaptation_count - 15) / 10.0)
        progress = torch.clamp(progress, 0.0, 0.9)
        
        target_mean = progress * batch_mean + (1 - progress) * self.source_mean
        target_var = progress * batch_var + (1 - progress) * self.source_var
        
        # Update running stats
        strength = self.adaptation_strength.item()
        adaptive_momentum = min(0.9, self.momentum * (1.0 + strength * 4.0))
        
        with torch.no_grad():
            self.running_mean.mul_(1 - adaptive_momentum).add_(target_mean, alpha=adaptive_momentum)
            self.running_var.mul_(1 - adaptive_momentum).add_(target_var, alpha=adaptive_momentum)
            self.adaptation_count += 1
        
        # Normalize
        x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        return x_norm * self.weight + self.bias


class EMDAdaBN2d(nn.BatchNorm2d):
    """2D Adaptive Batch Normalization with EMD guidance"""
    
    def __init__(self, num_features, layer_name='', emd_config=None,
                 eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        
        self.layer_name = layer_name
        self.emd_config = emd_config
        
        self.register_buffer('source_mean', torch.zeros(num_features))
        self.register_buffer('source_var', torch.ones(num_features))
        self.register_buffer('adaptation_count', torch.zeros(1))
        self.register_buffer('emd_value', torch.zeros(1))
        self.register_buffer('adaptation_strength', torch.zeros(1))
        self.register_buffer('use_adabn', torch.zeros(1, dtype=torch.bool))
    
    def set_emd_value(self, emd_value):
        """Set EMD value and update strategy"""
        self.emd_value.fill_(emd_value)
        
        if self.emd_config is not None:
            should_use = self.emd_config.should_use_adabn(self.layer_name, emd_value)
            self.use_adabn.fill_(should_use)
            
            if should_use:
                strength = self.emd_config.compute_adaptation_strength(self.layer_name, emd_value)
                self.adaptation_strength.fill_(strength)
        else:
            self.use_adabn.fill_(emd_value > 1.5)
            self.adaptation_strength.fill_(min(0.8, 0.15 * emd_value))
    
    def save_source_stats(self):
        """Save source domain statistics"""
        self.source_mean.copy_(self.running_mean)
        self.source_var.copy_(self.running_var)
    
    def reset_adaptation(self):
        """Reset adaptation state"""
        self.adaptation_count.zero_()
        self.running_mean.copy_(self.source_mean)
        self.running_var.copy_(self.source_var)
    
    def forward(self, x, adapt=False, save_source=False):
        if save_source:
            output = super().forward(x)
            self.save_source_stats()
            return output
        
        if adapt and self.use_adabn.item():
            return self._adaptive_forward(x)
        
        return super().forward(x)
    
    def _adaptive_forward(self, x):
        """Forward with adaptive BN"""
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got {x.dim()}D")
        
        batch_mean = x.mean(dim=(0, 2, 3))
        batch_var = x.var(dim=(0, 2, 3), unbiased=False)
        
        progress = torch.sigmoid((self.adaptation_count - 10) / 8.0)
        progress = torch.clamp(progress, 0.0, 0.95)
        
        target_mean = progress * batch_mean + (1 - progress) * self.source_mean
        target_var = progress * batch_var + (1 - progress) * self.source_var
        
        strength = self.adaptation_strength.item()
        adaptive_momentum = min(0.9, self.momentum * (1.0 + strength * 4.0))
        
        with torch.no_grad():
            self.running_mean.mul_(1 - adaptive_momentum).add_(target_mean, alpha=adaptive_momentum)
            self.running_var.mul_(1 - adaptive_momentum).add_(target_var, alpha=adaptive_momentum)
            self.adaptation_count += 1
        
        mean = self.running_mean.view(1, -1, 1, 1)
        var = self.running_var.view(1, -1, 1, 1)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * weight + bias


class EMDAdaBN3d(nn.BatchNorm3d):
    """3D Adaptive Batch Normalization with EMD guidance"""
    
    def __init__(self, num_features, layer_name='', emd_config=None,
                 eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        
        self.layer_name = layer_name
        self.emd_config = emd_config
        
        self.register_buffer('source_mean', torch.zeros(num_features))
        self.register_buffer('source_var', torch.ones(num_features))
        self.register_buffer('adaptation_count', torch.zeros(1))
        self.register_buffer('emd_value', torch.zeros(1))
        self.register_buffer('adaptation_strength', torch.zeros(1))
        self.register_buffer('use_adabn', torch.zeros(1, dtype=torch.bool))
    
    def set_emd_value(self, emd_value):
        """Set EMD value and update strategy"""
        self.emd_value.fill_(emd_value)
        
        if self.emd_config is not None:
            should_use = self.emd_config.should_use_adabn(self.layer_name, emd_value)
            self.use_adabn.fill_(should_use)
            
            if should_use:
                strength = self.emd_config.compute_adaptation_strength(self.layer_name, emd_value)
                self.adaptation_strength.fill_(strength)
        else:
            self.use_adabn.fill_(emd_value > 1.5)
            self.adaptation_strength.fill_(min(0.8, 0.15 * emd_value))
    
    def save_source_stats(self):
        """Save source domain statistics"""
        self.source_mean.copy_(self.running_mean)
        self.source_var.copy_(self.running_var)
    
    def reset_adaptation(self):
        """Reset adaptation state"""
        self.adaptation_count.zero_()
        self.running_mean.copy_(self.source_mean)
        self.running_var.copy_(self.source_var)
    
    def forward(self, x, adapt=False, save_source=False):
        if save_source:
            output = super().forward(x)
            self.save_source_stats()
            return output
        
        if adapt and self.use_adabn.item():
            return self._adaptive_forward(x)
        
        return super().forward(x)
    
    def _adaptive_forward(self, x):
        """Forward with adaptive BN"""
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input, got {x.dim()}D")
        
        batch_mean = x.mean(dim=(0, 2, 3, 4))
        batch_var = x.var(dim=(0, 2, 3, 4), unbiased=False)
        
        progress = torch.sigmoid((self.adaptation_count - 8) / 6.0)
        progress = torch.clamp(progress, 0.0, 0.9)
        
        target_mean = progress * batch_mean + (1 - progress) * self.source_mean
        target_var = progress * batch_var + (1 - progress) * self.source_var
        
        strength = self.adaptation_strength.item()
        adaptive_momentum = min(0.9, self.momentum * (1.0 + strength * 4.0))
        
        with torch.no_grad():
            self.running_mean.mul_(1 - adaptive_momentum).add_(target_mean, alpha=adaptive_momentum)
            self.running_var.mul_(1 - adaptive_momentum).add_(target_var, alpha=adaptive_momentum)
            self.adaptation_count += 1
        
        mean = self.running_mean.view(1, -1, 1, 1, 1)
        var = self.running_var.view(1, -1, 1, 1, 1)
        weight = self.weight.view(1, -1, 1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1, 1)
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * weight + bias
