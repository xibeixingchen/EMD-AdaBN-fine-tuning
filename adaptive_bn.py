#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EMD指导的自适应批归一化 (Adaptive Batch Normalization)
根据EMD距离自适应调整批归一化策略
"""

import torch
import torch.nn as nn
import logging


class EMDGuidedConfig:
    """EMD指导策略配置"""
    
    def __init__(self, emd_threshold=1.5, linear_factor=0.15, max_strength=0.8):
        self.emd_threshold = emd_threshold
        self.linear_factor = linear_factor
        self.max_strength = max_strength
        
        # 不同层的权重
        self.layer_weights = {
            'input': 1.0,
            'spectral': 0.6, 
            'cnn': 1.0,
            'spatial': 1.8,
            'pooled': 2.2
        }
    
    def should_use_adabn(self, layer_name, emd_value):
        """判断是否使用AdaBN"""
        layer_type = self._get_layer_type(layer_name)
        weighted_emd = emd_value * self.layer_weights.get(layer_type, 1.0)
        return weighted_emd > self.emd_threshold
    
    def compute_adaptation_strength(self, layer_name, emd_value):
        """计算适应强度"""
        layer_type = self._get_layer_type(layer_name)
        weighted_emd = emd_value * self.layer_weights.get(layer_type, 1.0)
        
        strength = self.linear_factor * weighted_emd
        return min(self.max_strength, max(0.05, strength))
    
    def _get_layer_type(self, layer_name):
        """获取层类型"""
        name_lower = layer_name.lower()
        if 'input' in name_lower:
            return 'input'
        elif 'spectral' in name_lower:
            return 'spectral'
        elif any(x in name_lower for x in ['cnn', 'backbone', 'conv']):
            return 'cnn'
        elif 'spatial' in name_lower:
            return 'spatial'
        elif any(x in name_lower for x in ['pooled', 'classifier']):
            return 'pooled'
        return 'cnn'


class EMDAdaBN(nn.Module):
    """EMD指导的自适应批归一化基类"""
    
    def __init__(self, num_features, layer_name, emd_config, 
                 eps=1e-5, momentum=0.1, track_running_stats=True):
        super().__init__()
        
        self.num_features = num_features
        self.layer_name = layer_name
        self.emd_config = emd_config
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        
        # 参数
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        if track_running_stats:
            # 统计量
            self.register_buffer('source_mean', torch.zeros(num_features))
            self.register_buffer('source_var', torch.ones(num_features))
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            
            # 控制参数
            self.register_buffer('adaptation_count', torch.zeros(1))
            self.register_buffer('emd_value', torch.zeros(1))
            self.register_buffer('adaptation_strength', torch.zeros(1))
            self.register_buffer('use_adabn', torch.zeros(1, dtype=torch.bool))
    
    def set_emd_value(self, emd_value):
        """设置EMD值并更新策略"""
        self.emd_value.fill_(emd_value)
        should_use = self.emd_config.should_use_adabn(self.layer_name, emd_value)
        self.use_adabn.fill_(should_use)
        
        if should_use:
            strength = self.emd_config.compute_adaptation_strength(self.layer_name, emd_value)
            self.adaptation_strength.fill_(strength)
            logging.info(f"Layer {self.layer_name}: EMD={emd_value:.4f} → Use AdaBN, Strength={strength:.4f}")
        else:
            self.adaptation_strength.fill_(0.0)
    
    def save_source_stats(self):
        """保存源域统计量"""
        if self.track_running_stats:
            self.source_mean.copy_(self.running_mean)
            self.source_var.copy_(self.running_var)
    
    def get_adaptive_momentum(self):
        """获取自适应动量"""
        if not self.use_adabn.item():
            return self.momentum
        
        strength = self.adaptation_strength.item()
        adaptive_momentum = self.momentum * (1.0 + strength * 4.0)
        return min(adaptive_momentum, 0.9)


class EMDAdaBN2d(EMDAdaBN):
    """2D自适应批归一化"""
    
    def forward(self, x, adapt=False, save_source=False):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B,C,H,W), got {x.dim()}D")
        
        # 保存源域统计量
        if save_source:
            batch_mean = x.mean(dim=(0, 2, 3))
            batch_var = x.var(dim=(0, 2, 3), unbiased=False)
            
            if self.track_running_stats:
                with torch.no_grad():
                    momentum = self.momentum
                    self.running_mean.mul_(1 - momentum).add_(batch_mean, alpha=momentum)
                    self.running_var.mul_(1 - momentum).add_(batch_var, alpha=momentum)
                    self.save_source_stats()
            
            mean, var = batch_mean, batch_var
            
        # 自适应调整
        elif adapt and self.use_adabn.item():
            batch_mean = x.mean(dim=(0, 2, 3))
            batch_var = x.var(dim=(0, 2, 3), unbiased=False)
            
            if self.track_running_stats:
                with torch.no_grad():
                    adaptive_momentum = self.get_adaptive_momentum()
                    
                    # 渐进式适应
                    progress = torch.sigmoid((self.adaptation_count - 10) / 8.0)
                    progress = torch.clamp(progress, 0.0, 0.95)
                    
                    target_mean = progress * batch_mean + (1 - progress) * self.source_mean
                    target_var = progress * batch_var + (1 - progress) * self.source_var
                    
                    self.running_mean.mul_(1 - adaptive_momentum).add_(target_mean, alpha=adaptive_momentum)
                    self.running_var.mul_(1 - adaptive_momentum).add_(target_var, alpha=adaptive_momentum)
                    self.adaptation_count += 1
            
            mean = self.running_mean.detach()
            var = self.running_var.detach()
            
        # 标准前向传播
        else:
            if self.training and not (adapt or save_source):
                if self.use_adabn.item() and self.track_running_stats:
                    mean = self.running_mean.detach()
                    var = self.running_var.detach()
                else:
                    mean = x.mean(dim=(0, 2, 3))
                    var = x.var(dim=(0, 2, 3), unbiased=False)
                    
                    if self.track_running_stats:
                        with torch.no_grad():
                            self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
                            self.running_var.mul_(1 - self.momentum).add_(var, alpha=self.momentum)
            else:
                mean = self.running_mean.detach() if self.track_running_stats else x.mean(dim=(0, 2, 3))
                var = self.running_var.detach() if self.track_running_stats else x.var(dim=(0, 2, 3), unbiased=False)
        
        # 批归一化
        mean = mean.view(1, -1, 1, 1)
        var = var.view(1, -1, 1, 1)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * weight + bias


class EMDAdaBN3d(EMDAdaBN):
    """3D自适应批归一化"""
    
    def forward(self, x, adapt=False, save_source=False):
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B,C,D,H,W), got {x.dim()}D")
        
        if save_source:
            batch_mean = x.mean(dim=(0, 2, 3, 4))
            batch_var = x.var(dim=(0, 2, 3, 4), unbiased=False)
            
            if self.track_running_stats:
                with torch.no_grad():
                    momentum = self.momentum
                    self.running_mean.mul_(1 - momentum).add_(batch_mean, alpha=momentum)
                    self.running_var.mul_(1 - momentum).add_(batch_var, alpha=momentum)
                    self.save_source_stats()
            
            mean, var = batch_mean, batch_var
            
        elif adapt and self.use_adabn.item():
            batch_mean = x.mean(dim=(0, 2, 3, 4))
            batch_var = x.var(dim=(0, 2, 3, 4), unbiased=False)
            
            if self.track_running_stats:
                with torch.no_grad():
                    adaptive_momentum = self.get_adaptive_momentum()
                    progress = torch.sigmoid((self.adaptation_count - 8) / 6.0)
                    progress = torch.clamp(progress, 0.0, 0.9)
                    
                    target_mean = progress * batch_mean + (1 - progress) * self.source_mean
                    target_var = progress * batch_var + (1 - progress) * self.source_var
                    
                    self.running_mean.mul_(1 - adaptive_momentum).add_(target_mean, alpha=adaptive_momentum)
                    self.running_var.mul_(1 - adaptive_momentum).add_(target_var, alpha=adaptive_momentum)
                    self.adaptation_count += 1
            
            mean = self.running_mean.detach()
            var = self.running_var.detach()
        else:
            if self.training and not (adapt or save_source):
                mean = x.mean(dim=(0, 2, 3, 4))
                var = x.var(dim=(0, 2, 3, 4), unbiased=False)
                
                if self.track_running_stats:
                    with torch.no_grad():
                        self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
                        self.running_var.mul_(1 - self.momentum).add_(var, alpha=self.momentum)
            else:
                mean = self.running_mean.detach() if self.track_running_stats else x.mean(dim=(0, 2, 3, 4))
                var = self.running_var.detach() if self.track_running_stats else x.var(dim=(0, 2, 3, 4), unbiased=False)
        
        mean = mean.view(1, -1, 1, 1, 1)
        var = var.view(1, -1, 1, 1, 1)
        weight = self.weight.view(1, -1, 1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1, 1)
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * weight + bias


class EMDAdaBN1d(EMDAdaBN):
    """1D自适应批归一化"""
    
    def forward(self, x, adapt=False, save_source=False):
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input (B,C), got {x.dim()}D")
        
        if save_source:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            
            if self.track_running_stats:
                with torch.no_grad():
                    momentum = self.momentum
                    self.running_mean.mul_(1 - momentum).add_(batch_mean, alpha=momentum)
                    self.running_var.mul_(1 - momentum).add_(batch_var, alpha=momentum)
                    self.save_source_stats()
            
            mean, var = batch_mean, batch_var
            
        elif adapt and self.use_adabn.item():
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            
            if self.track_running_stats:
                with torch.no_grad():
                    adaptive_momentum = self.get_adaptive_momentum()
                    progress = torch.sigmoid((self.adaptation_count - 15) / 10.0)
                    progress = torch.clamp(progress, 0.0, 0.9)
                    
                    target_mean = progress * batch_mean + (1 - progress) * self.source_mean
                    target_var = progress * batch_var + (1 - progress) * self.source_var
                    
                    self.running_mean.mul_(1 - adaptive_momentum).add_(target_mean, alpha=adaptive_momentum)
                    self.running_var.mul_(1 - adaptive_momentum).add_(target_var, alpha=adaptive_momentum)
                    self.adaptation_count += 1
            
            mean = self.running_mean.detach()
            var = self.running_var.detach()
        else:
            if self.training and not (adapt or save_source):
                mean = x.mean(dim=0)
                var = x.var(dim=0, unbiased=False)
                
                if self.track_running_stats:
                    with torch.no_grad():
                        self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
                        self.running_var.mul_(1 - self.momentum).add_(var, alpha=self.momentum)
            else:
                mean = self.running_mean.detach() if self.track_running_stats else x.mean(dim=0)
                var = self.running_var.detach() if self.track_running_stats else x.var(dim=0, unbiased=False)
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias