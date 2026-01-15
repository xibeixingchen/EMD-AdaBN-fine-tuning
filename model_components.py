#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SpectralNet Model Components with EMD-guided AdaBN
Structure matches the training model for correct weight loading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from adaptive_bn import EMDAdaBN1d, EMDAdaBN2d, EMDAdaBN3d, EMDGuidedConfig


class SpectralAttentionBlock(nn.Module):
    """Spectral attention block with AdaBN"""
    
    def __init__(self, num_bands, emd_config=None, reduction=8):
        super().__init__()
        
        self.num_bands = num_bands
        hidden_dim = max(num_bands // reduction, 4)
        
        # Global branch
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_bands, hidden_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_bands, 1, bias=False)
        )
        
        # Local branch with AdaBN
        self.local_branch_conv1 = nn.Conv2d(num_bands, hidden_dim, 3, padding=1, bias=False)
        self.local_branch_bn = EMDAdaBN2d(hidden_dim, 'spectral_attention.local_branch.1', emd_config)
        self.local_branch_relu = nn.ReLU(inplace=True)
        self.local_branch_conv2 = nn.Conv2d(hidden_dim, num_bands, 1, bias=False)
        
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x, adapt=False, save_source=False):
        global_attn = self.global_branch(x)
        
        local_x = self.local_branch_conv1(x)
        local_x = self.local_branch_bn(local_x, adapt=adapt, save_source=save_source)
        local_x = self.local_branch_relu(local_x)
        local_attn = self.local_branch_conv2(local_x)
        
        alpha = torch.sigmoid(self.fusion_weight)
        combined_attn = alpha * global_attn + (1 - alpha) * local_attn
        attention_weights = torch.sigmoid(combined_attn)
        
        return x * attention_weights, attention_weights.mean(dim=(2, 3))


class Conv3DBlock(nn.Module):
    """3D convolution block with AdaBN"""
    
    def __init__(self, in_channels, out_channels, layer_name, emd_config=None,
                 kernel_size=(3, 3, 3), stride=(1, 1, 1)):
        super().__init__()
        
        padding = tuple(k // 2 for k in kernel_size)
        
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False)
        self.bn3d = EMDAdaBN3d(out_channels, layer_name, emd_config)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, adapt=False, save_source=False):
        x = self.conv3d(x)
        x = self.bn3d(x, adapt=adapt, save_source=save_source)
        return self.relu(x)


class SpectralCNN(nn.Module):
    """3D CNN backbone with AdaBN"""
    
    def __init__(self, num_bands, feature_dim=256, emd_config=None):
        super().__init__()
        
        self.num_bands = num_bands
        self.feature_dim = feature_dim
        
        self.conv3d_layers = nn.ModuleList([
            Conv3DBlock(1, 32, 'backbone_3d.conv3d_layers.0.bn3d', emd_config,
                       kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            Conv3DBlock(32, 64, 'backbone_3d.conv3d_layers.1.bn3d', emd_config,
                       kernel_size=(3, 3, 3), stride=(2, 1, 1)),
            Conv3DBlock(64, 128, 'backbone_3d.conv3d_layers.2.bn3d', emd_config,
                       kernel_size=(3, 3, 3), stride=(2, 2, 2)),
        ])
        
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 7, 7))
        
        # Feature projection
        self.feature_projection_conv = nn.Conv2d(128, feature_dim, 1, bias=False)
        self.feature_projection_bn = EMDAdaBN2d(feature_dim, 'backbone_3d.feature_projection.1', emd_config)
        self.feature_projection_relu = nn.ReLU(inplace=True)
        self.feature_projection_dropout = nn.Dropout2d(0.1)
        
    def forward(self, x, adapt=False, save_source=False):
        x = x.unsqueeze(1)
        
        for conv_layer in self.conv3d_layers:
            x = conv_layer(x, adapt=adapt, save_source=save_source)
        
        x = self.adaptive_pool(x)
        x = x.squeeze(2)
        
        x = self.feature_projection_conv(x)
        x = self.feature_projection_bn(x, adapt=adapt, save_source=save_source)
        x = self.feature_projection_relu(x)
        x = self.feature_projection_dropout(x)
        
        return x


class SpatialAttentionProcessor(nn.Module):
    """Spatial attention processor"""
    
    def __init__(self, dim, num_heads=8):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, 16, 16))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        pos = F.interpolate(self.pos_embed, size=(h, w), mode='bilinear', align_corners=False)
        x = x + pos
        
        x_seq = x.flatten(2).transpose(1, 2)
        
        x_norm = self.norm1(x_seq)
        attn_out, attn_weights = self.attention(x_norm, x_norm, x_norm)
        x_seq = x_seq + attn_out
        
        x_seq = x_seq + self.ffn(self.norm2(x_seq))
        
        x_out = x_seq.transpose(1, 2).reshape(b, c, h, w)
        
        return x_out, attn_weights


class SpectralClassificationHead(nn.Module):
    """Classification head with AdaBN"""
    
    def __init__(self, input_dim, num_classes, emd_config=None, dropout_rate=0.15):
        super().__init__()
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        
        pool_dim = input_dim * 2
        
        # Classifier layers
        self.fc1 = nn.Linear(pool_dim, input_dim)
        self.bn1 = EMDAdaBN1d(input_dim, 'classifier.classifier.1', emd_config)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(input_dim, input_dim // 2)
        self.bn2 = EMDAdaBN1d(input_dim // 2, 'classifier.classifier.5', emd_config)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout_rate * 0.5)
        
        self.fc3 = nn.Linear(input_dim // 2, num_classes)
        
    def forward(self, x, adapt=False, save_source=False):
        gap_feat = self.gap(x).flatten(1)
        gmp_feat = self.gmp(x).flatten(1)
        combined = torch.cat([gap_feat, gmp_feat], dim=1)
        
        x = self.fc1(combined)
        x = self.bn1(x, adapt=adapt, save_source=save_source)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x, adapt=adapt, save_source=save_source)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


class SpectralNet(nn.Module):
    """Complete SpectralNet with EMD-guided AdaBN"""
    
    def __init__(self, num_bands=19, num_classes=5, config=None, emd_config=None):
        super().__init__()
        
        # Handle config
        if config is None:
            config = {
                'feature_dim': 256,
                'dropout_rate': 0.15,
                'spectral_attention_reduction': 8,
                'spatial_attention_heads': 8
            }
        
        self.config = config
        self.num_bands = num_bands
        self.num_classes = num_classes
        self.emd_config = emd_config
        
        feature_dim = config.get('feature_dim', 256)
        dropout_rate = config.get('dropout_rate', 0.15)
        spectral_reduction = config.get('spectral_attention_reduction', 8)
        spatial_heads = config.get('spatial_attention_heads', 8)
        
        # Input normalization
        self.input_norm = EMDAdaBN2d(num_bands, 'input_norm', emd_config)
        
        # Spectral attention
        self.spectral_attention = SpectralAttentionBlock(
            num_bands, emd_config, reduction=spectral_reduction
        )
        
        # 3D CNN backbone
        self.backbone_3d = SpectralCNN(num_bands, feature_dim, emd_config)
        
        # Spatial processor
        self.spatial_processor = SpatialAttentionProcessor(feature_dim, spatial_heads)
        
        # Classification head
        self.classifier = SpectralClassificationHead(
            feature_dim, num_classes, emd_config, dropout_rate
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def set_layer_emd_values(self, emd_values):
        """Set EMD values for all AdaBN layers"""
        # Mapping from EMD keys to layer name patterns
        emd_mapping = {
            'input_normalized': ['input_norm'],
            'spectral_attended': ['spectral_attention'],
            'cnn_features': ['backbone_3d', 'conv3d', 'feature_projection'],
            'spatial_features': ['spatial'],
            'fused_features': ['backbone_3d', 'spatial'],
            'pooled_features': ['classifier']
        }
        
        for module in self.modules():
            if isinstance(module, (EMDAdaBN1d, EMDAdaBN2d, EMDAdaBN3d)):
                layer_name = module.layer_name.lower()
                
                for emd_key, patterns in emd_mapping.items():
                    if emd_key in emd_values:
                        if any(p in layer_name for p in patterns):
                            module.set_emd_value(emd_values[emd_key])
                            break
    
    def save_source_statistics(self):
        """Save source domain BN statistics"""
        for module in self.modules():
            if isinstance(module, (EMDAdaBN1d, EMDAdaBN2d, EMDAdaBN3d)):
                module.save_source_stats()
    
    def reset_adaptation(self):
        """Reset adaptation state"""
        for module in self.modules():
            if isinstance(module, (EMDAdaBN1d, EMDAdaBN2d, EMDAdaBN3d)):
                module.reset_adaptation()
    
    def forward(self, x, adapt=False, save_source=False, return_features=False):
        features = {}
        
        # Input normalization
        x = self.input_norm(x, adapt=adapt, save_source=save_source)
        features['input_normalized'] = x
        
        # Spectral attention
        x_attended, spectral_weights = self.spectral_attention(x, adapt=adapt, save_source=save_source)
        features['spectral_attended'] = x_attended
        features['spectral_weights'] = spectral_weights
        
        # 3D CNN
        cnn_features = self.backbone_3d(x_attended, adapt=adapt, save_source=save_source)
        features['cnn_features'] = cnn_features
        
        # Spatial processing
        spatial_features, spatial_attention = self.spatial_processor(cnn_features)
        features['spatial_features'] = spatial_features
        features['spatial_attention'] = spatial_attention
        
        # Classification
        output = self.classifier(spatial_features, adapt=adapt, save_source=save_source)
        features['classification_output'] = output
        
        if return_features:
            return output, features
        return output


def load_pretrained_weights(model, checkpoint_path, device='cuda'):
    """Load pretrained weights from standard model to AdaBN model"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        source_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        source_dict = checkpoint['state_dict']
    else:
        source_dict = checkpoint
    
    target_dict = model.state_dict()
    loaded_keys = []
    
    for key, value in source_dict.items():
        if key in target_dict:
            if target_dict[key].shape == value.shape:
                target_dict[key] = value
                loaded_keys.append(key)
                
                # Copy running stats to source stats
                if 'running_mean' in key:
                    source_key = key.replace('running_mean', 'source_mean')
                    if source_key in target_dict:
                        target_dict[source_key] = value.clone()
                elif 'running_var' in key:
                    source_key = key.replace('running_var', 'source_var')
                    if source_key in target_dict:
                        target_dict[source_key] = value.clone()
    
    model.load_state_dict(target_dict, strict=False)
    
    return len(loaded_keys), len(source_dict)
