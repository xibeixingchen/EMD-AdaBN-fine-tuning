#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型组件 - 集成EMD指导的AdaBN的光谱网络模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from adaptive_bn import EMDAdaBN2d, EMDAdaBN3d, EMDAdaBN1d


class Conv3DBlock(nn.Module):
    """3D卷积块 with AdaBN"""
    
    def __init__(self, in_channels, out_channels, layer_name, emd_config, 
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


class SpectralAttentionBlock(nn.Module):
    """光谱注意力块 with AdaBN"""
    
    def __init__(self, num_bands, emd_config, reduction=8):
        super().__init__()
        hidden_dim = max(num_bands // reduction, 4)
        
        # 全局分支
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_bands, hidden_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_bands, 1, bias=False)
        )
        
        # 局部分支 with AdaBN
        self.local_branch = nn.Sequential(
            nn.Conv2d(num_bands, hidden_dim, 3, padding=1, bias=False),
            EMDAdaBN2d(hidden_dim, 'spectral_attention_bn', emd_config),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_bands, 1, bias=False)
        )
        
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x, adapt=False, save_source=False):
        # 全局注意力
        global_attn = self.global_branch(x)
        
        # 局部注意力 with AdaBN
        local_x = self.local_branch[0](x)
        local_x = self.local_branch[1](local_x, adapt=adapt, save_source=save_source)
        local_x = self.local_branch[2](local_x)
        local_attn = self.local_branch[3](local_x)
        
        # 自适应融合
        alpha = torch.sigmoid(self.fusion_weight)
        combined_attn = alpha * global_attn + (1 - alpha) * local_attn
        attention_weights = torch.sigmoid(combined_attn)
        
        return x * attention_weights, attention_weights.mean(dim=(2, 3))


class SpectralCNN(nn.Module):
    """3D CNN骨干网络 with AdaBN"""
    
    def __init__(self, num_bands, feature_dim, emd_config):
        super().__init__()
        
        self.conv3d_layers = nn.ModuleList([
            Conv3DBlock(1, 32, 'cnn_conv1_bn', emd_config, 
                       kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            Conv3DBlock(32, 64, 'cnn_conv2_bn', emd_config, 
                       kernel_size=(3, 3, 3), stride=(2, 1, 1)),
            Conv3DBlock(64, 128, 'cnn_conv3_bn', emd_config, 
                       kernel_size=(3, 3, 3), stride=(2, 2, 2)),
        ])
        
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 7, 7))
        
        self.feature_projection = nn.Sequential(
            nn.Conv2d(128, feature_dim, 1, bias=False),
            EMDAdaBN2d(feature_dim, 'cnn_projection_bn', emd_config),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
    def forward(self, x, adapt=False, save_source=False):
        # 添加通道维度: [B, C, H, W] -> [B, 1, C, H, W]
        x = x.unsqueeze(1)
        
        # 3D卷积
        for conv_layer in self.conv3d_layers:
            x = conv_layer(x, adapt=adapt, save_source=save_source)
        
        # 自适应池化
        x = self.adaptive_pool(x)
        x = x.squeeze(2)  # [B, 128, 1, 7, 7] -> [B, 128, 7, 7]
        
        # 特征投影
        x = self.feature_projection[0](x)
        x = self.feature_projection[1](x, adapt=adapt, save_source=save_source)
        x = self.feature_projection[2](x)
        x = self.feature_projection[3](x)
        
        return x


class SpatialProcessor(nn.Module):
    """空间注意力处理器"""
    
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, 16, 16))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 多头自注意力
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)
        
        # 前馈网络
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
        
        # 位置编码
        pos = F.interpolate(self.pos_embed, size=(h, w), mode='bilinear', align_corners=False)
        x = x + pos
        
        # 重塑为序列: [B, C, H, W] -> [B, H*W, C]
        x_seq = x.flatten(2).transpose(1, 2)
        
        # 自注意力
        x_norm = self.norm1(x_seq)
        attn_out, attn_weights = self.attention(x_norm, x_norm, x_norm)
        x_seq = x_seq + attn_out
        
        # 前馈网络
        x_seq = x_seq + self.ffn(self.norm2(x_seq))
        
        # 重塑回特征图: [B, H*W, C] -> [B, C, H, W]
        x_out = x_seq.transpose(1, 2).reshape(b, c, h, w)
        
        return x_out, attn_weights


class SpatialWrapper(nn.Module):
    """空间处理器包装器 with AdaBN"""
    
    def __init__(self, spatial_processor, emd_config, feature_dim=256):
        super().__init__()
        self.spatial_processor = spatial_processor
        
        self.pre_adabn = EMDAdaBN2d(feature_dim, 'spatial_pre_bn', emd_config)
        self.post_adabn = EMDAdaBN2d(feature_dim, 'spatial_post_bn', emd_config)
        
    def forward(self, x, adapt=False, save_source=False):
        x = self.pre_adabn(x, adapt=adapt, save_source=save_source)
        x, attn_weights = self.spatial_processor(x)
        x = self.post_adabn(x, adapt=adapt, save_source=save_source)
        return x, attn_weights


class ClassificationHead(nn.Module):
    """分类头 with AdaBN"""
    
    def __init__(self, input_dim, num_classes, emd_config, dropout_rate=0.15):
        super().__init__()
        
        # 多尺度池化
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        
        pool_dim = input_dim * 2
        
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, input_dim),
            EMDAdaBN1d(input_dim, 'classifier_bn1', emd_config),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(input_dim, input_dim // 2),
            EMDAdaBN1d(input_dim // 2, 'classifier_bn2', emd_config),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(input_dim // 2, num_classes)
        )
        
    def forward(self, x, adapt=False, save_source=False):
        # 多尺度池化
        gap_feat = self.gap(x).flatten(1)
        gmp_feat = self.gmp(x).flatten(1)
        combined = torch.cat([gap_feat, gmp_feat], dim=1)
        
        # 分类
        x = combined
        x = self.classifier[0](x)
        x = self.classifier[1](x, adapt=adapt, save_source=save_source)
        x = self.classifier[2](x)
        x = self.classifier[3](x)
        x = self.classifier[4](x)
        x = self.classifier[5](x, adapt=adapt, save_source=save_source)
        x = self.classifier[6](x)
        x = self.classifier[7](x)
        x = self.classifier[8](x)
        
        return x


class SpectralNet(nn.Module):
    """完整的EMD指导光谱网络"""
    
    def __init__(self, num_bands=19, num_classes=5, feature_dim=256, emd_config=None):
        super().__init__()
        
        self.num_bands = num_bands
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.emd_config = emd_config
        
        # 输入标准化 with AdaBN
        self.input_norm = EMDAdaBN2d(num_bands, 'input_normalized', emd_config)
        
        # 光谱注意力 with AdaBN
        self.spectral_attention = SpectralAttentionBlock(num_bands, emd_config)
        
        # 3D CNN骨干网络 with AdaBN
        self.backbone_3d = SpectralCNN(num_bands, feature_dim, emd_config)
        
        # 空间处理器 with AdaBN
        spatial_processor = SpatialProcessor(feature_dim, num_heads=8)
        self.spatial_processor = SpatialWrapper(spatial_processor, emd_config, feature_dim)
        
        # 分类头 with AdaBN
        self.classifier = ClassificationHead(feature_dim, num_classes, emd_config)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.BatchNorm1d, 
                              EMDAdaBN2d, EMDAdaBN3d, EMDAdaBN1d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def set_layer_emd_values(self, emd_values):
        """设置各层的EMD值"""
        # EMD映射关系
        emd_mapping = {
            'input_normalized': ['input_normalized'],
            'spectral_attended': ['spectral_attention_bn'],
            'cnn_features': ['cnn_conv1_bn', 'cnn_conv2_bn', 'cnn_conv3_bn', 'cnn_projection_bn'],
            'spatial_features': ['spatial_pre_bn', 'spatial_post_bn'],
            'pooled_features': ['classifier_bn1', 'classifier_bn2']
        }
        
        for emd_key, emd_value in emd_values.items():
            if emd_key in emd_mapping:
                target_layers = emd_mapping[emd_key]
                
                for layer_name in target_layers:
                    for name, module in self.named_modules():
                        if isinstance(module, (EMDAdaBN2d, EMDAdaBN3d, EMDAdaBN1d)):
                            if layer_name in module.layer_name:
                                module.set_emd_value(emd_value)
                                break
        
    def forward(self, x, adapt=False, save_source=False, return_features=False):
        """前向传播"""
        features = {}
        
        # 输入标准化
        x = self.input_norm(x, adapt=adapt, save_source=save_source)
        features['input_normalized'] = x
        
        # 光谱注意力
        x_attended, spectral_weights = self.spectral_attention(x, adapt=adapt, save_source=save_source)
        features['spectral_attended'] = x_attended
        features['spectral_weights'] = spectral_weights
        
        # 3D CNN特征提取
        cnn_features = self.backbone_3d(x_attended, adapt=adapt, save_source=save_source)
        features['cnn_features'] = cnn_features
        
        # 空间注意力处理
        spatial_features, spatial_attention = self.spatial_processor(cnn_features, adapt=adapt, save_source=save_source)
        features['spatial_features'] = spatial_features
        features['spatial_attention'] = spatial_attention
        
        # 分类
        output = self.classifier(spatial_features, adapt=adapt, save_source=save_source)
        features['classification_output'] = output
        
        if return_features:
            return output, features
        return output