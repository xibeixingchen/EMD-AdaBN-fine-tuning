#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
import argparse
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader, TensorDataset
import time

try:
    import ot
    HAS_POT = True
except ImportError:
    HAS_POT = False

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


class SpectralAttentionBlock(nn.Module):
    def __init__(self, num_bands, reduction=8):
        super().__init__()
        self.num_bands = num_bands
        hidden_dim = max(num_bands // reduction, 4)
        
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_bands, hidden_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_bands, 1, bias=False)
        )
        
        self.local_branch = nn.Sequential(
            nn.Conv2d(num_bands, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_bands, 1, bias=False)
        )
        
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        global_attn = self.global_branch(x)
        local_attn = self.local_branch(x)
        alpha = torch.sigmoid(self.fusion_weight)
        combined_attn = alpha * global_attn + (1 - alpha) * local_attn
        attention_weights = torch.sigmoid(combined_attn)
        return x * attention_weights, attention_weights.mean(dim=(2, 3))


class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1)):
        super().__init__()
        padding = tuple(k // 2 for k in kernel_size)
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False)
        self.bn3d = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn3d(self.conv3d(x)))


class SpectralCNN(nn.Module):
    def __init__(self, num_bands, feature_dim=256):
        super().__init__()
        self.conv3d_layers = nn.ModuleList([
            Conv3DBlock(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            Conv3DBlock(32, 64, kernel_size=(3, 3, 3), stride=(2, 1, 1)),
            Conv3DBlock(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2)),
        ])
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 7, 7))
        self.feature_projection = nn.Sequential(
            nn.Conv2d(128, feature_dim, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        for conv_layer in self.conv3d_layers:
            x = conv_layer(x)
        x = self.adaptive_pool(x).squeeze(2)
        return self.feature_projection(x)


class SpatialAttentionProcessor(nn.Module):
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
        return x_seq.transpose(1, 2).reshape(b, c, h, w), attn_weights


class SpectralClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.15):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        pool_dim = input_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(input_dim // 2, num_classes)
        )
        
    def forward(self, x):
        gap_feat = self.gap(x).flatten(1)
        gmp_feat = self.gmp(x).flatten(1)
        return self.classifier(torch.cat([gap_feat, gmp_feat], dim=1))


class SpectralNet(nn.Module):
    def __init__(self, num_bands=19, num_classes=5, config=None):
        super().__init__()
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
        feature_dim = config.get('feature_dim', 256)
        
        self.input_norm = nn.BatchNorm2d(num_bands)
        self.spectral_attention = SpectralAttentionBlock(
            num_bands, config.get('spectral_attention_reduction', 8)
        )
        self.backbone_3d = SpectralCNN(num_bands, feature_dim)
        self.spatial_processor = SpatialAttentionProcessor(
            feature_dim, config.get('spatial_attention_heads', 8)
        )
        self.classifier = SpectralClassificationHead(
            feature_dim, num_classes, config.get('dropout_rate', 0.15)
        )
        
    def forward(self, x, return_features=False):
        features = {}
        x = self.input_norm(x)
        features['input_normalized'] = x
        x_attended, _ = self.spectral_attention(x)
        features['spectral_attended'] = x_attended
        cnn_features = self.backbone_3d(x_attended)
        features['cnn_features'] = cnn_features
        spatial_features, _ = self.spatial_processor(cnn_features)
        features['spatial_features'] = spatial_features
        output = self.classifier(spatial_features)
        if return_features:
            return output, features
        return output


class FeatureExtractor:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.features = {}
        self.hooks = []
        
    def _get_activation(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            if len(output.shape) > 2:
                output = output.view(output.size(0), -1)
            self.features[name] = output.detach().cpu().numpy()
        return hook
    
    def register_hooks(self):
        hook_configs = [
            ('input_norm', 'input_normalized'),
            ('spectral_attention', 'spectral_attended'),
            ('backbone_3d.feature_projection', 'cnn_features'),
            ('spatial_processor', 'spatial_features'),
            ('classifier.classifier.1', 'pooled_features'),
        ]
        for module_path, feature_name in hook_configs:
            try:
                module = self._get_module(module_path)
                if module is not None:
                    hook = module.register_forward_hook(self._get_activation(feature_name))
                    self.hooks.append(hook)
            except Exception as e:
                logging.warning(f"Hook registration failed for {module_path}: {e}")
    
    def _get_module(self, path):
        parts = path.split('.')
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            elif part.isdigit() and hasattr(module, '__getitem__'):
                module = module[int(part)]
            else:
                return None
        return module
    
    def extract(self, dataloader, max_samples=1000):
        self.model.eval()
        feature_names = ['input_normalized', 'spectral_attended', 'cnn_features', 
                        'spatial_features', 'pooled_features']
        all_features = {name: [] for name in feature_names}
        sample_count = 0
        
        with torch.no_grad():
            for inputs, _ in dataloader:
                if sample_count >= max_samples:
                    break
                inputs = inputs.to(self.device)
                self.features.clear()
                _ = self.model(inputs)
                for name in feature_names:
                    if name in self.features:
                        all_features[name].append(self.features[name])
                sample_count += inputs.size(0)
        
        for name in feature_names:
            if all_features[name]:
                all_features[name] = np.concatenate(all_features[name], axis=0)
            else:
                all_features[name] = np.array([])
        
        if len(all_features['cnn_features']) > 0 and len(all_features['spatial_features']) > 0:
            n = min(len(all_features['cnn_features']), len(all_features['spatial_features']))
            all_features['fused_features'] = np.concatenate([
                all_features['cnn_features'][:n],
                all_features['spatial_features'][:n]
            ], axis=1)
        else:
            all_features['fused_features'] = np.array([])
        
        return all_features
    
    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


if HAS_NUMBA:
    @jit(nopython=True, parallel=True)
    def _wasserstein_1d_batch_numba(X_sorted, Y_sorted):
        n_proj = X_sorted.shape[1]
        distances = np.zeros(n_proj)
        for i in prange(n_proj):
            distances[i] = np.mean(np.abs(X_sorted[:, i] - Y_sorted[:, i]))
        return distances


class OptimizedEMD:
    EXACT_THRESHOLD = 1e6
    SINKHORN_THRESHOLD = 1e7
    
    def __init__(self, method='auto'):
        self.method = method
        self._last_method = None
    
    def compute(self, X, Y, **kwargs):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        method = self._select_method(X, Y) if self.method == 'auto' else self.method
        self._last_method = method
        
        if method == 'exact':
            return self._exact_emd(X, Y, **kwargs)
        elif method == 'sinkhorn':
            return self._sinkhorn_emd(X, Y, **kwargs)
        else:
            return self._sliced_emd(X, Y, **kwargs)
    
    def _select_method(self, X, Y):
        size = len(X) * len(Y)
        if size <= self.EXACT_THRESHOLD and HAS_POT:
            return 'exact'
        elif size <= self.SINKHORN_THRESHOLD and HAS_POT:
            return 'sinkhorn'
        return 'sliced'
    
    def _exact_emd(self, X, Y, metric='euclidean'):
        if not HAS_POT:
            return self._sliced_emd(X, Y)
        
        n, m = len(X), len(Y)
        a, b = np.ones(n) / n, np.ones(m) / m
        M = cdist(X, Y, metric=metric)
        max_val = M.max()
        if max_val > 0:
            M = M / max_val
        emd = ot.emd2(a, b, M)
        return emd * max_val if max_val > 0 else emd
    
    def _sinkhorn_emd(self, X, Y, reg=0.05, num_iters=200, metric='euclidean'):
        if not HAS_POT:
            return self._sliced_emd(X, Y)
        
        n, m = len(X), len(Y)
        a, b = np.ones(n) / n, np.ones(m) / m
        M = cdist(X, Y, metric=metric)
        max_val = M.max()
        if max_val > 0:
            M = M / max_val
        emd = ot.sinkhorn2(a, b, M, reg, numItermax=num_iters)
        return emd * max_val if max_val > 0 else emd
    
    def _sliced_emd(self, X, Y, n_projections=1000):
        d = X.shape[1]
        n_proj = min(n_projections, d * 10)
        
        thetas = np.random.randn(n_proj, d)
        thetas /= np.linalg.norm(thetas, axis=1, keepdims=True)
        
        X_proj = X @ thetas.T
        Y_proj = Y @ thetas.T
        
        if len(X) == len(Y):
            X_sorted = np.sort(X_proj, axis=0)
            Y_sorted = np.sort(Y_proj, axis=0)
            
            if HAS_NUMBA:
                X_sorted = np.ascontiguousarray(X_sorted)
                Y_sorted = np.ascontiguousarray(Y_sorted)
                distances = _wasserstein_1d_batch_numba(X_sorted, Y_sorted)
            else:
                distances = np.mean(np.abs(X_sorted - Y_sorted), axis=0)
            
            return np.mean(distances)
        else:
            distances = np.array([
                wasserstein_distance(X_proj[:, i], Y_proj[:, i])
                for i in range(n_proj)
            ])
            return np.mean(distances)
    
    @property
    def last_method(self):
        return self._last_method


class OptimizedEMDGPU:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
    
    def compute(self, X, Y, n_projections=1000):
        X = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        Y = torch.from_numpy(np.asarray(Y, dtype=np.float32)).to(self.device)
        
        d = X.shape[1]
        n_proj = min(n_projections, d * 10)
        
        thetas = torch.randn(n_proj, d, device=self.device)
        thetas = thetas / thetas.norm(dim=1, keepdim=True)
        
        X_proj = X @ thetas.T
        Y_proj = Y @ thetas.T
        
        X_sorted, _ = torch.sort(X_proj, dim=0)
        Y_sorted, _ = torch.sort(Y_proj, dim=0)
        
        if len(X) == len(Y):
            emd = torch.mean(torch.abs(X_sorted - Y_sorted)).item()
        else:
            n = max(len(X), len(Y))
            X_interp = F.interpolate(
                X_sorted.T.unsqueeze(0), size=n, mode='linear', align_corners=False
            ).squeeze(0).T
            Y_interp = F.interpolate(
                Y_sorted.T.unsqueeze(0), size=n, mode='linear', align_corners=False
            ).squeeze(0).T
            emd = torch.mean(torch.abs(X_interp - Y_interp)).item()
        
        return emd


class EMDCalculator:
    def __init__(self, model_path, num_bands=19, num_classes=5, emd_method='auto', use_gpu=True):
        self.model_path = model_path
        self.num_bands = num_bands
        self.num_classes = num_classes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.emd_method = emd_method
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        if self.use_gpu:
            self.emd_calculator = OptimizedEMDGPU(self.device)
        else:
            self.emd_calculator = OptimizedEMD(method=emd_method)
        
    def load_model(self):
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            config = checkpoint.get('config', {
                'feature_dim': 256,
                'dropout_rate': 0.15,
                'spectral_attention_reduction': 8,
                'spatial_attention_heads': 8
            })
            
            self.model = SpectralNet(
                num_bands=self.num_bands,
                num_classes=self.num_classes,
                config=config
            )
            
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.model.load_state_dict(state_dict, strict=True)
            self.model.to(self.device)
            self.model.eval()
            
            logging.info(f"Model loaded from {self.model_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            return False
    
    def compute_emd(self, features1, features2, sample_size=500, n_projections=1000):
        if len(features1) == 0 or len(features2) == 0:
            return 0.0
        
        if len(features1) > sample_size:
            idx = np.random.choice(len(features1), sample_size, replace=False)
            features1 = features1[idx]
        if len(features2) > sample_size:
            idx = np.random.choice(len(features2), sample_size, replace=False)
            features2 = features2[idx]
        
        if self.use_gpu:
            emd = self.emd_calculator.compute(features1, features2, n_projections)
        else:
            emd = self.emd_calculator.compute(features1, features2, n_projections=n_projections)
        
        return float(emd)
    
    def compute_dataset_emd(self, source_data, target_data, output_path,
                            max_samples=1000, batch_size=16):
        if not self.load_model():
            return None
        
        source_dataset = TensorDataset(source_data['spectral'], source_data['labels'])
        target_dataset = TensorDataset(target_data['spectral'], target_data['labels'])
        
        source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=4, pin_memory=True)
        target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=4, pin_memory=True)
        
        extractor = FeatureExtractor(self.model, self.device)
        extractor.register_hooks()
        
        logging.info("Extracting source features...")
        source_features = extractor.extract(source_loader, max_samples)
        
        logging.info("Extracting target features...")
        target_features = extractor.extract(target_loader, max_samples)
        
        extractor.cleanup()
        
        feature_names = ['input_normalized', 'spectral_attended', 'cnn_features',
                        'spatial_features', 'fused_features', 'pooled_features']
        
        emd_results = {}
        
        for name in feature_names:
            src = source_features.get(name, np.array([]))
            tgt = target_features.get(name, np.array([]))
            
            if len(src) > 0 and len(tgt) > 0:
                logging.info(f"Computing EMD for {name}...")
                start_time = time.time()
                
                emd = self.compute_emd(src, tgt, sample_size=500, n_projections=1000)
                
                elapsed = time.time() - start_time
                
                method_used = 'gpu' if self.use_gpu else getattr(
                    self.emd_calculator, 'last_method', 'sliced'
                )
                
                emd_results[name] = {
                    'emd_distance': float(emd),
                    'source_samples': int(src.shape[0]),
                    'target_samples': int(tgt.shape[0]),
                    'feature_dim': int(src.shape[1]),
                    'computation_time': round(elapsed, 4),
                    'method': method_used
                }
                
                logging.info(f"  {name}: EMD={emd:.4f} ({elapsed:.3f}s)")
        
        results = {
            'emd_results': emd_results,
            'model_path': self.model_path,
            'source_samples': int(source_data['spectral'].shape[0]),
            'target_samples': int(target_data['spectral'].shape[0]),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'emd_method': self.emd_method,
                'use_gpu': self.use_gpu,
                'has_pot': HAS_POT,
                'has_numba': HAS_NUMBA
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Results saved to {output_path}")
        return results


def load_data(file_path):
    try:
        if file_path.endswith('.pt'):
            data = torch.load(file_path)
            spectral_tensor = data['spectral']
            labels_tensor = data['labels']
            
            if not isinstance(spectral_tensor, torch.Tensor):
                spectral_tensor = torch.from_numpy(spectral_tensor).float()
            if not isinstance(labels_tensor, torch.Tensor):
                labels_tensor = torch.from_numpy(labels_tensor).long()
        else:
            data = np.load(file_path, allow_pickle=True)
            
            spectral_data = None
            labels_data = None
            
            for key in data.keys():
                if key in ['X', 'spectral'] or 'spectral' in key.lower():
                    spectral_data = data[key]
                elif key in ['y', 'labels'] or 'label' in key.lower():
                    labels_data = data[key]
            
            if spectral_data is None or labels_data is None:
                logging.error(f"Required data not found. Keys: {list(data.keys())}")
                return None
            
            spectral_tensor = torch.from_numpy(spectral_data).float()
            labels_tensor = torch.from_numpy(labels_data).long()
        
        if labels_tensor.dim() > 1 and labels_tensor.shape[1] > 1:
            labels_tensor = torch.argmax(labels_tensor, dim=1)
        
        logging.info(f"Loaded {file_path}: {spectral_tensor.shape}")
        return {'spectral': spectral_tensor, 'labels': labels_tensor}
        
    except Exception as e:
        logging.error(f"Failed to load {file_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Compute EMD between datasets')
    
    parser.add_argument('--source-data', required=True)
    parser.add_argument('--target-data', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--output-path', default='emd_analysis.json')
    parser.add_argument('--num-bands', type=int, default=19)
    parser.add_argument('--num-classes', type=int, default=5)
    parser.add_argument('--max-samples', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--emd-method', choices=['auto', 'exact', 'sinkhorn', 'sliced'], default='auto')
    parser.add_argument('--no-gpu', action='store_true')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info(f"POT available: {HAS_POT}")
    logging.info(f"Numba available: {HAS_NUMBA}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    
    source_data = load_data(args.source_data)
    target_data = load_data(args.target_data)
    
    if source_data is None or target_data is None:
        logging.error("Failed to load data")
        return 1
    
    calculator = EMDCalculator(
        args.model_path,
        args.num_bands,
        args.num_classes,
        emd_method=args.emd_method,
        use_gpu=not args.no_gpu
    )
    
    results = calculator.compute_dataset_emd(
        source_data, target_data, args.output_path,
        args.max_samples, args.batch_size
    )
    
    if results:
        logging.info("\n=== EMD Summary ===")
        for layer, info in results['emd_results'].items():
            logging.info(f"  {layer}: {info['emd_distance']:.4f} ({info['method']}, {info['computation_time']:.3f}s)")
        return 0
    
    return 1


if __name__ == "__main__":
    exit(main())
