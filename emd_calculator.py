#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Earth Mover's Distance (EMD) 计算器
计算不同数据集间各层特征的EMD距离
"""

import torch
import torch.nn as nn
import numpy as np
import json
import logging
import argparse
from pathlib import Path
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader, Dataset
import time


class FeatureExtractor(nn.Module):
    """特征提取器，用于获取各层的特征表示"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.features = {}
        self.hooks = []
        
    def register_hooks(self):
        """注册钩子函数获取中间层特征"""
        
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, (list, tuple)):
                    output = output[0]
                # 将特征展平并移到CPU
                if len(output.shape) > 2:
                    output = output.view(output.size(0), -1)
                self.features[name] = output.detach().cpu().numpy()
            return hook
        
        # 为各个关键层注册钩子
        layer_names = [
            ('input_norm', 'input_normalized'),
            ('spectral_attention', 'spectral_attended'), 
            ('backbone_3d.feature_projection.1', 'cnn_features'),
            ('spatial_processor.pre_adabn', 'spatial_features'),
            ('classifier.1', 'pooled_features')
        ]
        
        for module_name, feature_name in layer_names:
            try:
                module = self._get_module_by_name(self.model, module_name)
                if module is not None:
                    hook = module.register_forward_hook(get_activation(feature_name))
                    self.hooks.append(hook)
                    logging.info(f"注册钩子: {module_name} -> {feature_name}")
            except:
                logging.warning(f"无法为 {module_name} 注册钩子")
    
    def _get_module_by_name(self, model, name):
        """根据名称获取模块"""
        parts = name.split('.')
        module = model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module
    
    def extract_features(self, dataloader, max_samples=1000):
        """提取特征"""
        self.model.eval()
        all_features = {name: [] for name in ['input_normalized', 'spectral_attended', 
                                            'cnn_features', 'spatial_features', 'pooled_features']}
        
        sample_count = 0
        
        with torch.no_grad():
            for inputs, _ in dataloader:
                if sample_count >= max_samples:
                    break
                    
                inputs = inputs.cuda() if torch.cuda.is_available() else inputs
                
                # 清空特征字典
                self.features.clear()
                
                # 前向传播
                _ = self.model(inputs, save_source=True)
                
                # 收集特征
                for name in all_features.keys():
                    if name in self.features:
                        all_features[name].append(self.features[name])
                
                sample_count += inputs.size(0)
                
                if sample_count % 100 == 0:
                    logging.info(f"已提取 {sample_count} 个样本的特征")
        
        # 合并特征
        for name in all_features.keys():
            if all_features[name]:
                all_features[name] = np.concatenate(all_features[name], axis=0)
                logging.info(f"{name}: {all_features[name].shape}")
            else:
                logging.warning(f"未获取到 {name} 的特征")
                
        return all_features
    
    def cleanup(self):
        """清理钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class EMDCalculator:
    """EMD距离计算器"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        
    def load_model(self):
        """加载预训练模型"""
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # 创建模型架构（这里需要根据实际模型调整）
            from model_components import SpectralNet
            self.model = SpectralNet(num_bands=19, num_classes=5, feature_dim=256)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            # 加载权重
            self.model.load_state_dict(state_dict, strict=False)
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                
            logging.info(f"模型加载成功: {self.model_path}")
            return True
            
        except Exception as e:
            logging.error(f"模型加载失败: {e}")
            return False
    
    def compute_emd_between_features(self, features1, features2, sample_size=500):
        """计算两组特征间的EMD距离"""
        
        if features1.shape[0] == 0 or features2.shape[0] == 0:
            return 0.0
            
        # 随机采样以减少计算量
        if features1.shape[0] > sample_size:
            idx1 = np.random.choice(features1.shape[0], sample_size, replace=False)
            features1 = features1[idx1]
            
        if features2.shape[0] > sample_size:
            idx2 = np.random.choice(features2.shape[0], sample_size, replace=False)
            features2 = features2[idx2]
        
        # 对于高维特征，先进行PCA降维
        if features1.shape[1] > 100:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50, random_state=42)
            
            combined = np.vstack([features1, features2])
            combined_pca = pca.fit_transform(combined)
            
            features1 = combined_pca[:len(features1)]
            features2 = combined_pca[len(features1):]
        
        # 计算每个维度的EMD并取平均
        emd_distances = []
        
        for dim in range(min(features1.shape[1], 20)):  # 限制维度数量
            try:
                emd = wasserstein_distance(features1[:, dim], features2[:, dim])
                emd_distances.append(emd)
            except:
                continue
        
        return np.mean(emd_distances) if emd_distances else 0.0
    
    def compute_dataset_emd(self, source_data, target_data, output_path):
        """计算数据集间的EMD距离"""
        
        if not self.load_model():
            return None
            
        # 创建数据加载器
        from torch.utils.data import DataLoader, TensorDataset
        
        source_dataset = TensorDataset(source_data['spectral'], source_data['labels'])
        target_dataset = TensorDataset(target_data['spectral'], target_data['labels'])
        
        source_loader = DataLoader(source_dataset, batch_size=16, shuffle=True)
        target_loader = DataLoader(target_dataset, batch_size=16, shuffle=True)
        
        # 创建特征提取器
        extractor = FeatureExtractor(self.model)
        extractor.register_hooks()
        
        # 提取源域和目标域特征
        logging.info("提取源域特征...")
        source_features = extractor.extract_features(source_loader, max_samples=1000)
        
        logging.info("提取目标域特征...")
        target_features = extractor.extract_features(target_loader, max_samples=1000)
        
        # 计算各层EMD距离
        emd_results = {}
        
        for layer_name in source_features.keys():
            if layer_name in target_features and len(source_features[layer_name]) > 0 and len(target_features[layer_name]) > 0:
                logging.info(f"计算 {layer_name} 的EMD距离...")
                
                start_time = time.time()
                emd_dist = self.compute_emd_between_features(
                    source_features[layer_name], 
                    target_features[layer_name]
                )
                elapsed = time.time() - start_time
                
                emd_results[layer_name] = {
                    'emd_distance': float(emd_dist),
                    'source_samples': int(source_features[layer_name].shape[0]),
                    'target_samples': int(target_features[layer_name].shape[0]),
                    'feature_dim': int(source_features[layer_name].shape[1]),
                    'computation_time': elapsed
                }
                
                logging.info(f"{layer_name}: EMD={emd_dist:.4f}, 耗时={elapsed:.2f}s")
        
        # 清理资源
        extractor.cleanup()
        
        # 保存结果
        results = {
            'emd_results': emd_results,
            'model_path': self.model_path,
            'computation_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        logging.info(f"EMD计算完成，结果保存至: {output_path}")
        return results


def load_npz_data(file_path):
    """加载NPZ数据"""
    try:
        data = np.load(file_path, allow_pickle=True)
        
        # 查找数据
        spectral_data = None
        labels_data = None
        
        for key in data.keys():
            if 'spectral' in key.lower() or 'hyper' in key.lower() or key == 'X':
                spectral_data = data[key]
            elif 'label' in key.lower() or 'target' in key.lower() or key == 'y':
                labels_data = data[key]
        
        spectral_tensor = torch.from_numpy(spectral_data).float()
        labels_tensor = torch.from_numpy(labels_data).long()
        
        # 格式转换
        if len(spectral_tensor.shape) == 4 and spectral_tensor.shape[1] > spectral_tensor.shape[3]:
            spectral_tensor = spectral_tensor.permute(0, 3, 1, 2)
        
        return {
            'spectral': spectral_tensor,
            'labels': labels_tensor if labels_tensor.dim() == 1 else torch.argmax(labels_tensor, dim=1)
        }
        
    except Exception as e:
        logging.error(f"数据加载失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='计算数据集间的EMD距离')
    
    parser.add_argument('--source-data', required=True, help='源域数据路径')
    parser.add_argument('--target-data', required=True, help='目标域数据路径')
    parser.add_argument('--model-path', required=True, help='预训练模型路径')
    parser.add_argument('--output-path', default='emd_analysis.json', help='输出路径')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 加载数据
    logging.info("加载数据...")
    source_data = load_npz_data(args.source_data)
    target_data = load_npz_data(args.target_data)
    
    if source_data is None or target_data is None:
        logging.error("数据加载失败")
        return
    
    logging.info(f"源域数据: {source_data['spectral'].shape}")
    logging.info(f"目标域数据: {target_data['spectral'].shape}")
    
    # 计算EMD
    calculator = EMDCalculator(args.model_path)
    results = calculator.compute_dataset_emd(source_data, target_data, args.output_path)
    
    if results:
        logging.info("EMD计算完成!")
        for layer, info in results['emd_results'].items():
            logging.info(f"{layer}: {info['emd_distance']:.4f}")
    else:
        logging.error("EMD计算失败")


if __name__ == "__main__":
    main()