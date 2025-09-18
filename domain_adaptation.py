#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EMD指导的域适应主脚本
整合EMD计算、AdaBN和微调的完整域适应系统
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import json
import logging
import argparse
from datetime import datetime
import time
import random
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings

from adaptive_bn import EMDGuidedConfig, EMDAdaBN2d, EMDAdaBN3d, EMDAdaBN1d
from model_components import SpectralNet
from emd_calculator import EMDCalculator

warnings.filterwarnings('ignore')


def set_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file):
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


class SpectralDataset(Dataset):
    """光谱数据集"""
    
    def __init__(self, X_spectral, y, is_training=True):
        self.X_spectral = X_spectral.float()
        self.y = self._process_labels(y) if y is not None else None
        self.is_training = is_training
        
    def _process_labels(self, y):
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        if y.dim() > 1 and y.shape[1] > 1:
            y = torch.argmax(y, dim=1)
        return y.long()
        
    def __len__(self):
        return len(self.X_spectral)
    
    def __getitem__(self, idx):
        spectral_img = self.X_spectral[idx].clone()
        if self.y is not None:
            return spectral_img, self.y[idx].clone()
        else:
            return spectral_img, torch.tensor(-1)


def load_npz_dataset(file_path):
    """加载NPZ数据集"""
    try:
        logging.info(f"加载数据集: {file_path}")
        data = np.load(file_path, allow_pickle=True)
        
        # 查找光谱数据和标签
        spectral_data = None
        labels_data = None
        
        for key in data.keys():
            if 'spectral' in key.lower() or 'hyper' in key.lower() or key == 'X':
                spectral_data = data[key]
            elif 'label' in key.lower() or 'target' in key.lower() or key == 'y':
                labels_data = data[key]
        
        if spectral_data is None or labels_data is None:
            raise ValueError("未找到必要的数据")
        
        spectral_tensor = torch.from_numpy(spectral_data).float()
        labels_tensor = torch.from_numpy(labels_data)
        
        # 格式转换 [N, H, W, C] -> [N, C, H, W]
        if len(spectral_tensor.shape) == 4 and spectral_tensor.shape[1] > spectral_tensor.shape[3]:
            spectral_tensor = spectral_tensor.permute(0, 3, 1, 2)
        
        logging.info(f"数据集加载成功: {spectral_tensor.shape}")
        return spectral_tensor, labels_tensor
        
    except Exception as e:
        logging.error(f"加载数据集失败: {str(e)}")
        return None, None


class FewShotSampler:
    """小样本采样器"""
    
    def __init__(self, X, y, samples_per_class, test_samples_per_class=200, num_classes=5, seed=42):
        self.X = X
        self.y = y
        self.samples_per_class = samples_per_class
        self.test_samples_per_class = test_samples_per_class
        self.num_classes = num_classes
        self.seed = seed
        
    def sample_data(self):
        """采样训练和测试数据"""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        labels = self.y
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            labels = torch.argmax(labels, dim=1)
        
        train_indices = []
        test_indices = []
        
        for class_label in range(self.num_classes):
            class_indices = torch.where(labels == class_label)[0]
            
            if len(class_indices) == 0:
                continue
            
            shuffled_indices = class_indices[torch.randperm(len(class_indices))]
            
            # 训练样本
            train_count = min(self.samples_per_class, len(shuffled_indices))
            train_class_indices = shuffled_indices[:train_count]
            train_indices.extend(train_class_indices.tolist())
            
            # 测试样本
            remaining_indices = shuffled_indices[train_count:]
            test_count = min(self.test_samples_per_class, len(remaining_indices))
            
            if test_count > 0:
                test_class_indices = remaining_indices[:test_count]
                test_indices.extend(test_class_indices.tolist())
        
        train_indices = torch.tensor(train_indices)
        test_indices = torch.tensor(test_indices)
        
        train_data = {
            'spectral': self.X[train_indices],
            'labels': labels[train_indices]
        }
        
        test_data = None
        if len(test_indices) > 0:
            test_data = {
                'spectral': self.X[test_indices],
                'labels': labels[test_indices]
            }
        
        return train_data, test_data


class DomainAdapter:
    """域适应器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emd_config = EMDGuidedConfig(emd_threshold=1.5, linear_factor=0.15, max_strength=0.8)
        
        logging.info(f"使用设备: {self.device}")
        
    def compute_emd_if_needed(self, source_data, target_data):
        """如果需要，计算EMD距离"""
        emd_file = f"emd_analysis_{int(time.time())}.json"
        
        # 如果没有提供EMD文件，则计算EMD
        if not hasattr(self.args, 'emd_analysis_file') or not os.path.exists(self.args.emd_analysis_file):
            logging.info("计算EMD距离...")
            
            calculator = EMDCalculator(self.args.pretrained_model)
            emd_results = calculator.compute_dataset_emd(source_data, target_data, emd_file)
            
            if emd_results:
                return emd_results['emd_results']
            else:
                logging.warning("EMD计算失败，使用默认值")
                return self._get_default_emd_values()
        else:
            # 加载现有EMD文件
            try:
                with open(self.args.emd_analysis_file, 'r', encoding='utf-8') as f:
                    emd_data = json.load(f)
                
                if 'emd_results' in emd_data:
                    # 从复杂结构中提取EMD值
                    for key, value in emd_data['emd_results'].items():
                        if isinstance(value, dict):
                            return {k: v['emd_distance'] if isinstance(v, dict) and 'emd_distance' in v else v 
                                   for k, v in value.items()}
                    
                return emd_data.get('emd_results', self._get_default_emd_values())
                
            except Exception as e:
                logging.error(f"加载EMD文件失败: {e}")
                return self._get_default_emd_values()
    
    def _get_default_emd_values(self):
        """获取默认EMD值"""
        return {
            'input_normalized': 1.0,
            'spectral_attended': 2.5,
            'cnn_features': 3.2,
            'spatial_features': 4.1,
            'pooled_features': 5.8
        }
    
    def load_pretrained_model(self):
        """加载预训练模型"""
        try:
            checkpoint = torch.load(self.args.pretrained_model, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # 创建模型
            model = SpectralNet(
                num_bands=self.args.num_bands,
                num_classes=self.args.num_classes,
                feature_dim=256,
                emd_config=self.emd_config
            )
            
            # 过滤和加载权重
            model_dict = model.state_dict()
            filtered_state_dict = {}
            
            for k, v in state_dict.items():
                # 跳过分类器层
                if 'classifier' in k:
                    continue
                    
                if k in model_dict and model_dict[k].shape == v.shape:
                    filtered_state_dict[k] = v
            
            model.load_state_dict(filtered_state_dict, strict=False)
            
            # 重新初始化分类器
            for name, module in model.named_modules():
                if 'classifier' in name and isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, 0, 0.001)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            
            model.to(self.device)
            
            # 保存源域统计量
            with torch.no_grad():
                for module in model.modules():
                    if isinstance(module, (EMDAdaBN2d, EMDAdaBN3d, EMDAdaBN1d)):
                        if module.track_running_stats:
                            module.source_mean.copy_(module.running_mean)
                            module.source_var.copy_(module.running_var)
            
            logging.info("预训练模型加载成功")
            return model
            
        except Exception as e:
            logging.error(f"预训练模型加载失败: {e}")
            return None
    
    def adaptive_alignment(self, model, target_loader):
        """自适应对齐"""
        model.eval()
        logging.info("开始自适应对齐...")
        
        alignment_rounds = 5
        batches_per_round = 8
        
        adabn_layers = [(name, module) for name, module in model.named_modules() 
                       if isinstance(module, (EMDAdaBN2d, EMDAdaBN3d, EMDAdaBN1d)) 
                       and module.use_adabn.item()]
        
        logging.info(f"发现 {len(adabn_layers)} 个AdaBN层参与适应")
        
        for round_idx in range(alignment_rounds):
            batch_count = 0
            for inputs, _ in target_loader:
                if batch_count >= batches_per_round:
                    break
                
                inputs = inputs.to(self.device, non_blocking=True)
                
                with torch.no_grad():
                    _ = model(inputs, adapt=True, save_source=False)
                
                batch_count += 1
        
        logging.info("自适应对齐完成")
    
    def finetune_model(self, model, train_loader):
        """微调模型"""
        logging.info("开始微调...")
        
        # 冻结大部分参数，只微调分类器
        for param in model.parameters():
            param.requires_grad = False
        
        # 解冻分类器
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.args.lr * 2.0,  # 分类器使用更高学习率
            weight_decay=0.01
        )
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        model.train()
        finetune_epochs = 15
        
        for epoch in range(finetune_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            if epoch % 3 == 0:
                acc = 100. * correct / total
                logging.info(f"微调 Epoch {epoch+1:2d}/{finetune_epochs}: Loss={epoch_loss/len(train_loader):.4f}, Acc={acc:.2f}%")
        
        logging.info("微调完成")
    
    def evaluate_model(self, model, test_loader):
        """评估模型"""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def run_experiment(self, source_data, target_data, samples_per_class):
        """运行单次实验"""
        logging.info(f"开始实验: 每类{samples_per_class}个样本")
        
        # 计算EMD
        emd_values = self.compute_emd_if_needed(source_data, target_data)
        logging.info(f"EMD值: {emd_values}")
        
        # 加载模型
        model = self.load_pretrained_model()
        if model is None:
            return None
        
        # 设置EMD值
        model.set_layer_emd_values(emd_values)
        
        # 采样数据
        sampler = FewShotSampler(
            target_data['spectral'], target_data['labels'], 
            samples_per_class, test_samples_per_class=200, 
            num_classes=self.args.num_classes
        )
        
        train_data, test_data = sampler.sample_data()
        
        if test_data is None:
            logging.error("测试数据不足")
            return None
        
        # 创建数据加载器
        batch_size = min(self.args.batch_size, len(train_data['labels']) // 2)
        batch_size = max(4, batch_size)
        
        train_dataset = SpectralDataset(train_data['spectral'], train_data['labels'])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        test_dataset = SpectralDataset(test_data['spectral'], test_data['labels'])
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=2)
        
        # 自适应对齐
        self.adaptive_alignment(model, train_loader)
        
        # 微调
        self.finetune_model(model, train_loader)
        
        # 评估
        results = self.evaluate_model(model, test_loader)
        
        logging.info(f"结果 - 准确率: {results['accuracy']:.4f}, F1分数: {results['f1_score']:.4f}")
        
        return {
            'samples_per_class': samples_per_class,
            'accuracy': results['accuracy'],
            'f1_score': results['f1_score'],
            'emd_values': emd_values
        }
    
    def run_experiments(self, source_2022_data, source_2024_data):
        """运行完整实验"""
        results_2022 = []
        results_2024 = []
        
        for samples_per_class in self.args.samples_per_class_list:
            logging.info(f"\n=== 实验: 每类 {samples_per_class} 个样本 ===")
            
            # 2022年实验
            experiment_results_2022 = []
            for run in range(self.args.num_runs):
                set_seed(self.args.seed + run)
                result = self.run_experiment(source_2022_data, source_2022_data, samples_per_class)
                if result:
                    experiment_results_2022.append(result)
                torch.cuda.empty_cache()
            
            # 2024年实验
            experiment_results_2024 = []
            for run in range(self.args.num_runs):
                set_seed(self.args.seed + run)
                result = self.run_experiment(source_2022_data, source_2024_data, samples_per_class)
                if result:
                    experiment_results_2024.append(result)
                torch.cuda.empty_cache()
            
            # 计算平均结果
            if experiment_results_2022:
                avg_acc_2022 = np.mean([r['accuracy'] for r in experiment_results_2022])
                std_acc_2022 = np.std([r['accuracy'] for r in experiment_results_2022])
                results_2022.append({
                    'samples_per_class': samples_per_class,
                    'accuracy_mean': avg_acc_2022,
                    'accuracy_std': std_acc_2022
                })
            
            if experiment_results_2024:
                avg_acc_2024 = np.mean([r['accuracy'] for r in experiment_results_2024])
                std_acc_2024 = np.std([r['accuracy'] for r in experiment_results_2024])
                results_2024.append({
                    'samples_per_class': samples_per_class,
                    'accuracy_mean': avg_acc_2024,
                    'accuracy_std': std_acc_2024
                })
            
            # 输出结果
            if experiment_results_2022 and experiment_results_2024:
                diff = avg_acc_2024 - avg_acc_2022
                logging.info(f"样本数 {samples_per_class}: 2022年={avg_acc_2022:.4f}±{std_acc_2022:.4f}, "
                           f"2024年={avg_acc_2024:.4f}±{std_acc_2024:.4f}, 差异={diff:+.4f}")
        
        return results_2022, results_2024
    
    def save_results(self, results_2022, results_2024):
        """保存结果"""
        results = {
            'results_2022': results_2022,
            'results_2024': results_2024,
            'config': {
                'emd_threshold': self.emd_config.emd_threshold,
                'linear_factor': self.emd_config.linear_factor,
                'max_strength': self.emd_config.max_strength
            }
        }
        
        output_file = os.path.join(self.args.output_dir, 'domain_adaptation_results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 生成报告
        report_file = os.path.join(self.args.output_dir, 'results_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("EMD指导的域适应结果报告\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"{'样本数':>6} | {'2022年准确率':>15} | {'2024年准确率':>15} | {'差异':>10}\n")
            f.write("-" * 60 + "\n")
            
            for r2022, r2024 in zip(results_2022, results_2024):
                samples = r2022['samples_per_class']
                acc_2022 = r2022['accuracy_mean']
                std_2022 = r2022['accuracy_std']
                acc_2024 = r2024['accuracy_mean']
                std_2024 = r2024['accuracy_std']
                diff = acc_2024 - acc_2022
                
                f.write(f"{samples:>6} | {acc_2022:.4f}±{std_2022:.3f} | {acc_2024:.4f}±{std_2024:.3f} | {diff:>+9.4f}\n")
        
        logging.info(f"结果保存至: {self.args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='EMD指导的域适应')
    
    # 数据参数
    parser.add_argument('--source-2022-data', required=True, help='2022年数据路径')
    parser.add_argument('--source-2024-data', required=True, help='2024年数据路径')
    parser.add_argument('--pretrained-model', required=True, help='预训练模型路径')
    parser.add_argument('--emd-analysis-file', help='EMD分析文件路径(可选)')
    
    # 模型参数
    parser.add_argument('--num-classes', type=int, default=5)
    parser.add_argument('--num-bands', type=int, default=19)
    
    # 实验参数
    parser.add_argument('--samples-per-class-list', nargs='+', type=int, default=[50, 100, 200])
    parser.add_argument('--num-runs', type=int, default=3)
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    
    # 输出参数
    parser.add_argument('--output-dir', default='./results')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 设置种子
    set_seed(args.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"domain_adaptation_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    setup_logging(os.path.join(args.output_dir, 'domain_adaptation.log'))
    
    try:
        # 创建域适应器
        adapter = DomainAdapter(args)
        
        # 加载数据
        logging.info("加载数据集...")
        spectral_2022, labels_2022 = load_npz_dataset(args.source_2022_data)
        spectral_2024, labels_2024 = load_npz_dataset(args.source_2024_data)
        
        if spectral_2022 is None or spectral_2024 is None:
            raise ValueError("数据加载失败")
        
        source_2022_data = {'spectral': spectral_2022, 'labels': labels_2022}
        source_2024_data = {'spectral': spectral_2024, 'labels': labels_2024}
        
        logging.info(f"数据加载完成: 2022年{len(labels_2022)}样本, 2024年{len(labels_2024)}样本")
        
        # 运行实验
        logging.info("开始EMD指导的域适应实验...")
        results_2022, results_2024 = adapter.run_experiments(source_2022_data, source_2024_data)
        
        # 保存结果
        adapter.save_results(results_2022, results_2024)
        
        logging.info(f"实验完成! 结果保存在: {args.output_dir}")
        
        # 输出最终结果摘要
        logging.info("\n=== 结果摘要 ===")
        for r2022, r2024 in zip(results_2022, results_2024):
            samples = r2022['samples_per_class']
            acc_2022 = r2022['accuracy_mean']
            acc_2024 = r2024['accuracy_mean']
            diff = acc_2024 - acc_2022
            
            logging.info(f"样本数 {samples}: 2022年={acc_2022:.4f}, 2024年={acc_2024:.4f}, 差异={diff:+.4f}")
        
    except Exception as e:
        logging.error(f"实验失败: {e}")
        import traceback
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    main()