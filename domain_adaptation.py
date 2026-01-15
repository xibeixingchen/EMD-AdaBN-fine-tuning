#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import logging
import argparse
from datetime import datetime
import random
from sklearn.metrics import accuracy_score, f1_score
import warnings

from adaptive_bn import EMDGuidedConfig, EMDAdaBN1d, EMDAdaBN2d, EMDAdaBN3d
from model_components import SpectralNet, load_pretrained_weights

warnings.filterwarnings('ignore')


def set_seed(seed=42):
    """Set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file):
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


class SpectralDataset(Dataset):
    """Spectral image dataset"""
    
    def __init__(self, X_spectral, y):
        self.X_spectral = X_spectral.float()
        self.y = self._process_labels(y) if y is not None else None
        
    def _process_labels(self, y):
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        if y.dim() > 1 and y.shape[1] > 1:
            y = torch.argmax(y, dim=1)
        return y.long()
        
    def __len__(self):
        return len(self.X_spectral)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_spectral[idx], self.y[idx]
        return self.X_spectral[idx], torch.tensor(-1)



def load_npz_dataset(file_path):
    """Load NPZ or PT dataset"""
    try:
        logging.info(f"Loading: {file_path}")
        
        if file_path.endswith('.pt'):
            data = torch.load(file_path)
            spectral_tensor = data['spectral']
            labels_tensor = data['labels']
            
            if not isinstance(spectral_tensor, torch.Tensor):
                spectral_tensor = torch.from_numpy(spectral_tensor).float()
            if not isinstance(labels_tensor, torch.Tensor):
                labels_tensor = torch.from_numpy(labels_tensor).long()
            
            logging.info(f"Loaded: {spectral_tensor.shape}")
            return spectral_tensor, labels_tensor
        
        data = np.load(file_path, allow_pickle=True, mmap_mode='r')
        
        spectral_data = None
        labels_data = None
        
        for key in data.keys():
            if key in ['X', 'spectral'] or 'spectral' in key.lower():
                spectral_data = data[key]
            elif key in ['y', 'labels'] or 'label' in key.lower():
                labels_data = data[key]
        
        if spectral_data is None or labels_data is None:
            raise ValueError(f"Data not found. Keys: {list(data.keys())}")
        
        logging.info("Loading to memory...")
        spectral_tensor = torch.from_numpy(np.array(spectral_data)).float()
        labels_tensor = torch.from_numpy(np.array(labels_data))
        
        logging.info(f"Loaded: {spectral_tensor.shape}")
        return spectral_tensor, labels_tensor
        
    except Exception as e:
        logging.error(f"Load failed: {e}")
        return None, None


class FewShotSampler:
    """Few-shot data sampler"""
    
    def __init__(self, X, y, samples_per_class, test_samples_per_class=200,
                 num_classes=5, seed=42):
        self.X = X
        self.y = y
        self.samples_per_class = samples_per_class
        self.test_samples_per_class = test_samples_per_class
        self.num_classes = num_classes
        self.seed = seed
        
    def sample(self):
        """Sample train and test data"""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        labels = self.y
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            labels = torch.argmax(labels, dim=1)
        
        train_indices = []
        test_indices = []
        
        for cls in range(self.num_classes):
            cls_idx = torch.where(labels == cls)[0]
            if len(cls_idx) == 0:
                continue
            
            perm = cls_idx[torch.randperm(len(cls_idx))]
            
            # Train
            n_train = min(self.samples_per_class, len(perm))
            train_indices.extend(perm[:n_train].tolist())
            
            # Test
            remaining = perm[n_train:]
            n_test = min(self.test_samples_per_class, len(remaining))
            if n_test > 0:
                test_indices.extend(remaining[:n_test].tolist())
        
        train_idx = torch.tensor(train_indices)
        test_idx = torch.tensor(test_indices)
        
        train_data = {'spectral': self.X[train_idx], 'labels': labels[train_idx]}
        test_data = None
        if len(test_idx) > 0:
            test_data = {'spectral': self.X[test_idx], 'labels': labels[test_idx]}
        
        return train_data, test_data


class DomainAdapter:
    """Domain adaptation controller - OPTIMIZED"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        self.emd_config = EMDGuidedConfig(
            emd_threshold=args.emd_threshold,      
            linear_factor=args.emd_linear_factor,  
            max_strength=args.emd_max_strength     
        )
        logging.info(f"Device: {self.device}")
        logging.info(f"EMD Config: threshold={args.emd_threshold}, "
                    f"linear_factor={args.emd_linear_factor}, "
                    f"max_strength={args.emd_max_strength}")
        
    def load_emd_values(self):
        """Load EMD values from file or use defaults"""
        if self.args.emd_file and os.path.exists(self.args.emd_file):
            try:
                with open(self.args.emd_file, 'r', encoding='utf-8') as f:
                    emd_data = json.load(f)
                
                emd_results = emd_data.get('emd_results', {})
                emd_values = {}
                
                for k, v in emd_results.items():
                    if isinstance(v, dict) and 'emd_distance' in v:
                        emd_values[k] = v['emd_distance']
                    else:
                        emd_values[k] = float(v)
                
                logging.info(f"Loaded EMD from {self.args.emd_file}")
                
              
                logging.info("Layer adaptation strengths:")
                for layer, emd in sorted(emd_values.items(), key=lambda x: x[1], reverse=True):
                    strength = min(1.0, self.args.emd_linear_factor * emd)
                    status = "ADAPT" if emd > self.args.emd_threshold else "SKIP"
                    logging.info(f"  {layer:20s}: EMD={emd:6.2f}, α={strength:.3f} [{status}]")
                
                return emd_values
                
            except Exception as e:
                logging.warning(f"Failed to load EMD file: {e}")
        
        # Default values
        return {
            'input_normalized': 1.0,
            'spectral_attended': 2.5,
            'cnn_features': 3.2,
            'spatial_features': 4.1,
            'fused_features': 3.5,
            'pooled_features': 5.8
        }
    
    def load_source_model(self):
        """Load pretrained source model"""
        try:
            checkpoint = torch.load(self.args.source_model, map_location=self.device, 
                                   weights_only=False)
            
            config = checkpoint.get('config', {
                'feature_dim': 256,
                'dropout_rate': 0.15,
                'spectral_attention_reduction': 8,
                'spatial_attention_heads': 8
            })
            
            model = SpectralNet(
                num_bands=self.args.num_bands,
                num_classes=self.args.num_classes,
                config=config,
                emd_config=self.emd_config
            )
            
            loaded, total = load_pretrained_weights(model, self.args.source_model, self.device)
            logging.info(f"Loaded {loaded}/{total} parameters")
            
            model.to(self.device)
            model.save_source_statistics()
            
            logging.info("Source model loaded")
            return model
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def adaptive_alignment(self, model, target_loader):
        """Adaptive BN alignment - OPTIMIZED"""
        model.eval()
        logging.info(f"Starting adaptive alignment (rounds={self.args.adabn_rounds}, "
                    f"batches={self.args.adabn_batches})...")
        
        for r in range(self.args.adabn_rounds):
            batch_count = 0
            for inputs, _ in target_loader:
                if batch_count >= self.args.adabn_batches:
                    break
                
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    _ = model(inputs, adapt=True)
                batch_count += 1
            
            if (r + 1) % 3 == 0:
                logging.info(f"  Alignment round {r+1}/{self.args.adabn_rounds} completed")
        
        logging.info("Alignment completed")
    
    def progressive_finetune(self, model, train_loader):
        """Progressive fine-tuning with staged unfreezing"""
        logging.info("Starting progressive fine-tuning...")
        
        criterion = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)
        
        # Stage 1: Classifier only
        logging.info("Stage 1: Fine-tuning classifier only")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        optimizer1 = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.args.lr * self.args.ft_lr_multiplier,
            weight_decay=self.args.weight_decay * 0.5
        )
        
        self._train_stage(model, train_loader, optimizer1, criterion, 
                         epochs=self.args.ft_stage1_epochs, stage_name="Stage1")
        
        # Stage 2: Classifier + Spatial Processor
        if self.args.ft_stage2_epochs > 0:
            logging.info("Stage 2: Fine-tuning classifier + spatial_processor")
            for param in model.spatial_processor.parameters():
                param.requires_grad = True
            
            optimizer2 = optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=self.args.lr * (self.args.ft_lr_multiplier * 0.8),
                weight_decay=self.args.weight_decay
            )
            
            self._train_stage(model, train_loader, optimizer2, criterion,
                            epochs=self.args.ft_stage2_epochs, stage_name="Stage2")
        
        # Stage 3: All layers (optional)
        if self.args.ft_stage3_epochs > 0:
            logging.info("Stage 3: Fine-tuning all layers")
            for param in model.parameters():
                param.requires_grad = True
            
            optimizer3 = optim.AdamW(
                model.parameters(),
                lr=self.args.lr * (self.args.ft_lr_multiplier * 0.5),
                weight_decay=self.args.weight_decay * 1.5
            )
            
            # 使用cosine annealing
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer3,
                T_0=5,
                T_mult=2,
                eta_min=self.args.lr * 0.01
            )
            
            self._train_stage(model, train_loader, optimizer3, criterion,
                            epochs=self.args.ft_stage3_epochs, stage_name="Stage3",
                            scheduler=scheduler)
        
        logging.info("Progressive fine-tuning completed")
    
    def _train_stage(self, model, train_loader, optimizer, criterion, epochs, 
                     stage_name="", scheduler=None):
        """Train a single stage"""
        model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=self.args.gradient_clip
                )
                optimizer.step()
                
                if scheduler is not None:
                    scheduler.step()
                
                total_loss += loss.item()
                _, pred = outputs.max(1)
                total += targets.size(0)
                correct += pred.eq(targets).sum().item()
            
            if epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1:
                acc = 100. * correct / total
                avg_loss = total_loss / len(train_loader)
                logging.info(f"  {stage_name} Epoch {epoch+1:2d}/{epochs}: "
                           f"Loss={avg_loss:.4f}, Acc={acc:.2f}%")
    
    def evaluate(self, model, test_loader):
        """Evaluate model"""
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                _, preds = outputs.max(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.numpy())
        
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        
        return {'accuracy': accuracy, 'f1_score': f1}
    
    def run_single_experiment(self, target_data, samples_per_class):
        """Run single experiment"""
        logging.info(f"\n{'='*60}")
        logging.info(f"Experiment: {samples_per_class} samples/class")
        logging.info(f"{'='*60}")
        
        emd_values = self.load_emd_values()
        
        model = self.load_source_model()
        if model is None:
            return None
        
        model.set_layer_emd_values(emd_values)
        
        # Sample data
        sampler = FewShotSampler(
            target_data['spectral'],
            target_data['labels'],
            samples_per_class,
            test_samples_per_class=200,
            num_classes=self.args.num_classes,
            seed=self.args.seed
        )
        
        train_data, test_data = sampler.sample()
        
        if test_data is None:
            logging.error("Insufficient test data")
            return None
        
        logging.info(f"Train samples: {len(train_data['labels'])}, "
                    f"Test samples: {len(test_data['labels'])}")
        
        batch_size = max(4, min(self.args.batch_size, len(train_data['labels']) // 2))
        
        train_dataset = SpectralDataset(train_data['spectral'], train_data['labels'])
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2,
            drop_last=True if len(train_dataset) > batch_size else False
        )
        
        test_dataset = SpectralDataset(test_data['spectral'], test_data['labels'])
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            num_workers=2
        )
        
        # Adapt and evaluate
        self.adaptive_alignment(model, train_loader)
        self.progressive_finetune(model, train_loader)
        results = self.evaluate(model, test_loader)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"RESULTS: Accuracy={results['accuracy']:.4f}, F1={results['f1_score']:.4f}")
        logging.info(f"{'='*60}\n")
        
        return {
            'samples_per_class': samples_per_class,
            'accuracy': results['accuracy'],
            'f1_score': results['f1_score'],
            'emd_values': emd_values
        }
    
    def run_experiments(self, target_data):
        """Run all experiments"""
        all_results = []
        
        for samples in self.args.samples_list:
            run_results = []
            
            for run in range(self.args.num_runs):
                logging.info(f"\n{'#'*70}")
                logging.info(f"# Run {run+1}/{self.args.num_runs} with {samples} samples/class")
                logging.info(f"{'#'*70}\n")
                
                set_seed(self.args.seed + run)
                result = self.run_single_experiment(target_data, samples)
                
                if result:
                    run_results.append(result)
                
                torch.cuda.empty_cache()
            
            if run_results:
                avg_acc = np.mean([r['accuracy'] for r in run_results])
                std_acc = np.std([r['accuracy'] for r in run_results])
                avg_f1 = np.mean([r['f1_score'] for r in run_results])
                std_f1 = np.std([r['f1_score'] for r in run_results])
                
                all_results.append({
                    'samples_per_class': samples,
                    'accuracy_mean': float(avg_acc),
                    'accuracy_std': float(std_acc),
                    'f1_mean': float(avg_f1),
                    'f1_std': float(std_f1),
                    'num_runs': len(run_results),
                    'individual_runs': [
                        {'accuracy': r['accuracy'], 'f1_score': r['f1_score']}
                        for r in run_results
                    ]
                })
                
                logging.info(f"\n{'='*70}")
                logging.info(f"SUMMARY for {samples} samples/class:")
                logging.info(f"  Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
                logging.info(f"  F1-Score: {avg_f1:.4f} ± {std_f1:.4f}")
                logging.info(f"{'='*70}\n")
        
        return all_results
    
    def save_results(self, results):
        """Save results"""
        output = {
            'results': results,
            'config': {
                'source_model': self.args.source_model,
                'target_data': self.args.target_data,
                'emd_threshold': self.args.emd_threshold,
                'emd_linear_factor': self.args.emd_linear_factor,
                'emd_max_strength': self.args.emd_max_strength,
                'adabn_rounds': self.args.adabn_rounds,
                'adabn_batches': self.args.adabn_batches,
                'ft_lr_multiplier': self.args.ft_lr_multiplier,
                'ft_stage1_epochs': self.args.ft_stage1_epochs,
                'ft_stage2_epochs': self.args.ft_stage2_epochs,
                'ft_stage3_epochs': self.args.ft_stage3_epochs,
                'num_runs': self.args.num_runs,
                'seed': self.args.seed
            },
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save JSON
        output_file = os.path.join(self.args.output_dir, 'results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        
        # Save report
        report_file = os.path.join(self.args.output_dir, 'report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("EMD-Guided Domain Adaptation Results (OPTIMIZED)\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("Configuration:\n")
            for k, v in output['config'].items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")
            
            f.write("Results:\n")
            f.write("-" * 70 + "\n")
            for r in results:
                f.write(f"\nSamples: {r['samples_per_class']}\n")
                f.write(f"  Accuracy: {r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}\n")
                f.write(f"  F1-Score: {r['f1_mean']:.4f} ± {r['f1_std']:.4f}\n")
                f.write(f"  Runs: {r['num_runs']}\n")
        
        logging.info(f"Results saved to {self.args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='EMD-guided Domain Adaptation (OPTIMIZED)')
    
    # Required arguments
    parser.add_argument('--source-model', required=True, help='Source model path')
    parser.add_argument('--target-data', required=True, help='Target data path')
    parser.add_argument('--emd-file', default=None, help='EMD file path')
    
    # Model config
    parser.add_argument('--num-classes', type=int, default=5)
    parser.add_argument('--num-bands', type=int, default=19)
    
    # Experiment config
    parser.add_argument('--samples-list', nargs='+', type=int, default=[50, 100, 200])
    parser.add_argument('--num-runs', type=int, default=3)
    
    # EMD-guided adaptation config (OPTIMIZED)
    parser.add_argument('--emd-threshold', type=float, default=3.5,
                       help='EMD threshold for layer selection (default: 3.5)')
    parser.add_argument('--emd-linear-factor', type=float, default=0.35,
                       help='Linear factor for adaptation strength (default: 0.35)')
    parser.add_argument('--emd-max-strength', type=float, default=1.0,
                       help='Maximum adaptation strength (default: 1.0)')
    
    # AdaBN config (OPTIMIZED)
    parser.add_argument('--adabn-rounds', type=int, default=10,
                       help='AdaBN alignment rounds (default: 10)')
    parser.add_argument('--adabn-batches', type=int, default=12,
                       help='Batches per AdaBN round (default: 12)')
    
    # Fine-tuning config (OPTIMIZED)
    parser.add_argument('--ft-lr-multiplier', type=float, default=5.0,
                       help='Fine-tuning LR multiplier (default: 5.0)')
    parser.add_argument('--ft-stage1-epochs', type=int, default=10,
                       help='Stage 1 (classifier) epochs (default: 10)')
    parser.add_argument('--ft-stage2-epochs', type=int, default=15,
                       help='Stage 2 (classifier+spatial) epochs (default: 15)')
    parser.add_argument('--ft-stage3-epochs', type=int, default=15,
                       help='Stage 3 (all layers) epochs (default: 15)')
    
    # Training config
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--label-smoothing', type=float, default=0.15)
    parser.add_argument('--gradient-clip', type=float, default=0.5)
    
    # Output config
    parser.add_argument('--output-dir', default='./results_optimized')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"adaptation_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    setup_logging(os.path.join(args.output_dir, 'adaptation.log'))
    
    # Log configuration
    logging.info("="*70)
    logging.info("EMD-GUIDED DOMAIN ADAPTATION - OPTIMIZED VERSION")
    logging.info("="*70)
    logging.info("\nConfiguration:")
    for arg, value in sorted(vars(args).items()):
        logging.info(f"  {arg}: {value}")
    logging.info("")
    
    try:
        adapter = DomainAdapter(args)
        
        # Load target data
        spectral, labels = load_npz_dataset(args.target_data)
        if spectral is None:
            raise ValueError("Failed to load target data")
        
        target_data = {'spectral': spectral, 'labels': labels}
        logging.info(f"Target data loaded: {len(labels)} samples\n")
        
        # Run experiments
        results = adapter.run_experiments(target_data)
        
        # Save results
        adapter.save_results(results)
        
        # Final summary
        logging.info("\n" + "="*70)
        logging.info("FINAL SUMMARY")
        logging.info("="*70)
        for r in results:
            logging.info(f"Samples {r['samples_per_class']:3d}: "
                        f"Acc={r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f}, "
                        f"F1={r['f1_mean']:.4f}±{r['f1_std']:.4f}")
        logging.info("="*70)
        
    except Exception as e:
        logging.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
