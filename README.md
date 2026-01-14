# EMD-AdaBN Fine-tuning
Earth Mover's Distance guided Adaptive Batch Normalization for domain adaptation in multispectral seed maturity classification.

## Overview
This repository implements an EMD-guided domain adaptation framework that uses Adaptive Batch Normalization (AdaBN) for few-shot transfer learning between different agricultural seasons. The method automatically computes Earth Mover's Distance (EMD) between feature distributions and selectively applies AdaBN based on the domain shift magnitude.

## Key Components
- **EMD Calculator**: Computes feature-level domain shift using Earth Mover's Distance
- **Adaptive Batch Normalization**: EMD-guided selective application of AdaBN layers
- **Progressive Fine-tuning**: Multi-stage layer unfreezing based on EMD values
- **Few-shot Learning**: Effective adaptation with limited target domain samples (50-200 per class)

## Quick Start

### Step 1: Compute EMD Distance
```bash
# Compute EMD from source (2023) to target (2022)
python emd_calculator.py \
    --source-data path/to/seed_data_2023.npz \
    --target-data path/to/seed_data_2022.npz \
    --model-path path/to/pretrained_model.pt \
    --output-path emd_2023_to_2022.json \
    --num-bands 19 \
    --num-classes 5 \
    --max-samples 1000 \
    --batch-size 16

# Compute EMD from source (2023) to target (2024)
python emd_calculator.py \
    --source-data path/to/seed_data_2023.npz \
    --target-data path/to/seed_data_2024.npz \
    --model-path path/to/pretrained_model.pt \
    --output-path emd_2023_to_2024.json
```

### Step 2: Domain Adaptation

#### Basic Usage (Default Parameters)
```bash
python domain_adaptation.py \
    --source-model path/to/pretrained_model.pt \
    --target-data path/to/target_data.npz \
    --emd-file emd_2023_to_2022.json
```

## Arguments

### emd_calculator.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--source-data` | str | **Required** | Source domain data path (.npz format) |
| `--target-data` | str | **Required** | Target domain data path (.npz format) |
| `--model-path` | str | **Required** | Pretrained model checkpoint path (.pt/.pth) |
| `--output-path` | str | `emd_analysis.json` | Output JSON file path for EMD results |
| `--num-bands` | int | `19` | Number of spectral bands in input data |
| `--num-classes` | int | `5` | Number of classification classes |
| `--max-samples` | int | `1000` | Maximum samples to use for EMD computation |
| `--batch-size` | int | `16` | Batch size for feature extraction |
| `--seed` | int | `42` | Random seed for reproducibility |

**Output**: JSON file containing EMD distances for each layer:
```json
{
  "emd_results": {
    "input_normalized": {"emd_distance": 13.69, "source_samples": 1000, ...},
    "spectral_attended": {"emd_distance": 6.43, ...},
    "cnn_features": {"emd_distance": 1.82, ...},
    "spatial_features": {"emd_distance": 17.59, ...},
    "fused_features": {"emd_distance": 18.36, ...},
    "pooled_features": {"emd_distance": 0.68, ...}
  }
}
```

### domain_adaptation.py

#### Required Arguments
| Argument | Type | Description |
|----------|------|-------------|
| `--source-model` | str | Pretrained source model path (.pt/.pth) |
| `--target-data` | str | Target domain data path (.npz) |

#### Model Configuration
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num-classes` | int | `5` | Number of classification classes |
| `--num-bands` | int | `19` | Number of spectral bands |

#### EMD-Guided Adaptation Parameters
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--emd-file` | str | `None` | Precomputed EMD file (uses defaults if not provided) |
| `--emd-threshold` | float | `3.5` | **EMD threshold for layer selection**. Only layers with EMD > threshold will be adapted. Higher = more selective. Range: [1.5, 5.0] |
| `--emd-linear-factor` | float | `0.35` | **Adaptation strength coefficient**. α = linear_factor × EMD. Higher = stronger adaptation. Range: [0.15, 0.45] |
| `--emd-max-strength` | float | `1.0` | **Maximum adaptation strength cap**. Limits α to this value. Range: [0.8, 1.0] |

**EMD Guidance Formula**: 
```
α_l = min(emd_max_strength, emd_linear_factor × EMD_l)
μ_adapted = (1-α)·μ_source + α·μ_target
σ²_adapted = (1-α)·σ²_source + α·σ²_target
```

#### AdaBN Configuration
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--adabn-rounds` | int | `10` | **Number of AdaBN alignment rounds**. More rounds = better statistics estimation. Range: [5, 15] |
| `--adabn-batches` | int | `12` | **Batches per AdaBN round**. Total batches = rounds × batches. Range: [8, 15] |

#### Progressive Fine-tuning Configuration
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ft-lr-multiplier` | float | `5.0` | **Fine-tuning learning rate multiplier**. actual_lr = base_lr × multiplier. Higher = faster adaptation. Range: [2.0, 7.0] |
| `--ft-stage1-epochs` | int | `10` | **Stage 1 epochs** (classifier only). Range: [5, 15] |
| `--ft-stage2-epochs` | int | `15` | **Stage 2 epochs** (classifier + spatial_processor). Range: [10, 20] |
| `--ft-stage3-epochs` | int | `15` | **Stage 3 epochs** (all layers). Range: [10, 20] |

**Progressive Fine-tuning Stages**:

#### Training Hyperparameters
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch-size` | int | `16` | Training batch size. Larger = more stable. Range: [8, 32] |
| `--lr` | float | `0.0002` | Base learning rate. Stage 1 uses lr × ft_lr_multiplier |
| `--weight-decay` | float | `0.01` | L2 regularization strength. Higher = less overfitting. Range: [0.005, 0.02] |
| `--label-smoothing` | float | `0.15` | Label smoothing factor. Higher = more regularization. Range: [0.1, 0.2] |
| `--gradient-clip` | float | `0.5` | Gradient clipping max norm. Lower = more stable. Range: [0.5, 1.5] |

#### Experiment Configuration
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--samples-list` | int[] | `[50, 100, 200]` | List of samples per class to evaluate |
| `--num-runs` | int | `3` | Number of runs per experiment (for mean±std) |
| `--output-dir` | str | `./results` | Output directory for results and logs |
| `--seed` | int | `42` | Random seed for reproducibility |



## Requirements
```
Python >= 3.8
PyTorch >= 2.0
numpy >= 1.20
scipy >= 1.7
scikit-learn >= 1.0
```

## Installation
```bash
git clone https://github.com/xibeixingchen/EMD-AdaBN-fine-tuning.git
cd EMD-AdaBN-fine-tuning
pip install torch torchvision numpy scipy scikit-learn
```

## Data Format

### Input NPZ File Structure
```python
npz_file = {
    'X' or 'spectral': np.ndarray,  # Shape: [N, C, H, W] or [N, H, W, C]
    'y' or 'labels': np.ndarray      # Shape: [N,] or [N, num_classes]
}
```

**Automatic Format Handling**:
- Spectral data: Auto-converts `[N, H, W, C]` → `[N, C, H, W]`
- Labels: Auto-converts one-hot `[N, num_classes]` → `[N,]` via argmax


### Adaptation Results
```
results/
├── adaptation_20260114_210902/
│   ├── results.json           # Detailed results with all runs
│   ├── report.txt             # Human-readable summary
│   └── adaptation.log         # Full training logs
```


## License
MIT License
