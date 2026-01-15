# EMD-AdaBN Fine-tuning

Earth Mover's Distance guided Adaptive Batch Normalization for domain adaptation in multispectral seed maturity classification.

## Overview

EMD-guided domain adaptation framework using Adaptive Batch Normalization (AdaBN) for few-shot transfer learning between agricultural seasons.

**Key Features**:
- EMD-based domain shift measurement
- Selective AdaBN layer adaptation
- Progressive fine-tuning with layer unfreezing
- Few-shot learning (50-200 samples per class)

## Quick Start

### Step 1: Compute EMD Distance
```bash
python emd_calculator.py \
    --source-data data/seed_2023.npz \
    --target-data data/seed_2022.npz \
    --model-path models/pretrained.pt \
    --output-path emd_results.json
```

### Step 2: Domain Adaptation
```bash
python domain_adaptation.py \
    --source-model models/pretrained.pt \
    --target-data data/seed_2022.npz \
    --emd-file emd_results.json
```

## Arguments

### emd_calculator.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--source-data` | Required | Source domain data (.npz/.pt) |
| `--target-data` | Required | Target domain data (.npz/.pt) |
| `--model-path` | Required | Pretrained model (.pt) |
| `--output-path` | `emd_analysis.json` | Output JSON path |
| `--num-bands` | `19` | Spectral bands |
| `--num-classes` | `5` | Number of classes |
| `--max-samples` | `1000` | Max samples for EMD |
| `--batch-size` | `32` | Batch size |
| `--emd-method` | `auto` | Algorithm: auto/exact/sinkhorn/sliced |
| `--no-gpu` | `False` | Disable GPU acceleration |

**Output**:
```json
{
  "emd_results": {
    "cnn_features": {"emd_distance": 1.82, "method": "gpu", ...},
    "spatial_features": {"emd_distance": 17.59, ...},
    "pooled_features": {"emd_distance": 0.68, ...}
  }
}
```

### domain_adaptation.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--source-model` | Required | Pretrained model path |
| `--target-data` | Required | Target domain data |
| `--emd-file` | `None` | Precomputed EMD file |
| `--emd-threshold` | `3.5` | Layer selection threshold |
| `--emd-linear-factor` | `0.35` | Adaptation strength factor |
| `--adabn-rounds` | `10` | AdaBN alignment rounds |
| `--ft-stage1-epochs` | `10` | Classifier-only epochs |
| `--ft-stage2-epochs` | `15` | +Spatial processor epochs |
| `--ft-stage3-epochs` | `15` | All layers epochs |
| `--batch-size` | `16` | Training batch size |
| `--lr` | `0.0002` | Base learning rate |
| `--samples-list` | `[50,100,200]` | Samples per class to test |

**EMD-Guided Adaptation**:
```
α = min(max_strength, linear_factor × EMD)
μ_adapted = (1-α)·μ_source + α·μ_target
```

## Data Format

```python
# NPZ structure
{
    'X' or 'spectral': ndarray,  # [N, C, H, W]
    'y' or 'labels': ndarray     # [N,]
}
```

## Installation

```bash
pip install torch numpy scipy scikit-learn POT numba
```

## License

MIT License
