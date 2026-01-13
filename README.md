# EMD-AdaBN Fine-tuning

Earth Mover's Distance guided Adaptive Batch Normalization for domain adaptation in multispectral seed maturity classification.

## Overview

This repository implements an EMD-guided domain adaptation framework that uses Adaptive Batch Normalization (AdaBN) for few-shot transfer learning between different agricultural seasons. The method automatically computes Earth Mover's Distance (EMD) between feature distributions and selectively applies AdaBN based on the domain shift magnitude.

## Key Components

- **EMD Calculator**: Computes feature-level domain shift using Earth Mover's Distance
- **Adaptive Batch Normalization**: EMD-guided selective application of AdaBN layers
- **Smart Fine-tuning**: Intelligent layer unfreezing based on EMD values
- **Few-shot Learning**: Effective adaptation with limited target domain samples

## Quick Start

### Step 1: Compute EMD Distance

```bash
# Compute EMD from source (2023) to target (2022)
python emd_calculator.py \
    --source-data path/to/seed_data_2023.npz \
    --target-data path/to/seed_data_2022.npz \
    --model-path path/to/pretrained_model.pt \
    --output-path emd_2023_to_2022.json

# Compute EMD from source (2023) to target (2024)
python emd_calculator.py \
    --source-data path/to/seed_data_2023.npz \
    --target-data path/to/seed_data_2024.npz \
    --model-path path/to/pretrained_model.pt \
    --output-path emd_2023_to_2024.json
```

### Step 2: Domain Adaptation

```bash
# Basic usage
python domain_adaptation.py \
    --source-model path/to/pretrained_model.pt \
    --target-data path/to/target_data.npz

# With precomputed EMD file
python domain_adaptation.py \
    --source-model path/to/pretrained_model.pt \
    --target-data path/to/seed_data_2022.npz \
    --emd-file emd_2023_to_2022.json
```

### Full Configuration

```bash
python domain_adaptation.py \
    --source-model path/to/pretrained_model.pt \
    --target-data path/to/target_data.npz \
    --emd-file path/to/emd_analysis.json \
    --num-classes 5 \
    --num-bands 19 \
    --samples-list 50 100 200 \
    --num-runs 3 \
    --batch-size 8 \
    --lr 0.0001 \
    --output-dir ./results \
    --seed 42
```

## Arguments

### emd_calculator.py

| Argument | Description | Required |
|----------|-------------|----------|
| `--source-data` | Source domain data path (.npz) | Yes |
| `--target-data` | Target domain data path (.npz) | Yes |
| `--model-path` | Pretrained model path (.pt) | Yes |
| `--output-path` | Output JSON path | No (default: emd_analysis.json) |

### domain_adaptation.py

| Argument | Description | Required |
|----------|-------------|----------|
| `--source-model` | Pretrained source model path (.pt) | Yes |
| `--target-data` | Target domain data path (.npz) | Yes |
| `--emd-file` | Precomputed EMD file path (.json) | No |
| `--num-classes` | Number of classes | No (default: 5) |
| `--num-bands` | Number of spectral bands | No (default: 19) |
| `--samples-list` | Samples per class to test | No (default: 50 100 200) |
| `--num-runs` | Runs per experiment | No (default: 3) |
| `--batch-size` | Batch size | No (default: 8) |
| `--lr` | Learning rate | No (default: 0.0001) |
| `--output-dir` | Output directory | No (default: ./results) |
| `--seed` | Random seed | No (default: 42) |

## File Structure

```
├── emd_calculator.py          # EMD distance computation
├── adaptive_bn.py             # EMD-guided AdaBN implementation
├── model_components.py        # Model architecture with AdaBN
├── domain_adaptation.py       # Main adaptation script
└── README.md
```

## Method

1. **EMD Computation**: Calculate Earth Mover's Distance between source and target feature distributions at each layer
2. **Adaptive Strategy**: Use EMD values to determine which layers need AdaBN (threshold-based)
3. **Selective Adaptation**: Apply different adaptation strengths based on domain shift magnitude
4. **Progressive Fine-tuning**: Classifier fine-tuning with frozen backbone

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, SciPy, scikit-learn
- CUDA (optional, for GPU acceleration)

## Installation

```bash
git clone https://github.com/xibeixingchen/EMD-AdaBN-fine-tuning.git
cd EMD-AdaBN-fine-tuning
pip install torch torchvision numpy scipy scikit-learn
```

## Data Format

Input data should be in NPZ format containing:
- Multispectral images: `[N, C, H, W]` or `[N, H, W, C]` format (auto-converted)
- Labels: `[N,]` format

## Results

The framework generates comprehensive results including:
- Performance metrics (accuracy, F1-score)
- EMD analysis reports (JSON)
- Detailed experimental logs

## Citation

If you use this code in your research, please cite:

```bibtex
@article{emd_adabn2024,
  title={Climate-Resilient Evaluation of Alfalfa Seed Maturity Using an Earth Mover's Distance-Guided Multispectral Imaging Framework},
  author={Zhicheng Jia},
  journal={Under Peer Review},
  year={2026}
}
```

## License

MIT License

## Contact

For questions about implementation or agricultural applications, please open an issue on this repository.
