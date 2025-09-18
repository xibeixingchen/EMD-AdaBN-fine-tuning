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

### Basic Usage

```bash
python domain_adaptation.py \
    --source-2022-data path/to/2022_data.npz \
    --source-2024-data path/to/2024_data.npz \
    --pretrained-model path/to/pretrained_model.pt
```

### Full Configuration

```bash
python domain_adaptation.py \
    --source-2022-data path/to/2022_data.npz \
    --source-2024-data path/to/2024_data.npz \
    --pretrained-model path/to/pretrained_model.pt \
    --samples-per-class-list 50 100 200 \
    --num-runs 3 \
    --batch-size 8 \
    --lr 0.0001 \
    --output-dir ./results
```

## File Structure

```
├── emd_calculator.py          # EMD distance computation
├── adaptive_bn.py             # EMD-guided AdaBN implementation
├── model_components.py        # Model architecture with AdaBN
├── domain_adaptation.py       # Main training script
└── README.md
```

## Method

1. **EMD Computation**: Calculate Earth Mover's Distance between source and target feature distributions
2. **Adaptive Strategy**: Use EMD values to determine which layers need AdaBN
3. **Selective Adaptation**: Apply different adaptation strengths based on domain shift magnitude
4. **Progressive Fine-tuning**: Intelligent layer unfreezing guided by EMD analysis

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
- Multispectral images: `[N, C, H, W]` format
- Labels: `[N,]` format

## Results

The framework generates comprehensive results including:
- Performance metrics (accuracy, F1-score, kappa)
- Training curves and confusion matrices
- EMD analysis reports
- Detailed experimental logs

## Citation

If you use this code in your research, please cite:

```bibtex
@article{emd_adabn2024,
  title={Climate-Resilient Evaluation of Alfalfa Seed Maturity Using an Earth Mover's Distance-Guided Multispectral Imaging Framework},
  author={Zhicheng Jia},
  journal={Under Peer Review},
  year={2024}
}
```

## License

MIT License

## Contact

For questions about implementation or agricultural applications, please open an issue on this repository.
