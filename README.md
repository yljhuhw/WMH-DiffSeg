# MedSegDiff Clean - Medical Image Segmentation with Diffusion Models

This is a clean, organized version of the MedSegDiff project for easy deployment and reproduction.

## Project Structure

```
MedSegDiffClean/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── data/                        # Dataset directory (to be populated)
│   ├── DRIVE/
│   ├── CHASEDB1/
│   └── ISIC/
├── guided_diffusion/            # Core diffusion model library
├── scripts/                     # Training and inference scripts
│   ├── train/                   # Training scripts
│   │   ├── train_medsegdiff.py
│   │   ├── train_wmh.py
│   │   └── train_transunet.py
│   ├── inference/               # Inference scripts
│   │   ├── infer_medsegdiff.py
│   │   ├── infer_wmh.py
│   │   └── infer_transunet.py
│   └── evaluation/              # Evaluation scripts
│       ├── evaluate.py
│       └── compare_methods.py
├── configs/                     # Configuration files
└── docs/                        # Documentation

```

## Quick Start

### 1. Environment Setup

```bash
conda create -n medsegdiff python=3.9
conda activate medsegdiff
pip install -r requirements.txt
```

### 2. Dataset Preparation

Place your datasets in the `data/` directory following the structure above.

### 3. Training

**MedSegDiff:**
```bash
python scripts/train/train_medsegdiff.py --dataset DRIVE
```

**WMH-DiffSeg:**
```bash
python scripts/train/train_wmh.py --dataset DRIVE
```

**TransUNet:**
```bash
python scripts/train/train_transunet.py --dataset DRIVE
```

### 4. Inference

```bash
python scripts/inference/infer_wmh.py --dataset DRIVE --model_path checkpoints/wmh_drive.pt
```

### 5. Evaluation

```bash
python scripts/evaluation/compare_methods.py --dataset DRIVE
```

## Supported Datasets

- **DRIVE**: Retinal vessel segmentation (20 train / 20 test)
- **CHASEDB1**: Retinal vessel segmentation (22 train / 6 test)
- **ISIC**: Skin lesion segmentation (900 train / 379 test)

## Models

1. **MedSegDiff**: Baseline diffusion model for medical image segmentation
2. **WMH-DiffSeg**: Enhanced model with Wavelet Modulation and Hybrid Encoder
3. **TransUNet**: Transformer-based U-Net baseline

## Results

| Method | DRIVE | CHASEDB1 | ISIC |
|--------|-------|----------|------|
| TransUNet | 0.8082 | 0.8144 | 0.8962 |
| MedSegDiff | 0.8405 | 0.8449 | 0.8914 |
| **WMH-DiffSeg** | **0.8739** | **0.8607** | **0.9109** |

## Citation

If you use this code, please cite:

```bibtex
@article{medsegdiff2023,
  title={MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model},
  author={Wu, Junde and Fu, Rao and Fang, Huihui and Liu, Yuanpei and Wang, Zhaowei and Xu, Yanwu and Jin, Yueming and Arbel, Tal},
  journal={arXiv preprint arXiv:2211.00611},
  year={2023}
}
```

## License

MIT License
