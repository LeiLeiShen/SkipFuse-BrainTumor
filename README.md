# SkipFuse: Parameter-Efficient Adaptation of SAM-Med3D for Brain Tumor Segmentation

This repository contains the official implementation of **SkipFuse**, a parameter-efficient fine-tuning framework that adapts [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D) for 3D brain tumor segmentation on BraTS2021.

## Overview

SkipFuse introduces a hybrid architecture combining the frozen SAM-Med3D ViT-B encoder with a lightweight parallel CNN branch and multi-scale skip connections. Only ~24M parameters are trainable (via LoRA r=16 on the ViT encoder + CNN branch + decoder), while the full SAM-Med3D encoder remains frozen.

**Key results on BraTS2021 (region-based Dice %):**

| Method | ET | TC | WT | Mean |
|--------|------|------|------|------|
| SAM-Med3D (zero-shot) | 0.3 | 1.0 | 0.9 | 0.7 |
| SAMed | 56.4 | 71.2 | 74.8 | 67.5 |
| GBT-SAM | — | — | 83.1 | 83.1* |
| 3D SAM-adapter | 79.2 | 88.5 | 89.3 | 85.7 |
| **SkipFuse (Ours)** | **83.2** | **88.8** | **89.5** | **87.4** |

*GBT-SAM reports WT-only.

## Architecture

```
Input (128×128×128, 4ch)
  ├── SAM-Med3D ViT-B Encoder (frozen + LoRA r=16)
  │     └── Multi-scale features via skip connections
  ├── Parallel CNN Branch (lightweight 3D ConvNet)
  │     └── Multi-scale features at matching resolutions
  └── Hybrid Decoder
        ├── Skip fusion (ViT + CNN features at each scale)
        └── Progressive upsampling → 3-class segmentation
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/SkipFuse-BrainTumor.git
cd SkipFuse-BrainTumor
pip install -r requirements.txt
```

### Pre-trained Weights

Download the SAM-Med3D ViT-B checkpoint from [SAM-Med3D releases](https://github.com/uni-medical/SAM-Med3D) and place it in `checkpoints/`:

```bash
mkdir -p checkpoints
# Download sam_med3d_turbo.pth or sam_med3d.pth
```

## Data Preparation

1. Download [BraTS2021 Training Data](https://www.synapse.org/#!Synapse:syn27046444/wiki/616571).
2. Run preprocessing:

```bash
python scripts/preprocess_brats.py \
    --data_dir /path/to/BraTS2021_Training_Data \
    --output_dir data/processed
```

3. Data splits are provided in `data/splits/`.

## Training

```bash
python scripts/train_hybrid.py \
    --data_dir /path/to/BraTS2021_Training_Data \
    --checkpoint checkpoints/sam_med3d_turbo.pth \
    --lora_r 16 \
    --cnn_channels 32 \
    --epochs 200 \
    --batch_size 2 \
    --lr 1e-4
```

## Evaluation

```bash
python scripts/eval_brats_regions.py \
    --data_dir /path/to/BraTS2021_Training_Data \
    --model_path checkpoints/best_model.pth \
    --split_file data/splits/brats2021_split.json
```

## Baseline Reproduction

We reproduced several SAM-based baselines for fair comparison. Pre-computed results are in `baseline_results/`. For reproduction details, see the original repositories:

- [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D) — zero-shot evaluation
- [SAMed](https://github.com/hitachinsk/SAMed) — 2D LoRA-based adaptation
- [3D SAM-adapter](https://github.com/med-air/3DSAM-adapter) — 3D adapter tuning
- [GBT-SAM](https://github.com/Lizhecheng02/GBT-SAM) — gradient-based tuning

## Project Structure

```
├── models/                  # Model definitions
│   ├── proto_sam_hybrid.py  # Main SkipFuse model
│   ├── cnn_branch_3d.py     # Parallel CNN branch
│   ├── hybrid_decoder_3d.py # Hybrid decoder with skip fusion
│   ├── lora3d.py            # 3D LoRA injection
│   └── patch_embed.py       # 4-channel patch embedding
├── segment_anything/        # SAM-Med3D backbone (3D)
├── data_processing/         # Dataset and data loading
├── utils/                   # Loss functions and metrics
├── scripts/                 # Training, evaluation, preprocessing
├── data/splits/             # Train/val/test split files
├── baseline_results/        # Reproduced baseline results (JSON)
└── docs/                    # Architecture notes
```

## Citation

```bibtex
@mastersthesis{shen2026skipfuse,
  title={Brain Tumor Segmentation with Foundation Model Adaptation: From 2D Architectural Innovation to 3D Parameter-Efficient Fine-Tuning},
  author={Shen, Lei},
  year={2026},
  school={University of Nottingham Ningbo China}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgments

- [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D) for the 3D medical SAM foundation model
- [Segment Anything](https://github.com/facebookresearch/segment-anything) (Meta AI) for the original SAM architecture
- BraTS2021 challenge organizers for the dataset
