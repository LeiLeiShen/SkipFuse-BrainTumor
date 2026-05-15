# Brain Tumor Segmentation with Foundation Model Adaptation

This repository contains the code for our research on adapting the Segment Anything Model (SAM) for brain tumor segmentation, progressing from 2D architectural innovation to 3D parameter-efficient fine-tuning.

## Research Overview

This work addresses a central question: **how to effectively adapt visual foundation models for brain tumor segmentation under limited computational resources?**

We approach this through two stages:

**Stage 1 — 2D Dual-Encoder Architecture (Paper 1)**
We propose a dual-encoder architecture combining SAM's pre-trained ViT-B encoder with a ConvNeXt-based CNN encoder, enhanced by Swin-Transformer sub-branches and an adaptive feature fusion mechanism. On the BraTS2020 dataset, this architecture achieves 82.17% mean Dice, outperforming nnU-Net (79.38%) and several transformer-based models. This stage validates that a parallel CNN branch can effectively complement SAM's global features with local spatial details.

**Stage 2 — 3D Parameter-Efficient Fine-Tuning (Paper 2)**
Extending the dual-encoder insight to 3D, we adapt SAM-Med3D for volumetric brain tumor segmentation. We first conduct a systematic PEFT study revealing that encoder-side adaptation saturates rapidly (LoRA r=4 already reaches 99.7% of r=32 performance). We diagnose the bottleneck as the loss of multi-scale spatial information after 16× downsampling to 8³ resolution (25.7% of cases lose all ET voxels). We then propose **HSM3D** (Hybrid SAM-Med3D), which parallels a lightweight 3D CNN branch alongside the frozen ViT encoder and passes multi-scale features to the decoder through skip connections. With only 24M trainable parameters, HSM3D achieves 87.5% mean Dice on BraTS2021.

## Key Results

### Paper 1: 2D Dual-Encoder on BraTS2020

| Model | Dice (%) | Parameters |
|-------|----------|------------|
| U-Net | 50.93 | 7.76M |
| UNETR | 68.09 | 130.79M |
| Swin-UNETR | 69.44 | 15.57M |
| nnU-Net | 79.38 | 50.8M |
| **Conv-Swin-SAM (Ours)** | **82.17** | **52.5M** |

### Paper 2: 3D HSM3D on BraTS2021

| Method | ET | TC | WT | Mean | Trainable Params |
|--------|------|------|------|------|-----------------|
| SAM-Med3D (zero-shot) | 0.3 | 1.0 | 0.9 | 0.7 | — |
| SAMed | 49.2 | 67.8 | 85.5 | 67.5 | 4.35M |
| GBT-SAM | — | — | 83.1 | — | 16.6M |
| 3D SAM-adapter | 80.0 | 88.2 | 88.9 | 85.7 | 34.9M |
| **HSM3D+DS (Ours)** | **83.7** | **89.2** | **89.8** | **87.5** | **24.0M** |
| nnU-Net (fully supervised) | 86.6 | 91.7 | 94.0 | 90.8 | 31.0M* |

*nnU-Net trains all parameters from scratch with automated architecture search.

## Architecture

### Paper 1: Conv-Swin-SAM (2D)

```
Input (4-ch MRI, 1024×1024)
  ├── SAM ViT-B Encoder (progressive unfreezing)
  │     └── Global semantic features
  ├── ConvNeXt + Swin-Transformer CNN Branch
  │     └── Multi-scale local features with adaptive fusion
  └── Bottleneck (adaptive fusion of both branches)
        └── U-Net Decoder with deep supervision
```

### Paper 2: HSM3D (3D)

```
Input (4-ch MRI, 128³)
  ├── SAM-Med3D ViT-B Encoder (frozen + LoRA r=16)
  │     └── Global features at 8³ resolution
  ├── Lightweight 3D CNN Branch
  │     └── Multi-scale features at 64³, 32³, 16³, 8³
  └── Bottleneck Fusion + Hybrid Decoder
        ├── Skip connections (CNN → Decoder)
        └── Deep supervision → 3-class segmentation (128³)
```

## Repository Structure

```
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── paper1_2d/                   # Paper 1: 2D dual-encoder
│   └── conv_swin_sam.ipynb      # Training notebook (Google Colab)
│
├── models/                      # Paper 2: 3D model definitions
│   ├── proto_sam_hybrid.py      # Main HSM3D model
│   ├── cnn_branch_3d.py         # Parallel CNN branch
│   ├── hybrid_decoder_3d.py     # Hybrid decoder with skip fusion
│   ├── lora3d.py                # 3D LoRA injection
│   ├── patch_embed.py           # 4-channel patch embedding
│   └── decoders.py              # Decoder variants
│
├── segment_anything/            # SAM-Med3D 3D backbone
├── data_processing/             # Dataset and data loading
├── utils/                       # Loss functions and metrics
│
├── scripts/                     # Training, evaluation, preprocessing
│   ├── train_hybrid.py          # Main training script
│   ├── eval_brats_regions.py    # Region-based evaluation
│   ├── preprocess_brats.py      # Data preprocessing
│   └── setup_5fold.py           # Cross-validation setup
│
├── data/splits/                 # Train/val/test split files
├── baseline_results/            # Reproduced baseline results (JSON)
│   ├── gbt_sam/
│   ├── samed/
│   └── sam_med3d_vanilla/
├── results/                     # Our experimental results
└── docs/                        # Architecture notes
```

## Installation

```bash
git clone https://github.com/LeiLeiShen/SkipFuse-BrainTumor.git
cd SkipFuse-BrainTumor
pip install -r requirements.txt
```

### Pre-trained Weights

**Paper 1 (2D):** Download SAM ViT-B checkpoint from [SAM releases](https://github.com/facebookresearch/segment-anything#model-checkpoints).

**Paper 2 (3D):** Download SAM-Med3D checkpoint from [SAM-Med3D releases](https://github.com/uni-medical/SAM-Med3D) and place it in `checkpoints/`.

## Usage

### Paper 2: Training HSM3D

```bash
python scripts/train_hybrid.py \
    --data_dir /path/to/BraTS2021_Training_Data \
    --checkpoint checkpoints/sam_med3d_turbo.pth \
    --lora_r 16 \
    --cnn_channels 32 \
    --epochs 100 \
    --batch_size 2 \
    --lr 1e-4
```

### Paper 2: Evaluation

```bash
python scripts/eval_brats_regions.py \
    --data_dir /path/to/BraTS2021_Training_Data \
    --model_path checkpoints/best_model.pth \
    --split_file data/splits/brats2021_split.json
```

### Paper 1: 2D Training

See `paper1_2d/conv_swin_sam.ipynb` for the complete training pipeline on Google Colab.

## Baseline Reproduction

We reproduced several SAM-based baselines for fair comparison. Pre-computed results are in `baseline_results/`. For reproduction details, see the original repositories:

- [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D) — zero-shot evaluation
- [SAMed](https://github.com/hitachinsk/SAMed) — 2D LoRA-based adaptation
- [3D SAM-adapter](https://github.com/med-air/3DSAM-adapter) — 3D adapter tuning
- [GBT-SAM](https://github.com/Lizhecheng02/GBT-SAM) — gradient-based tuning

## Citation

```bibtex
@mastersthesis{shen2026braintumor,
  title={Brain Tumor Segmentation with Foundation Model Adaptation:
         From 2D Architectural Innovation to 3D Parameter-Efficient Fine-Tuning},
  author={Shen, Lei},
  year={2026},
  school={University of Nottingham Ningbo China}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgments

- [SAM](https://github.com/facebookresearch/segment-anything) and [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D) for the foundation models
- BraTS challenge organizers for the datasets
- University of Nottingham Ningbo China for computational resources
