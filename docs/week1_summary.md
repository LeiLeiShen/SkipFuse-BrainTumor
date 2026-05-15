# ProtoSAM-Med3D: Week 1 Summary

> Date: 2026-02-14
> Days completed: 1–7
> Status: ✅ All milestones achieved

## Accomplishments

### Day 1–2: Environment & Architecture Analysis
- Cloud GPU platform configured (NVIDIA L20, 48GB, PyTorch 2.10.0+cu128)
- SAM-Med3D turbo loaded and verified (100.5M params)
- Complete architecture trace with forward hook shape analysis
- Key finding: embedding dim = **384** (not 256 as initially assumed)

### Day 3: PGP-SAM Source Code Analysis
- Discovered **dual-prototype system** (inter + intra prototypes)
- Identified real module names: PrototypeRefinement, PromptGenerator (Dense+Sparse), ClassAttention
- Core file: `prototype_prompt_encoder.py` (735 lines, 15+ classes)
- Found built-in LoRA implementation in PGP-SAM's image_encoder.py
- Generated `docs/pgp_sam_analysis.md`

### Day 4: Data Preparation
- BraTS2024 blocked by Synapse regional restrictions (NIH policy)
- Pivoted to **BraTS2021** (already on platform): 1251 samples, all complete
- Label mapping: {0,1,2,4} — 3 foreground classes (NCR, ED, ET)

### Day 5: Preprocessing Pipeline
- Pipeline: crop brain region (pad=5) → Z-score normalize → resize 128³ → remap labels
- Output: image (4,128,128,128) float32 + label (128,128,128) int8, labels {0,1,2,3}
- All 1251 samples processed in 35 min, saved as .npy

### Day 6: Data Loading Infrastructure
- `BraTS2021Dataset`: augmentation (3-axis flip + intensity shift/scale)
- `FewShotSampler`: K-shot support set sampling + masked average pooling prototypes
- `get_dataloaders()`: one-line train/val/test creation
- End-to-end verified: npy → DataLoader → GPU → SAM-Med3D encoder

## Key Statistics

| Metric | Value |
|--------|-------|
| Total samples | 1251 |
| Train / Val / Test | 1000 / 125 / 126 |
| Image shape | (4, 128, 128, 128) |
| Label classes | 0=BG, 1=NCR, 2=ED, 3=ET |
| BG vs Tumor ratio | 98.93% : 1.07% |
| Mean WT volume | 95.97 cm³ (median 89.33) |
| Samples missing ET | 33/1251 (2.6%) |
| Samples missing NCR | 43/1251 (3.4%) |
| Complete (3-class) samples in train | 944/1000 |
| Processed data size | ~41.5 GB |

## File Inventory

| File | Purpose |
|------|---------|
| `data/BraTS2021/` → symlink | Raw BraTS2021 data (1251 cases) |
| `data/processed_BraTS2021/` | Preprocessed .npy files |
| `data/splits/brats2021_split.json` | 80/10/10 split (seed=42) |
| `data_processing/dataset.py` | Dataset + FewShotSampler classes |
| `docs/architecture_trace.md` | SAM-Med3D architecture reference |
| `docs/pgp_sam_analysis.md` | PGP-SAM module analysis + 3D plan |
| `notebooks/Day1–7_*.ipynb` | Daily experiment notebooks |
| `figs/data_exploration/` | Visualizations |

## Critical Numbers for Week 2

These dimensions must be used consistently across all new modules:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `feat_dim` | **384** | SAM-Med3D neck output |
| `spatial_size` | **(8, 8, 8)** | Feature map spatial |
| `num_tokens` | **512** | 8³ flattened |
| `input_channels` | **4** | T1, T1ce, T2, FLAIR |
| `num_classes` | **4** | BG + 3 foreground (BraTS2021) |
| `num_fg_classes` | **3** | NCR, ED, ET |
| `patch_size` | **16** | ViT patch embedding |
| `vit_dim` | **768** | ViT internal dim |

## Week 2 Plan: Model Construction

| Day | Task | Key Output |
|-----|------|------------|
| Day 8 | Modify PatchEmbed: Conv3d(1→4 channels) | `models/patch_embed.py` |
| Day 9 | Implement LoRA3D for ViT qkv | `models/lora3d.py` |
| Day 10 | Build GlobalPrototypes3D | `models/prototypes.py` |
| Day 11 | Build PrototypeRefinement3D | `models/prototype_refinement.py` |
| Day 12 | Build PromptGenerator3D (Dense+Sparse) | `models/prompt_generator.py` |
| Day 13 | Modify MaskDecoder3D for multi-class | `models/mask_decoder.py` |
| Day 14 | Integration: ProtoSAM_Med3D full forward | `models/proto_sam_med3d.py` |

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| BraTS2024 data inaccessible | Use BraTS2021; add 2024 later via label mapping |
| Memory overflow (4ch × 128³) | batch_size=2, gradient accumulation, mixed precision |
| 3D deformable conv too expensive | Replace with standard Conv3d |
| No label 3 in BraTS2021 | Remap 4→3; pipeline already handles this |
