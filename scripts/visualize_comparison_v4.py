#!/usr/bin/env python3
"""
Visualization v4: Compare HSM3D (ProtoSAM_Hybrid) vs nnU-Net in ORIGINAL space.

Changes from v3:
  - Model: ProtoSAM_Hybrid (not ProtoSAM_Ablation)
  - Checkpoint: checkpoints/hybrid_c32_ds_withckpt/best.pth
  - Val split: fold 0 from data/splits/brats2021_5fold.json (not 80/20)

Usage:
  cd .
  python scripts/visualize_comparison_v4.py
"""

import sys, os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
from glob import glob
from scipy.ndimage import zoom
import json

PROJECT = '.'
sys.path.insert(0, os.path.join(PROJECT, 'SAM-Med3D'))
sys.path.insert(0, PROJECT)

DATA_DIR = os.path.join(PROJECT, 'data/processed_BraTS2021')
HYBRID_CKPT = os.path.join(PROJECT, 'checkpoints/hybrid_c32_ds_withckpt/best.pth')
SAM_CKPT = os.path.join(PROJECT, 'SAM-Med3D/ckpt/sam_med3d_turbo.pth')
RAW_DIR = './BraTS2021_Training_Data'
NNUNET_DIR = (
    './nnUNet_data/nnUNet_results/'
    'Dataset001_BraTS2021/nnUNetTrainer__nnUNetPlans__3d_fullres/'
    'fold_0/validation'
)
FOLD_SPLIT = os.path.join(PROJECT, 'data/splits/brats2021_5fold.json')
OUTPUT_DIR = os.path.join(PROJECT, 'visualizations')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TUMOR_COLORS = {
    1: np.array([255, 50, 50]),    # NCR - red
    2: np.array([50, 255, 50]),    # ED - green
    3: np.array([255, 255, 50]),   # ET - yellow
}


# ============================================================================
# Hybrid model (ProtoSAM_Hybrid)
# ============================================================================

def load_hybrid_model(device='cuda'):
    from models.proto_sam_hybrid import ProtoSAM_Hybrid
    model = ProtoSAM_Hybrid(
        sam_checkpoint=SAM_CKPT,
        num_classes=4,
        lora_r=16,
        unfreeze_blocks=0,
        cnn_base_ch=32,
    ).to(device)
    ckpt = torch.load(HYBRID_CKPT, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    md = model.state_dict()
    loaded = {k: v for k, v in state.items() if k in md and v.shape == md[k].shape}
    md.update(loaded)
    model.load_state_dict(md, strict=False)
    print(f"  HSM3D loaded: {len(loaded)}/{len(md)} keys")
    if 'best_dice' in ckpt:
        print(f"  Checkpoint best_dice: {ckpt['best_dice']:.4f}")
    model.eval()
    return model


@torch.no_grad()
def hybrid_predict_128(model, image_np, device='cuda'):
    img = torch.tensor(image_np, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.amp.autocast('cuda'):
        out = model(img)
        # Handle deep supervision: model may return list
        if isinstance(out, (list, tuple)):
            logits = out[0]
        else:
            logits = out
    return logits.argmax(1).squeeze().cpu().numpy().astype(np.uint8)


# ============================================================================
# Reverse mapping: Hybrid 128³ → original space
# ============================================================================

def get_bbox_from_raw(case_name):
    """Compute non-zero bbox from 4-modality union (same as preprocessing)."""
    mods = ['t1', 't1ce', 't2', 'flair']
    raw = os.path.join(RAW_DIR, case_name)
    volumes = []
    for m in mods:
        d = nib.load(os.path.join(raw, f'{case_name}_{m}.nii.gz')).get_fdata()
        volumes.append(d)
    image = np.stack(volumes, axis=0)
    orig_shape = image.shape[1:]  # (240, 240, 155)
    brain_mask = np.any(image > 0, axis=0)
    coords = np.where(brain_mask)

    # Match preprocessing: pad = 5
    pad = 5
    mins = np.array([
        max(0, coords[0].min() - pad),
        max(0, coords[1].min() - pad),
        max(0, coords[2].min() - pad),
    ])
    maxs = np.array([
        min(orig_shape[0], coords[0].max() + pad + 1),
        min(orig_shape[1], coords[1].max() + pad + 1),
        min(orig_shape[2], coords[2].max() + pad + 1),
    ])
    return mins, maxs, orig_shape


def hybrid_to_original(pred_128, case_name):
    """Map Hybrid prediction from 128³ back to original space."""
    mins, maxs, orig_shape = get_bbox_from_raw(case_name)
    crop_shape = maxs - mins

    # Upsample 128³ → crop_shape (nearest neighbor)
    scale = tuple(c / 128.0 for c in crop_shape)
    pred_crop = zoom(pred_128, scale, order=0).astype(np.uint8)

    # Place into full volume
    pred_full = np.zeros(orig_shape, dtype=np.uint8)
    pred_full[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]] = pred_crop
    return pred_full


# ============================================================================
# Load GT and nnU-Net in original space
# ============================================================================

def load_gt_original(case_name):
    seg = nib.load(os.path.join(RAW_DIR, case_name, f'{case_name}_seg.nii.gz')).get_fdata()
    seg[seg == 4] = 3  # BraTS remap
    return seg.astype(np.uint8)


def load_nnunet_pred(case_name):
    path = os.path.join(NNUNET_DIR, f'{case_name}.nii.gz')
    if not os.path.exists(path):
        return None
    return nib.load(path).get_fdata().astype(np.uint8)


def load_mri_original(case_name, modality='t1ce'):
    return nib.load(os.path.join(RAW_DIR, case_name, f'{case_name}_{modality}.nii.gz')).get_fdata()


# ============================================================================
# Metrics & visualization
# ============================================================================

def dice(p, g):
    if p.sum() == 0 and g.sum() == 0: return 1.0
    if p.sum() == 0 or g.sum() == 0: return 0.0
    return float(2 * (p & g).sum() / (p.sum() + g.sum()))

def brats_dice(pred, gt):
    return {
        'ET': dice(pred == 3, gt == 3),
        'TC': dice((pred == 1) | (pred == 3), (gt == 1) | (gt == 3)),
        'WT': dice(pred >= 1, gt >= 1),
    }

def mean_d(d):
    return np.mean([d['ET'], d['TC'], d['WT']])


def find_best_slice(label_vol):
    tumor = [(label_vol[:, :, z] > 0).sum() for z in range(label_vol.shape[2])]
    return int(np.argmax(tumor))


def make_overlay(mri_slice, seg_slice, alpha=0.6):
    mri = mri_slice.copy().astype(float)
    fg = mri > 0
    if fg.any():
        p1, p99 = np.percentile(mri[fg], [1, 99])
    else:
        p1, p99 = 0, 1
    mri = np.clip((mri - p1) / (p99 - p1 + 1e-8), 0, 1)
    rgb = np.stack([mri * 255] * 3, axis=-1).astype(np.float32)

    for cls_id in [2, 1, 3]:
        mask = (seg_slice == cls_id)
        if mask.any():
            rgb[mask] = rgb[mask] * (1 - alpha) + TUMOR_COLORS[cls_id].astype(np.float32) * alpha

    return np.clip(rgb, 0, 255).astype(np.uint8)


def make_mri_only(mri_slice):
    mri = mri_slice.copy().astype(float)
    fg = mri > 0
    if fg.any():
        p1, p99 = np.percentile(mri[fg], [1, 99])
    else:
        p1, p99 = 0, 1
    mri = np.clip((mri - p1) / (p99 - p1 + 1e-8), 0, 1)
    return np.stack([mri * 255] * 3, axis=-1).astype(np.uint8)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Visualization v4: HSM3D vs nnU-Net (Original Space)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load fold 0 validation cases
    print("\n[1/5] Loading fold 0 validation cases...")
    with open(FOLD_SPLIT) as f:
        fold_data = json.load(f)
    val_cases_fold = set(fold_data['fold_0']['val'])
    print(f"  Fold 0 val cases: {len(val_cases_fold)}")

    # Check which have nnU-Net predictions
    nn_cases = set()
    for f_name in os.listdir(NNUNET_DIR):
        if f_name.endswith('.nii.gz'):
            nn_cases.add(f_name.replace('.nii.gz', ''))

    # Check which have processed 128³ data
    processed_cases = set()
    for p in glob(os.path.join(DATA_DIR, '*_image.npy')):
        processed_cases.add(os.path.basename(p).replace('_image.npy', ''))

    overlap = sorted(val_cases_fold & nn_cases & processed_cases)
    print(f"  Overlap (fold0_val ∩ nnunet ∩ processed): {len(overlap)} cases")

    if len(overlap) == 0:
        # Fallback: use fold0 val cases that have processed data
        overlap_no_nn = sorted(val_cases_fold & processed_cases)
        print(f"  Fallback (fold0_val ∩ processed, no nnU-Net filter): {len(overlap_no_nn)} cases")
        overlap = overlap_no_nn

    # 2. Load HSM3D model
    print("\n[2/5] Loading HSM3D model...")
    model = load_hybrid_model(device)

    # 3. Inference + metrics in original space
    print("\n[3/5] Running inference...")
    results = []

    for i, case in enumerate(overlap):
        # HSM3D: predict in 128³, then map back
        img_path = os.path.join(DATA_DIR, f'{case}_image.npy')
        if not os.path.exists(img_path):
            continue
        img_128 = np.load(img_path)
        pred_128 = hybrid_predict_128(model, img_128, device)
        hybrid_orig = hybrid_to_original(pred_128, case)

        # GT & nnU-Net in original space
        gt_orig = load_gt_original(case)
        nn_orig = load_nnunet_pred(case)

        if gt_orig.max() == 0:
            continue

        hd = brats_dice(hybrid_orig, gt_orig)
        r = {'case': case, 'hybrid_dice': hd, 'hybrid_mean': mean_d(hd),
             'hybrid_orig': hybrid_orig}

        if nn_orig is not None:
            nd = brats_dice(nn_orig, gt_orig)
            r['nnunet_dice'] = nd
            r['nnunet_mean'] = mean_d(nd)
            r['nnunet_orig'] = nn_orig

        r['gt_orig'] = gt_orig
        r['case_name'] = case
        results.append(r)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(overlap)}]")

    print(f"  Valid: {len(results)}")

    # 4. Alignment check
    hy_all = [r['hybrid_mean'] for r in results]
    nn_all = [r['nnunet_mean'] for r in results if 'nnunet_mean' in r]
    print(f"\n  HSM3D mean:  {np.mean(hy_all):.4f}")
    if nn_all:
        print(f"  nnU-Net mean:   {np.mean(nn_all):.4f}")

    # Per-region averages
    for region in ['ET', 'TC', 'WT']:
        hy_r = np.mean([r['hybrid_dice'][region] for r in results])
        nn_r = np.mean([r[region] for r in [r2['nnunet_dice'] for r2 in results if 'nnunet_dice' in r2]])
        print(f"  {region}: HSM3D={hy_r:.4f}  nnU-Net={nn_r:.4f}")

    # 5. Select and visualize
    print("\n[4/5] Selecting cases...")
    results.sort(key=lambda x: x['hybrid_mean'])
    n = len(results)

    if n >= 5:
        indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        labels = ['Hard', 'Below Med.', 'Median', 'Above Med.', 'Easy']
    elif n >= 3:
        indices = [0, n // 2, n - 1]
        labels = ['Hard', 'Median', 'Easy']
    else:
        indices = list(range(n))
        labels = [f'Case {i}' for i in range(n)]

    selected = [(results[idx], labels[i]) for i, idx in enumerate(indices)]

    print("\n[5/5] Generating figures...")

    # Combined figure
    n_rows = len(selected)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.2 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row, (r, diff) in enumerate(selected):
        case = r['case_name']
        gt = r['gt_orig']
        z = find_best_slice(gt)
        mri = load_mri_original(case, 't1ce')[:, :, z]

        # Col 0: MRI
        axes[row, 0].imshow(make_mri_only(mri), aspect='auto')
        axes[row, 0].axis('off')
        axes[row, 0].set_ylabel(f"{diff}\n{case}\nz={z}",
                                 fontsize=9, fontweight='bold',
                                 rotation=0, labelpad=90, va='center')

        # Col 1: GT
        axes[row, 1].imshow(make_overlay(mri, gt[:, :, z]), aspect='auto')
        axes[row, 1].axis('off')

        # Col 2: HSM3D
        hd = r['hybrid_dice']
        axes[row, 2].imshow(make_overlay(mri, r['hybrid_orig'][:, :, z]), aspect='auto')
        axes[row, 2].axis('off')
        axes[row, 2].set_xlabel(
            f"ET={hd['ET']:.0%} TC={hd['TC']:.0%} WT={hd['WT']:.0%}\nMean={r['hybrid_mean']:.1%}",
            fontsize=8)

        # Col 3: nnU-Net
        if 'nnunet_orig' in r:
            nd = r['nnunet_dice']
            axes[row, 3].imshow(make_overlay(mri, r['nnunet_orig'][:, :, z]), aspect='auto')
            axes[row, 3].set_xlabel(
                f"ET={nd['ET']:.0%} TC={nd['TC']:.0%} WT={nd['WT']:.0%}\nMean={r['nnunet_mean']:.1%}",
                fontsize=8)
        else:
            axes[row, 3].text(0.5, 0.5, 'N/A', ha='center', va='center',
                               fontsize=14, transform=axes[row, 3].transAxes)
        axes[row, 3].axis('off')

        if row == 0:
            for c, t in enumerate(['T1ce MRI', 'Ground Truth', 'HSM3D (Ours)', 'nnU-Net']):
                axes[0, c].set_title(t, fontsize=13, fontweight='bold')

    legend_patches = [
        mpatches.Patch(color=np.array([255, 50, 50]) / 255, label='NCR'),
        mpatches.Patch(color=np.array([50, 255, 50]) / 255, label='ED'),
        mpatches.Patch(color=np.array([255, 255, 50]) / 255, label='ET'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'comparison_v4.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"  Saved: {path}")

    # Individual figures
    for r, diff in selected:
        case = r['case_name']
        gt = r['gt_orig']
        z = find_best_slice(gt)
        mri = load_mri_original(case, 't1ce')[:, :, z]
        has_nn = 'nnunet_orig' in r
        nc = 4 if has_nn else 3

        fig2, ax = plt.subplots(1, nc, figsize=(4.8 * nc, 4.8))
        ax[0].imshow(make_mri_only(mri), aspect='auto')
        ax[0].set_title('T1ce', fontsize=12, fontweight='bold')
        ax[0].axis('off')

        ax[1].imshow(make_overlay(mri, gt[:, :, z]), aspect='auto')
        ax[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
        ax[1].axis('off')

        hd = r['hybrid_dice']
        ax[2].imshow(make_overlay(mri, r['hybrid_orig'][:, :, z]), aspect='auto')
        ax[2].set_title(f"HSM3D (Mean={r['hybrid_mean']:.1%})", fontsize=12, fontweight='bold')
        ax[2].axis('off')

        if has_nn:
            nd = r['nnunet_dice']
            ax[3].imshow(make_overlay(mri, r['nnunet_orig'][:, :, z]), aspect='auto')
            ax[3].set_title(f"nnU-Net (Mean={r['nnunet_mean']:.1%})", fontsize=12, fontweight='bold')
            ax[3].axis('off')

        fig2.legend(handles=legend_patches, loc='lower center', ncol=3,
                    fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))
        plt.tight_layout()
        fp = os.path.join(OUTPUT_DIR, f"vis_v4_{diff.lower().replace(' ','_').replace('.','')  }_{case}.png")
        fig2.savefig(fp, dpi=200, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"  {case} [{diff}]: HSM3D={r['hybrid_mean']:.3f}"
              + (f"  nnU-Net={r['nnunet_mean']:.3f}" if has_nn else ""))

    # Summary JSON
    summary = []
    for r, diff in selected:
        s = {'case': r['case_name'], 'difficulty': diff,
             'hybrid': r['hybrid_dice'], 'hybrid_mean': r['hybrid_mean']}
        if 'nnunet_dice' in r:
            s['nnunet'] = r['nnunet_dice']
            s['nnunet_mean'] = r['nnunet_mean']
        summary.append(s)
    with open(os.path.join(OUTPUT_DIR, 'summary_v4.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nDone!")


if __name__ == '__main__':
    main()
