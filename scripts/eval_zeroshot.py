#!/usr/bin/env python3
"""
Evaluate SAM-Med3D zero-shot on BraTS2021 (no fine-tuning, no LoRA).

Uses the ORIGINAL SAM-Med3D mask decoder with oracle point prompts
(ground-truth center-of-mass per tumor class).

This establishes the "SAM-Med3D zero-shot" baseline in the paper:
  - Pretrained encoder: frozen, no LoRA, no adaptation
  - Original mask decoder: frozen, prompt-based
  - Input: single-channel (average of 4 modalities, since original model is 1-ch)
  - Prompts: GT center-of-mass per foreground class (oracle, best-case scenario)

Two evaluation modes:
  (A) Oracle-prompted: 1 GT center point per foreground class → 3 binary masks → combine
  (B) Multi-point prompted: 5 GT foreground points per class → 3 binary masks → combine

Usage:
  cd .
  python scripts/eval_zeroshot.py
"""

import sys, os
import numpy as np
import torch
import torch.nn.functional as F
from glob import glob
from scipy import ndimage
import json, time

# Paths
PROJECT = '.'
sys.path.insert(0, os.path.join(PROJECT, 'SAM-Med3D'))
sys.path.insert(0, PROJECT)

SAM_CKPT = os.path.join(PROJECT, 'SAM-Med3D/ckpt/sam_med3d_turbo.pth')
DATA_DIR = os.path.join(PROJECT, 'data/processed_BraTS2021')  # preprocessed BraTS2021 GLI


# ============================================================================
# Load SAM-Med3D (original, unmodified)
# ============================================================================

def load_sam_med3d(checkpoint_path, device='cuda'):
    """Load the original SAM-Med3D model without any modifications."""
    from segment_anything.build_sam3D import sam_model_registry3D

    sam = sam_model_registry3D['vit_b_ori'](checkpoint=None)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt)
        # Filter out incompatible keys
        model_dict = sam.state_dict()
        compatible = {k: v for k, v in state.items()
                      if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(compatible)
        sam.load_state_dict(model_dict, strict=False)
        print(f"Loaded SAM-Med3D: {len(compatible)}/{len(state)} keys matched")
    else:
        print(f"WARNING: checkpoint not found at {checkpoint_path}")

    sam = sam.to(device).eval()
    for p in sam.parameters():
        p.requires_grad = False

    return sam


# ============================================================================
# Prompt generation from GT labels
# ============================================================================

def get_center_of_mass(label_vol, class_id):
    """Get center-of-mass coordinates for a specific class."""
    mask = (label_vol == class_id)
    if mask.sum() == 0:
        return None
    coords = ndimage.center_of_mass(mask.astype(float))
    return np.array(coords, dtype=np.float32)


def get_random_foreground_points(label_vol, class_id, n_points=5, seed=42):
    """Sample n random foreground points for a specific class."""
    rng = np.random.RandomState(seed)
    mask = (label_vol == class_id)
    if mask.sum() == 0:
        return None
    coords = np.argwhere(mask)  # (N, 3)
    if len(coords) == 0:
        return None
    idx = rng.choice(len(coords), min(n_points, len(coords)), replace=False)
    return coords[idx].astype(np.float32)  # (n_points, 3)


# ============================================================================
# SAM-Med3D inference with point prompts
# ============================================================================

@torch.no_grad()
def sam_predict_with_points(sam, image_1ch, points_3d, point_labels, device='cuda'):
    """
    Run SAM-Med3D forward pass with point prompts.

    Args:
        sam: SAM-Med3D model
        image_1ch: (1, 1, 128, 128, 128) single-channel image
        points_3d: (N, 3) point coordinates in voxel space
        point_labels: (N,) labels (1=foreground, 0=background)

    Returns:
        binary_mask: (128, 128, 128) numpy array
    """
    # Encode image
    img_embed = sam.image_encoder(image_1ch)  # (1, 384, 8, 8, 8)

    # Prepare prompts - SAM-Med3D expects (B, N, 3) points and (B, N) labels
    pts = torch.tensor(points_3d, dtype=torch.float32, device=device).unsqueeze(0)  # (1, N, 3)
    lbl = torch.tensor(point_labels, dtype=torch.long, device=device).unsqueeze(0)  # (1, N)

    # Prompt encoder
    sparse_embed, dense_embed = sam.prompt_encoder(
        points=(pts, lbl),
        boxes=None,
        masks=None,
    )

    # Mask decoder
    low_res_masks, iou_pred = sam.mask_decoder(
        image_embeddings=img_embed,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embed,
        dense_prompt_embeddings=dense_embed,
        multimask_output=False,
    )

    # Upsample to 128³
    masks = F.interpolate(
        low_res_masks, size=(128, 128, 128),
        mode='trilinear', align_corners=False
    )

    binary = (masks.squeeze() > 0).cpu().numpy()  # threshold at 0
    return binary


# ============================================================================
# Multi-class prediction by combining per-class binary masks
# ============================================================================

def predict_multiclass(sam, image_4ch, label, mode='center', n_points=5, device='cuda'):
    """
    Predict multi-class BraTS segmentation using per-class oracle prompts.

    Strategy:
      1. For each foreground class (1=NCR, 2=ED, 3=ET), generate oracle point prompts
      2. Run SAM-Med3D to get a binary mask per class
      3. Combine masks with priority: ET(3) > NCR(1) > ED(2) > BG(0)

    Args:
        image_4ch: (4, 128, 128, 128) numpy array
        label: (128, 128, 128) numpy array
        mode: 'center' for center-of-mass, 'multi' for multiple random points
    """
    # SAM-Med3D expects single-channel input → average modalities
    image_avg = image_4ch.mean(axis=0, keepdims=True)  # (1, 128, 128, 128)
    image_tensor = torch.tensor(image_avg, dtype=torch.float32, device=device).unsqueeze(0)
    # → (1, 1, 128, 128, 128)

    pred = np.zeros((128, 128, 128), dtype=np.uint8)
    masks_per_class = {}

    for cls_id in [1, 2, 3]:
        if mode == 'center':
            center = get_center_of_mass(label, cls_id)
            if center is None:
                continue
            points = center.reshape(1, 3)  # (1, 3)
            labels = np.array([1], dtype=np.int64)  # foreground
        elif mode == 'multi':
            points = get_random_foreground_points(label, cls_id, n_points=n_points)
            if points is None:
                continue
            labels = np.ones(len(points), dtype=np.int64)  # all foreground

        binary_mask = sam_predict_with_points(sam, image_tensor, points, labels, device)
        masks_per_class[cls_id] = binary_mask

    # Combine: lower priority first, higher priority overwrites
    # Priority: ED(2) → NCR(1) → ET(3)
    for cls_id in [2, 1, 3]:
        if cls_id in masks_per_class:
            pred[masks_per_class[cls_id]] = cls_id

    return pred


# ============================================================================
# BraTS region-based Dice evaluation
# ============================================================================

def dice_coefficient(pred_mask, gt_mask):
    if pred_mask.sum() == 0 and gt_mask.sum() == 0:
        return 1.0
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return 0.0
    intersection = (pred_mask & gt_mask).sum()
    return float(2 * intersection / (pred_mask.sum() + gt_mask.sum()))


def brats_region_dice(pred, gt):
    """Compute BraTS region-based Dice (ET, TC, WT)."""
    et_p, et_g = (pred == 3), (gt == 3)
    tc_p, tc_g = (pred == 1) | (pred == 3), (gt == 1) | (gt == 3)
    wt_p, wt_g = (pred >= 1), (gt >= 1)

    return {
        'ET': dice_coefficient(et_p, et_g),
        'TC': dice_coefficient(tc_p, tc_g),
        'WT': dice_coefficient(wt_p, wt_g),
    }


# ============================================================================
# Main evaluation
# ============================================================================

def main():
    print("=" * 70)
    print("SAM-Med3D Zero-Shot Evaluation on BraTS2021")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print("\n[1/3] Loading SAM-Med3D (original, no modifications)...")
    sam = load_sam_med3d(SAM_CKPT, device)
    n_params = sum(p.numel() for p in sam.parameters())
    print(f"  Total parameters: {n_params/1e6:.1f}M")

    # Get validation cases (same split as training experiments: last 20%)
    print("\n[2/3] Loading validation data...")
    all_images = sorted(glob(os.path.join(DATA_DIR, '*_image.npy')))
    all_labels = sorted(glob(os.path.join(DATA_DIR, '*_label.npy')))
    assert len(all_images) == len(all_labels), \
        f"Mismatch: {len(all_images)} images vs {len(all_labels)} labels"

    n_total = len(all_images)
    n_train = int(n_total * 0.8)
    val_images = all_images[n_train:]
    val_labels = all_labels[n_train:]
    print(f"  Total: {n_total}, Train: {n_train}, Val: {len(val_images)}")

    # Evaluate both modes
    for mode, mode_name in [('center', 'Oracle Center-of-Mass (1 point/class)'),
                             ('multi', 'Oracle Multi-Point (5 points/class)')]:
        print(f"\n[3/3] Evaluating: {mode_name}")
        print("-" * 60)

        results = {'ET': [], 'TC': [], 'WT': []}
        t0 = time.time()
        skipped = 0

        for i, (img_path, lbl_path) in enumerate(zip(val_images, val_labels)):
            case_name = os.path.basename(img_path).replace('_image.npy', '')

            image = np.load(img_path)  # (4, 128, 128, 128)
            label = np.load(lbl_path)  # (128, 128, 128)

            # Skip cases with no tumor
            if label.max() == 0:
                skipped += 1
                continue

            try:
                pred = predict_multiclass(sam, image, label, mode=mode,
                                           n_points=5, device=device)
                d = brats_region_dice(pred, label)

                results['ET'].append(d['ET'])
                results['TC'].append(d['TC'])
                results['WT'].append(d['WT'])

                if (i + 1) % 50 == 0:
                    n = len(results['ET'])
                    print(f"  [{i+1}/{len(val_images)}] "
                          f"ET={np.mean(results['ET']):.4f} "
                          f"TC={np.mean(results['TC']):.4f} "
                          f"WT={np.mean(results['WT']):.4f} "
                          f"Mean={np.mean([np.mean(results[r]) for r in ['ET','TC','WT']]):.4f}")

            except Exception as e:
                print(f"  ERROR on {case_name}: {e}")
                skipped += 1
                continue

        elapsed = time.time() - t0
        n = len(results['ET'])

        print(f"\n{'='*60}")
        print(f"Results: {mode_name}")
        print(f"{'='*60}")
        print(f"  Cases evaluated: {n} (skipped: {skipped})")
        print(f"  Time: {elapsed:.0f}s ({elapsed/max(n,1):.1f}s/case)")
        print(f"  ET:   {np.mean(results['ET']):.4f} ± {np.std(results['ET']):.4f}")
        print(f"  TC:   {np.mean(results['TC']):.4f} ± {np.std(results['TC']):.4f}")
        print(f"  WT:   {np.mean(results['WT']):.4f} ± {np.std(results['WT']):.4f}")
        mean_dice = np.mean([np.mean(results[r]) for r in ['ET', 'TC', 'WT']])
        print(f"  Mean: {mean_dice:.4f}")

        # Save results
        out_path = os.path.join(PROJECT, 'logs_enhanced',
                                f'zeroshot_{mode}_results.json')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump({
                'mode': mode,
                'description': mode_name,
                'n_cases': n,
                'n_skipped': skipped,
                'ET_mean': float(np.mean(results['ET'])),
                'TC_mean': float(np.mean(results['TC'])),
                'WT_mean': float(np.mean(results['WT'])),
                'mean_dice': float(mean_dice),
                'ET_std': float(np.std(results['ET'])),
                'TC_std': float(np.std(results['TC'])),
                'WT_std': float(np.std(results['WT'])),
                'per_case': {
                    'ET': [float(x) for x in results['ET']],
                    'TC': [float(x) for x in results['TC']],
                    'WT': [float(x) for x in results['WT']],
                },
            }, f, indent=2)
        print(f"  Saved: {out_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
