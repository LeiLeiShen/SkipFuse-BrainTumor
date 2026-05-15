#!/usr/bin/env python3
"""
Convert nnU-Net per-label evaluation to BraTS region-based Dice (ET/TC/WT).

BraTS region definitions:
  ET (Enhancing Tumor)  = label 3
  TC (Tumor Core)       = label 1 ∪ label 3
  WT (Whole Tumor)      = label 1 ∪ label 2 ∪ label 3

Usage:
  python eval_brats_regions.py \
    --pred_dir /path/to/nnUNet/predictions \
    --gt_dir /path/to/nnUNet/ground_truth \
    --output results_brats_regions.json
    
Both directories should contain .nii.gz files with matching filenames.
Predictions are nnU-Net output (integer labels 0,1,2,3).
"""

import os
import sys
import json
import argparse
import numpy as np

# Try nibabel, fall back to npy if not available
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("Warning: nibabel not found, will try .npy files")


def dice_score(pred_mask, gt_mask):
    """Compute Dice between two binary masks."""
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    
    if pred_sum == 0 and gt_sum == 0:
        return 1.0  # Both empty = perfect match
    if pred_sum == 0 or gt_sum == 0:
        return 0.0  # One empty = no overlap
    
    intersection = (pred_mask & gt_mask).sum()
    return float(2.0 * intersection / (pred_sum + gt_sum))


def compute_brats_regions(pred, gt):
    """
    Compute BraTS region-based Dice scores.
    
    Args:
        pred: numpy array with integer labels (0,1,2,3)
        gt: numpy array with integer labels (0,1,2,3)
    
    Returns:
        dict with ET, TC, WT Dice scores
    """
    # ET = label 3
    et_pred = (pred == 3)
    et_gt = (gt == 3)
    
    # TC = label 1 ∪ label 3
    tc_pred = (pred == 1) | (pred == 3)
    tc_gt = (gt == 1) | (gt == 3)
    
    # WT = label 1 ∪ label 2 ∪ label 3
    wt_pred = (pred >= 1)
    wt_gt = (gt >= 1)
    
    return {
        'ET': dice_score(et_pred, et_gt),
        'TC': dice_score(tc_pred, tc_gt),
        'WT': dice_score(wt_pred, wt_gt),
    }


def load_volume(filepath):
    """Load a volume from .nii.gz or .npy file."""
    if filepath.endswith('.npy'):
        return np.load(filepath)
    elif filepath.endswith('.nii.gz') or filepath.endswith('.nii'):
        return nib.load(filepath).get_fdata().astype(np.int64)
    else:
        raise ValueError(f"Unsupported format: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='BraTS region-based Dice evaluation')
    parser.add_argument('--pred_dir', required=True, help='Directory with nnU-Net prediction files')
    parser.add_argument('--gt_dir', required=True, help='Directory with ground truth files')
    parser.add_argument('--output', default='brats_region_dice.json', help='Output JSON file')
    parser.add_argument('--ext', default='.nii.gz', help='File extension (.nii.gz or .npy)')
    args = parser.parse_args()
    
    # Find matching files
    pred_files = sorted([f for f in os.listdir(args.pred_dir) if f.endswith(args.ext)])
    gt_files = sorted([f for f in os.listdir(args.gt_dir) if f.endswith(args.ext)])
    
    # Match by filename
    pred_set = set(pred_files)
    gt_set = set(gt_files)
    common = sorted(pred_set & gt_set)
    
    if not common:
        # Try matching by case ID (strip suffixes like _seg, _pred, etc.)
        print("No exact filename matches. Trying fuzzy matching...")
        pred_map = {}
        for f in pred_files:
            # Extract case ID (first part before any suffix)
            case_id = f.replace(args.ext, '').split('_seg')[0].split('_pred')[0]
            pred_map[case_id] = f
        
        gt_map = {}
        for f in gt_files:
            case_id = f.replace(args.ext, '').split('_seg')[0].split('_pred')[0]
            gt_map[case_id] = f
        
        common_ids = sorted(set(pred_map.keys()) & set(gt_map.keys()))
        if not common_ids:
            print(f"ERROR: No matching cases found!")
            print(f"  Pred dir: {len(pred_files)} files")
            print(f"  GT dir:   {len(gt_files)} files")
            print(f"  Sample pred: {pred_files[:3]}")
            print(f"  Sample gt:   {gt_files[:3]}")
            return
        
        pairs = [(pred_map[cid], gt_map[cid]) for cid in common_ids]
        print(f"Found {len(pairs)} matching cases by case ID")
    else:
        pairs = [(f, f) for f in common]
        print(f"Found {len(pairs)} matching cases by filename")
    
    # Evaluate
    all_results = []
    et_scores, tc_scores, wt_scores = [], [], []
    
    for i, (pred_file, gt_file) in enumerate(pairs):
        pred_path = os.path.join(args.pred_dir, pred_file)
        gt_path = os.path.join(args.gt_dir, gt_file)
        
        pred = load_volume(pred_path)
        gt = load_volume(gt_path)
        
        scores = compute_brats_regions(pred, gt)
        scores['case'] = pred_file.replace(args.ext, '')
        all_results.append(scores)
        
        et_scores.append(scores['ET'])
        tc_scores.append(scores['TC'])
        wt_scores.append(scores['WT'])
        
        if (i + 1) % 50 == 0 or (i + 1) == len(pairs):
            print(f"  [{i+1}/{len(pairs)}] Running mean: "
                  f"ET={np.mean(et_scores):.4f} TC={np.mean(tc_scores):.4f} "
                  f"WT={np.mean(wt_scores):.4f} Mean={np.mean(et_scores + tc_scores + wt_scores) / 3:.4f}")
    
    # Summary
    summary = {
        'ET_dice_mean': float(np.mean(et_scores)),
        'ET_dice_std': float(np.std(et_scores)),
        'TC_dice_mean': float(np.mean(tc_scores)),
        'TC_dice_std': float(np.std(tc_scores)),
        'WT_dice_mean': float(np.mean(wt_scores)),
        'WT_dice_std': float(np.std(wt_scores)),
        'mean_dice': float(np.mean([np.mean(et_scores), np.mean(tc_scores), np.mean(wt_scores)])),
        'n_cases': len(pairs),
    }
    
    # Also compute per-label Dice for reference
    label_dice = {1: [], 2: [], 3: []}
    for pred_file, gt_file in pairs:
        pred = load_volume(os.path.join(args.pred_dir, pred_file))
        gt = load_volume(os.path.join(args.gt_dir, gt_file))
        for lbl in [1, 2, 3]:
            label_dice[lbl].append(dice_score(pred == lbl, gt == lbl))
    
    summary['per_label'] = {
        f'label_{lbl}_dice_mean': float(np.mean(label_dice[lbl]))
        for lbl in [1, 2, 3]
    }
    
    # Print results
    print("\n" + "=" * 65)
    print("nnU-Net BraTS Region-Based Dice Results")
    print("=" * 65)
    print(f"  Cases evaluated: {summary['n_cases']}")
    print(f"\n  Region-based (BraTS standard):")
    print(f"    ET Dice:   {summary['ET_dice_mean']:.4f} ± {summary['ET_dice_std']:.4f}")
    print(f"    TC Dice:   {summary['TC_dice_mean']:.4f} ± {summary['TC_dice_std']:.4f}")
    print(f"    WT Dice:   {summary['WT_dice_mean']:.4f} ± {summary['WT_dice_std']:.4f}")
    print(f"    Mean Dice: {summary['mean_dice']:.4f}")
    print(f"\n  Per-label (nnU-Net default):")
    for lbl in [1, 2, 3]:
        print(f"    Label {lbl} Dice: {summary['per_label'][f'label_{lbl}_dice_mean']:.4f}")
    print("=" * 65)
    
    # Save
    output = {
        'summary': summary,
        'per_case': all_results,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
