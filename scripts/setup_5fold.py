#!/usr/bin/env python3
"""
Generate 5-fold CV splits for BraTS2021 and update train_repro.py to support --fold.

Usage:
  python setup_5fold.py
"""

import os, json, ast
import numpy as np

PROJECT = '.'

# ===========================================================================
# Step 1: Generate 5-fold splits
# ===========================================================================
print("Step 1: Generating 5-fold splits")

split_file = os.path.join(PROJECT, 'data/splits/brats2021_split.json')
with open(split_file) as f:
    orig = json.load(f)

# Merge train + val, keep test separate
all_cases = sorted(orig['train'] + orig['val'])  # 1125 cases
test_cases = orig['test']  # 126 cases
print(f"  Train+Val pool: {len(all_cases)} cases")
print(f"  Test (held out): {len(test_cases)} cases")

# Shuffle deterministically
rng = np.random.RandomState(42)
indices = rng.permutation(len(all_cases))

# Split into 5 folds
folds = {}
fold_size = len(all_cases) // 5  # 225
for i in range(5):
    start = i * fold_size
    end = start + fold_size if i < 4 else len(all_cases)  # last fold gets remainder
    val_idx = indices[start:end]
    train_idx = np.concatenate([indices[:start], indices[end:]])
    
    folds[f'fold_{i}'] = {
        'train': [all_cases[j] for j in sorted(train_idx)],
        'val': [all_cases[j] for j in sorted(val_idx)],
        'test': test_cases,
    }
    print(f"  Fold {i}: train={len(folds[f'fold_{i}']['train'])}, "
          f"val={len(folds[f'fold_{i}']['val'])}")

out_path = os.path.join(PROJECT, 'data/splits/brats2021_5fold.json')
with open(out_path, 'w') as f:
    json.dump(folds, f, indent=2)
print(f"  Saved to {out_path}")

# ===========================================================================
# Step 2: Update train_repro.py to support --fold
# ===========================================================================
print("\nStep 2: Updating train_repro.py with --fold support")

train_path = os.path.join(PROJECT, 'scripts/train_repro.py')
with open(train_path) as f:
    src = f.read()

# Add --fold argument after --deep_supervision
old_args = "    p.add_argument('--deep_supervision', action='store_true')"
new_args = """    p.add_argument('--deep_supervision', action='store_true')
    p.add_argument('--fold', type=int, default=-1,
                   help='5-fold CV fold index (0-4). -1 = use original split.')"""
src = src.replace(old_args, new_args)

# Replace the data loading section to support fold
old_data = """    # Data
    train_loader, val_loader, _, _ = get_dataloaders(
        PROJECT, batch_size=args.batch_size, num_workers=4, strong_augment=False)
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")"""

new_data = """    # Data
    if args.fold >= 0:
        # 5-fold CV mode
        fold_split = os.path.join(PROJECT, 'data/splits/brats2021_5fold.json')
        with open(fold_split) as f:
            fold_data = json.load(f)
        fold_key = f'fold_{args.fold}'
        assert fold_key in fold_data, f"Fold {args.fold} not found in {fold_split}"
        
        from data_processing.dataset import BraTS2021Dataset
        from torch.utils.data import DataLoader
        data_dir = os.path.join(PROJECT, 'data/processed_BraTS2021')
        train_dataset = BraTS2021Dataset(data_dir, fold_data[fold_key]['train'], augment=True)
        val_dataset = BraTS2021Dataset(data_dir, fold_data[fold_key]['val'], augment=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)
        print(f"5-Fold CV: fold {args.fold}")
    else:
        # Original split mode
        train_loader, val_loader, _, _ = get_dataloaders(
            PROJECT, batch_size=args.batch_size, num_workers=4, strong_augment=False)
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")"""

src = src.replace(old_data, new_data)

with open(train_path, 'w') as f:
    f.write(src)

ast.parse(src)
print("  train_repro.py updated and syntax verified")

# ===========================================================================
# Step 3: Verify
# ===========================================================================
print("\nStep 3: Verification")
print(f"  --fold argument added: {'--fold' in src}")
print(f"  5-fold split loading: {'brats2021_5fold.json' in src}")
print(f"  BraTS2021Dataset imported: {'BraTS2021Dataset' in src}")
print(f"  Original split fallback: {'fold >= 0' in src}")

# Check BraTS2021Dataset constructor signature
ds_path = os.path.join(PROJECT, 'data_processing/dataset.py')
with open(ds_path) as f:
    ds_src = f.read()
# Find __init__ signature
import re
match = re.search(r'class BraTS2021Dataset.*?def __init__\(self,(.*?)\)', ds_src, re.DOTALL)
if match:
    print(f"  BraTS2021Dataset.__init__ args: {match.group(1).strip()}")

print("\n" + "=" * 60)
print("SETUP COMPLETE")
print("=" * 60)
print("""
Run each fold separately:

  cd .

  # Fold 0
  mkdir -p logs/hybrid_c64_ds_fold0
  nohup python scripts/train_repro.py --exp_name hybrid_c64_ds_fold0 --cnn_base_ch 64 --deep_supervision --fold 0 > logs/hybrid_c64_ds_fold0/console.log 2>&1 &

  # Fold 1 (after fold 0 finishes)
  mkdir -p logs/hybrid_c64_ds_fold1
  nohup python scripts/train_repro.py --exp_name hybrid_c64_ds_fold1 --cnn_base_ch 64 --deep_supervision --fold 1 > logs/hybrid_c64_ds_fold1/console.log 2>&1 &

  # ... same for fold 2, 3, 4

  # Collect results:
  python3 -c "
  import json, numpy as np
  dices = []
  for i in range(5):
      d = json.load(open(f'logs/hybrid_c64_ds_fold{i}/log.json'))
      dices.append(d['best_dice'])
      print(f'Fold {i}: {d[\"best_dice\"]:.4f}')
  print(f'Mean: {np.mean(dices):.4f} +/- {np.std(dices):.4f}')
  "
""")
