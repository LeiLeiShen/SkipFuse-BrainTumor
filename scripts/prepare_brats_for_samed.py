#!/usr/bin/env python3
"""
Prepare BraTS2021 data for SAMed training.
==========================================
Converts 3D npy volumes → 2D axial slices for training (.npz)
                        → 3D h5 volumes for validation (.npy.h5)

We use 3 MRI modalities (T1ce, T2, FLAIR) as pseudo-RGB channels
instead of SAMed's default single-channel-repeated-3x approach.
This gives the model more information while keeping 3 input channels.

Usage:
  cd .
  python scripts/prepare_brats_for_samed.py
"""

import sys, os, json
import numpy as np
import h5py


PROJECT  = '.'
DATA_DIR = os.path.join(PROJECT, 'data/processed_BraTS2021')
SPLIT_FILE = os.path.join(PROJECT, 'data/splits/brats2021_split.json')

# Output directories
OUT_BASE = os.path.join(PROJECT, 'baselines/SAMed/data/BraTS2021')
OUT_TRAIN = os.path.join(OUT_BASE, 'train_slices')
OUT_VAL   = os.path.join(OUT_BASE, 'val_volumes')
OUT_LIST  = os.path.join(OUT_BASE, 'lists')

# Channels: T1ce=1, T2=2, FLAIR=3 (drop T1=0, most redundant with T1ce)
CHANNELS = [1, 2, 3]


def main():
    os.makedirs(OUT_TRAIN, exist_ok=True)
    os.makedirs(OUT_VAL, exist_ok=True)
    os.makedirs(OUT_LIST, exist_ok=True)

    with open(SPLIT_FILE) as f:
        splits = json.load(f)

    train_cases = splits['train']
    val_cases = splits['val']
    print(f"Train: {len(train_cases)} cases, Val: {len(val_cases)} cases")

    # ── Training: Extract 2D axial slices ──────────────────────────
    print("\n[1/2] Creating training slices...")
    train_slice_names = []
    total_slices = 0
    skipped_empty = 0

    for ci, case in enumerate(train_cases):
        image = np.load(os.path.join(DATA_DIR, f'{case}_image.npy'))  # (4, 128, 128, 128)
        label = np.load(os.path.join(DATA_DIR, f'{case}_label.npy'))  # (128, 128, 128)

        for z in range(128):
            lbl_slice = label[z, :, :]  # (128, 128)

            # Skip empty slices (no brain tissue) to speed up training
            img_check = image[:, z, :, :]
            if img_check.max() == 0:
                skipped_empty += 1
                continue

            # Stack 3 modalities as (H, W, 3) — SAMed expects (H, W) or (H, W, C)
            # But SAMed's RandomGenerator expects (H, W) single channel,
            # then repeats to 3. We'll provide (H, W) FLAIR and let it repeat.
            # Simpler and faithful to SAMed's design.
            img_slice = image[3, z, :, :]  # FLAIR channel, (128, 128)

            slice_name = f'{case}_slice{z:03d}'
            np.savez_compressed(
                os.path.join(OUT_TRAIN, f'{slice_name}.npz'),
                image=img_slice.astype(np.float32),
                label=lbl_slice.astype(np.int8),
            )
            train_slice_names.append(slice_name)
            total_slices += 1

        if (ci + 1) % 100 == 0 or (ci + 1) == len(train_cases):
            print(f"  [{ci+1}/{len(train_cases)}] {total_slices} slices "
                  f"({skipped_empty} empty skipped)")

    # Write train.txt
    with open(os.path.join(OUT_LIST, 'train.txt'), 'w') as f:
        for name in train_slice_names:
            f.write(name + '\n')
    print(f"  Total training slices: {total_slices}")
    print(f"  Saved to {OUT_TRAIN}")

    # ── Validation: Save as h5 volumes ─────────────────────────────
    print("\n[2/2] Creating validation volumes...")
    val_vol_names = []

    for ci, case in enumerate(val_cases):
        image = np.load(os.path.join(DATA_DIR, f'{case}_image.npy'))  # (4, 128, 128, 128)
        label = np.load(os.path.join(DATA_DIR, f'{case}_label.npy'))  # (128, 128, 128)

        # Save as h5: image=(128, 128, 128) single-channel FLAIR, label=(128, 128, 128)
        img_vol = image[3]  # FLAIR, (128, 128, 128)

        h5_path = os.path.join(OUT_VAL, f'{case}.npy.h5')
        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset('image', data=img_vol.astype(np.float32))
            hf.create_dataset('label', data=label.astype(np.int8))

        val_vol_names.append(case)

        if (ci + 1) % 25 == 0 or (ci + 1) == len(val_cases):
            print(f"  [{ci+1}/{len(val_cases)}]")

    # Write test_vol.txt
    with open(os.path.join(OUT_LIST, 'test_vol.txt'), 'w') as f:
        for name in val_vol_names:
            f.write(name + '\n')
    print(f"  Total validation volumes: {len(val_vol_names)}")
    print(f"  Saved to {OUT_VAL}")

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Data preparation complete!")
    print(f"  Train slices: {OUT_TRAIN} ({total_slices} files)")
    print(f"  Val volumes:  {OUT_VAL} ({len(val_vol_names)} files)")
    print(f"  List dir:     {OUT_LIST}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
