#!/usr/bin/env python3
"""
Correct per-volume Dice evaluation for GBT-SAM.

BraTS standard protocol:
  1. For each case, predict all 155 slices
  2. Stack predictions back into 3D volume
  3. Compute ONE Dice score for the entire volume
  4. Average Dice across all 125 val cases

This is NOT what function.validation_sam does (it averages per-chunk dice
which inflates the score because empty chunks get Dice=1.0).
"""
import sys
sys.modules["pyarrow"] = None
sys.modules["pyarrow.lib"] = None

sys.argv = ['val_gbt_volume.py',
    '-net', 'sam', '-mod', 'sam_lora',
    '-sam_ckpt', 'logs/brats2021_wt_2026_04_07_16_42_08/Model/best_dice',
    '-weights', 'logs/brats2021_wt_2026_04_07_16_42_08/Model/best_dice',
    '-b', '1', '-dataset', 'brats_2020', '-thd', 'True',
    '-data_path', './data', '-w', '4', '-four_chan', 'True',
    '-mode', 'Validation', '-evl_chunk', '4',
    '-exp_name', 'val_gbt_volume']

import os, json
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from tqdm import tqdm

import cfg_valid
from dataset import Brats
from utils import get_network, generate_click_prompt
from monai.transforms import Compose, CropForegroundd, Orientationd

# ── Setup ──────────────────────────────────────────────────
args = cfg_valid.parse_args()
GPUdevice = torch.device('cuda', args.gpu_device)
print(f"Device: {GPUdevice}")

# ── Build model ────────────────────────────────────────────
print("Building model...")
net = get_network(args, args.net, use_gpu=args.gpu,
                   gpu_device=GPUdevice, distribution=args.distributed)

# ── Load checkpoint ─────────────────────────────────────────
print(f"Loading checkpoint: {args.weights}")
checkpoint = torch.load(args.weights, map_location=GPUdevice, weights_only=False)
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
elif isinstance(checkpoint, dict) and 'model' in checkpoint:
    state_dict = checkpoint['model']
else:
    state_dict = checkpoint

new_state = OrderedDict()
for k, v in state_dict.items():
    name = k.replace('module.', '') if k.startswith('module.') else k
    new_state[name] = v
net.load_state_dict(new_state, strict=False)
net.eval()
print("Model loaded.")

# ── Load val split ─────────────────────────────────────────
SPLIT_F = 'data/splits/brats2021_split.json'
with open(SPLIT_F) as f:
    splits = json.load(f)
val_cases = splits['val']
print(f"Val cases: {len(val_cases)}")

# ── Build dataset (no transforms - we need raw volumes) ─────
val_transforms = Compose([
    CropForegroundd(keys=["image", "label"], source_key="image"),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
])
dataset = Brats(args, args.data_path, mode='Validation', transform=val_transforms)

# Filter to val cases only
data_root = os.path.join(args.data_path, 'brats_2020')
dataset.subfolders = [os.path.join(data_root, c) for c in val_cases
                      if os.path.isdir(os.path.join(data_root, c))]
print(f"Filtered dataset size: {len(dataset)}")

# ── Volume-level Dice ──────────────────────────────────────
def volume_dice(pred_vol, gt_vol):
    """Compute Dice over the entire 3D volume (not per-slice)."""
    pred = (pred_vol > 0).astype(np.uint8)
    gt = (gt_vol > 0).astype(np.uint8)
    inter = np.logical_and(pred, gt).sum()
    psum = pred.sum()
    gsum = gt.sum()
    if psum == 0 and gsum == 0:
        return 1.0  # both empty — trivially correct
    if psum + gsum == 0:
        return 0.0
    return 2.0 * inter / (psum + gsum)

def volume_iou(pred_vol, gt_vol):
    pred = (pred_vol > 0).astype(np.uint8)
    gt = (gt_vol > 0).astype(np.uint8)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return inter / union

# ── Inference loop ─────────────────────────────────────────
evl_ch = int(args.evl_chunk) if args.evl_chunk else None
print(f"evl_chunk: {evl_ch}")

all_dice, all_iou = [], []

with torch.no_grad():
    for idx in tqdm(range(len(dataset)), desc='Per-volume eval'):
        pack = dataset[idx]
        # image: (4, 240, 240, 155), label: (1, 240, 240, 155) after transforms
        imgsw = pack['image'].unsqueeze(0).to(dtype=torch.float32, device=GPUdevice)
        masksw = pack['label'].unsqueeze(0).to(dtype=torch.float32, device=GPUdevice)

        # Generate click prompt from GT (same as training protocol)
        imgsw, ptw, masksw = generate_click_prompt(imgsw, masksw)

        # Process in chunks along depth (dim -1)
        D = imgsw.size(-1)
        pred_slices = []
        gt_slices = []
        buoy = 0

        while buoy < D:
            end = min(buoy + evl_ch, D)
            pt = ptw[:, :, buoy:end]
            imgs = imgsw[..., buoy:end]
            masks = masksw[..., buoy:end]
            buoy = end

            if args.thd:
                pt = rearrange(pt, 'b n d -> (b d) n')
                imgs = rearrange(imgs, 'b c h w d -> (b d) c h w')
                masks = rearrange(masks, 'b c h w d -> (b d) c h w')
                point_labels = torch.ones(imgs.size(0))
                pt = torch.Tensor(np.array([
                    (pt[i].detach().cpu().numpy() * (args.out_size, args.out_size)) / masks.shape[2:]
                    for i in range(pt.shape[0])
                ]))
                imgs = torchvision.transforms.Resize(
                    (args.image_size, args.image_size), antialias=None
                )(imgs)
                masks = torchvision.transforms.Resize(
                    (args.out_size, args.out_size), antialias=None
                )(masks)

            coords = torch.as_tensor(pt, dtype=torch.float, device=GPUdevice)
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            coords, labels = coords[None, :, :], labels[None, :]
            pt_tuple = (coords, labels)

            imgs = imgs.to(dtype=torch.float32, device=GPUdevice)

            imge = net.image_encoder(imgs)
            se, de = net.prompt_encoder(points=pt_tuple, boxes=None, masks=None)
            pred, _ = net.mask_decoder(
                image_embeddings=imge,
                image_pe=net.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de,
                multimask_output=False,
            )
            pred = F.interpolate(pred, size=(args.out_size, args.out_size))
            pred_bin = (torch.sigmoid(pred) > 0.5).float()

            # pred_bin shape: (chunk_size, 1, H, W)
            pred_slices.append(pred_bin.cpu().numpy())
            gt_slices.append(masks.cpu().numpy())

        # Stack into 3D volume: (total_slices, 1, H, W) → (H, W, D)
        pred_vol = np.concatenate(pred_slices, axis=0)[:, 0]  # (D, H, W)
        gt_vol = np.concatenate(gt_slices, axis=0)[:, 0]       # (D, H, W)

        d = volume_dice(pred_vol, gt_vol)
        i = volume_iou(pred_vol, gt_vol)
        all_dice.append(d)
        all_iou.append(i)

        del imgsw, masksw, pred_slices, gt_slices, pred_vol, gt_vol
        torch.cuda.empty_cache()

# ── Report ─────────────────────────────────────────────────
mean_dice = float(np.mean(all_dice))
std_dice = float(np.std(all_dice))
mean_iou = float(np.mean(all_iou))

print(f"\n{'='*60}")
print(f"GBT-SAM WT — PER-VOLUME Dice (BraTS standard protocol)")
print(f"{'='*60}")
print(f"N cases:   {len(all_dice)}")
print(f"Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}")
print(f"Mean IoU:  {mean_iou:.4f}")
print(f"Median:    {float(np.median(all_dice)):.4f}")
print(f"Min/Max:   {float(np.min(all_dice)):.4f} / {float(np.max(all_dice)):.4f}")
print(f"{'='*60}\n")

# Save
results = {
    'method': 'GBT-SAM (med-sam-brain), per-volume Dice',
    'region': 'WT',
    'n_cases': len(all_dice),
    'dice_mean': round(mean_dice, 4),
    'dice_std': round(std_dice, 4),
    'dice_median': round(float(np.median(all_dice)), 4),
    'iou_mean': round(mean_iou, 4),
    'per_case_dice': [round(d, 4) for d in all_dice],
}
out_dir = 'baseline_results/gbt_sam'
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'results_volume.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved to {out_dir}/results_volume.json")
