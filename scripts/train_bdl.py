#!/usr/bin/env python3
"""
Training script for Hybrid SAM-Med3D on BraTS2021.

Loss: Dice + Focal + Boundary (with linear warmup schedule).

Changes from original train_hybrid.py:
  - CompoundLoss: adds Boundary Loss with warmup schedule
  - ET-boosted Focal Loss: higher gamma on ET voxels
  - BraTS post-processing in validation
  - Per-component loss logging

Usage:
  # Full improvement (Boundary Loss + ET boost + post-processing)
  python scripts/train_bdl.py --exp_name hybrid_bdl_v1

  # Ablation: no boundary loss (= original baseline behavior)
  python scripts/train_bdl.py --exp_name abl_no_bdl --bdl_weight 0 --et_gamma_boost 0

  # Ablation: boundary loss only, no ET boost
  python scripts/train_bdl.py --exp_name abl_bdl_only --et_gamma_boost 0

  # Ablation: ET boost only, no boundary loss
  python scripts/train_bdl.py --exp_name abl_et_only --bdl_weight 0
"""

import sys, os
PROJECT = '.'
os.chdir(PROJECT)
sys.path.insert(0, os.path.join(PROJECT, 'SAM-Med3D'))
sys.path.insert(0, PROJECT)

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json, time, math

from data_processing.dataset import get_dataloaders
from models.proto_sam_hybrid import ProtoSAM_Hybrid
from losses import CompoundLoss, brats_postprocess


# ===========================================================================
# BraTS Region-Based Metrics
# ===========================================================================

class SegMetrics:
    """BraTS standard evaluation: Dice on ET, TC, WT regions."""

    def __init__(self, use_postprocess=True,
                 et_min=200, tc_min=200, wt_min=200):
        self.dice = {'ET': [], 'TC': [], 'WT': []}
        self.use_postprocess = use_postprocess
        self.et_min = et_min
        self.tc_min = tc_min
        self.wt_min = wt_min

    @staticmethod
    def _dice(p, g):
        if p.sum() == 0 and g.sum() == 0:
            return 1.0
        if p.sum() == 0 or g.sum() == 0:
            return 0.0
        return float(2.0 * (p & g).sum() / (p.sum() + g.sum()))

    def update(self, logits, target):
        pred = logits.argmax(1).cpu().numpy()
        gt = target.cpu().numpy()
        for b in range(pred.shape[0]):
            p = pred[b]
            g = gt[b]

            if self.use_postprocess:
                p = brats_postprocess(
                    p, et_min=self.et_min,
                    tc_min=self.tc_min, wt_min=self.wt_min
                )

            self.dice['ET'].append(self._dice(p == 3, g == 3))
            self.dice['TC'].append(self._dice(
                (p == 1) | (p == 3), (g == 1) | (g == 3)))
            self.dice['WT'].append(self._dice(p >= 1, g >= 1))

    def compute(self):
        r = {f'{k}_dice': float(np.mean(v)) for k, v in self.dice.items()}
        r['mean_dice'] = float(np.mean(
            [r['ET_dice'], r['TC_dice'], r['WT_dice']]))
        return r


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser('Hybrid SAM-Med3D Training (Boundary Loss)')

    # Experiment
    p.add_argument('--exp_name', type=str, default='hybrid_bdl_v1')

    # Architecture (defaults match your best config)
    p.add_argument('--lora_r', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--unfreeze_blocks', type=int, default=2)
    p.add_argument('--cnn_base_ch', type=int, default=32)

    # Training
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--accum', type=int, default=2)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--val_interval', type=int, default=5)
    p.add_argument('--no_amp', action='store_true',
                   help='Disable mixed precision')

    # Learning rates (match train_hybrid.py defaults)
    p.add_argument('--lr_enc', type=float, default=2e-4,
                   help='LR for LoRA + PatchEmbed')
    p.add_argument('--lr_cnn', type=float, default=5e-4,
                   help='LR for CNN branch')
    p.add_argument('--lr_dec', type=float, default=1e-3,
                   help='LR for Fusion + Decoder')
    p.add_argument('--lr_block', type=float, default=5e-5,
                   help='LR for unfrozen ViT blocks')
    p.add_argument('--wd', type=float, default=0.01)

    # --- Loss function (NEW) ---
    p.add_argument('--bdl_weight', type=float, default=1.0,
                   help='Max weight for Boundary Loss (0=disable)')
    p.add_argument('--bdl_warmup', type=int, default=20,
                   help='Epoch to start Boundary Loss')
    p.add_argument('--bdl_ramp', type=int, default=20,
                   help='Epochs to ramp Boundary Loss to full weight')
    p.add_argument('--focal_gamma', type=float, default=2.0,
                   help='Focal Loss gamma')
    p.add_argument('--et_gamma_boost', type=float, default=0.5,
                   help='Extra gamma for ET class (0=disable)')
    p.add_argument('--use_region_dice', action='store_true',
                   help='Use region-based Dice (ET/TC/WT) instead of per-class')
    p.add_argument('--focal_weight', type=float, default=1.0,
                   help='Focal Loss weight (0.5 recommended with region dice)')

    # --- Post-processing (NEW) ---
    p.add_argument('--et_min', type=int, default=200,
                   help='Min ET voxels (smaller → relabel as NCR)')
    p.add_argument('--tc_min', type=int, default=200,
                   help='Min TC voxels')
    p.add_argument('--wt_min', type=int, default=200,
                   help='Min WT voxels')
    p.add_argument('--et_aux_weight', type=float, default=0.0,
                   help='Weight for ET binary head BCE loss at 32³ (0=disabled)')
    p.add_argument('--use_vit_skip', action='store_true',
                   help='Enable ViT intermediate skip connections (blocks 3,6,9 → 64/32/16³)')
    p.add_argument('--deep_supervision', action='store_true',
                   help='Enable deep supervision with auxiliary heads at 16/32/64 resolutions')
    p.add_argument('--no_postprocess', action='store_true',
                   help='Disable post-processing in validation')

    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print(f"Hybrid SAM-Med3D + Boundary Loss | {args.exp_name}")
    print(f"  LoRA r={args.lora_r}  unfreeze={args.unfreeze_blocks}  "
          f"CNN_ch={args.cnn_base_ch}")
    print(f"  Loss: Dice + Focal(g={args.focal_gamma}, ET+{args.et_gamma_boost})"
          f" + Boundary(w={args.bdl_weight}, warm={args.bdl_warmup},"
          f" ramp={args.bdl_ramp})")
    pp_str = (f'ET>={args.et_min} TC>={args.tc_min} WT>={args.wt_min}'
              if not args.no_postprocess else 'OFF')
    print(f"  PostProc: {pp_str}")
    print("=" * 70)

    device = torch.device('cuda')
    ckpt_dir = os.path.join(PROJECT, 'checkpoints', args.exp_name)
    log_dir = os.path.join(PROJECT, 'logs', args.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("\n[1/4] Loading data...")
    train_loader, val_loader, _, _ = get_dataloaders(
        PROJECT, batch_size=args.batch_size, num_workers=4,
        strong_augment=False
    )
    print(f"  Train: {len(train_loader)} batches  "
          f"Val: {len(val_loader)} batches")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("\n[2/4] Building model...")
    model = ProtoSAM_Hybrid(
        sam_checkpoint=os.path.join(
            PROJECT, 'SAM-Med3D/ckpt/sam_med3d_turbo.pth'),
        num_classes=4,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        in_channels=4,
        unfreeze_blocks=args.unfreeze_blocks,
        cnn_base_ch=args.cnn_base_ch,
        #use_vit_skip=args.use_vit_skip,
    ).to(device)

    stats = model.get_param_stats()
    print(f"  Total:     {stats['total']/1e6:.2f}M")
    print(f"  Trainable: {stats['trainable']/1e6:.2f}M "
          f"({stats['trainable_ratio']:.1f}%)")

    # ------------------------------------------------------------------
    # Loss & Optimizer
    # ------------------------------------------------------------------
    loss_fn = CompoundLoss(
        num_classes=4,
        focal_gamma=args.focal_gamma,
        et_gamma_boost=args.et_gamma_boost,
        bdl_weight=args.bdl_weight,
        bdl_warmup=args.bdl_warmup,
        bdl_ramp=args.bdl_ramp,
        use_region_dice=args.use_region_dice,    # NEW
        focal_weight=args.focal_weight,          # NEW
    ).to(device)

    param_groups = model.get_param_groups(
        lr_enc=args.lr_enc, lr_cnn=args.lr_cnn,
        lr_dec=args.lr_dec, lr_block=args.lr_block,
        weight_decay=args.wd
    )
    optimizer = torch.optim.AdamW(param_groups)
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    print(f"\n[3/4] Optimizer ({len(param_groups)} groups):")
    for pg in param_groups:
        n = sum(p.numel() for p in pg['params'])
        print(f"  {pg['name']}: {n/1e6:.2f}M  lr={pg['lr']:.1e}")

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch, best_dice = 0, 0.0
    train_hist, val_hist = [], []
    resume = os.path.join(ckpt_dir, 'latest.pth')
    if os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device)
        md = model.state_dict()
        loaded = {k: v for k, v in ckpt['model_state_dict'].items()
                  if k in md and v.shape == md[k].shape}
        md.update(loaded)
        model.load_state_dict(md)
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except Exception:
            pass
        start_epoch = ckpt['epoch'] + 1
        best_dice = ckpt.get('best_dice', 0.0)
        train_hist = ckpt.get('train_hist', [])
        val_hist = ckpt.get('val_hist', [])
        print(f"  Resumed epoch {start_epoch}, best={best_dice:.4f}")

    # ------------------------------------------------------------------
    # LR schedule: warmup + cosine
    # ------------------------------------------------------------------
    def lr_scale(epoch):
        if epoch < args.warmup:
            return (epoch + 1) / args.warmup
        t = (epoch - args.warmup) / max(args.epochs - args.warmup, 1)
        return max(0.5 * (1 + math.cos(math.pi * t)), 1e-2)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\n[4/4] Training epochs {start_epoch}..{args.epochs}  "
          f"eff_batch={args.batch_size * args.accum}")
    print("=" * 70)

    use_amp = not args.no_amp
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        s = lr_scale(epoch)
        for i, pg in enumerate(optimizer.param_groups):
            pg['lr'] = base_lrs[i] * s

        model.train()
        epoch_losses = []
        epoch_components = []
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            img = batch['image'].to(device)
            lbl = batch['label'].to(device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                out = model(img)
                if args.deep_supervision:
                    # out = [main(128³), aux_64, aux_32, aux_16, (aux_et_32 if et_aux>0)]
                    # First 4 are 4-class outputs; weights normalized to sum=1
                    weights = [1.0, 0.5, 0.25, 0.125]
                    weights = [w / sum(weights) for w in weights]
                    loss = 0.0
                    comp = None
                    for w, aux_logits in zip(weights, out[:4]):
                        if aux_logits.shape[-1] != lbl.shape[-1]:
                            lbl_ds = F.interpolate(
                                lbl.unsqueeze(1).float(),
                                size=aux_logits.shape[-3:],
                                mode='nearest'
                            ).squeeze(1).long()
                        else:
                            lbl_ds = lbl
                        l_i, c_i = loss_fn(aux_logits, lbl_ds, epoch=epoch)
                        loss = loss + w * l_i
                        if comp is None:
                            comp = c_i
                    
                    # Optional binary ET head at 32³ (ET sweet spot)
                    if args.et_aux_weight > 0 and len(out) >= 5:
                        et_logits_32 = out[4]  # (B, 1, 32, 32, 32)
                        # Downsample GT to 32³ and binarize for ET (class 3)
                        lbl_32 = F.interpolate(
                            lbl.unsqueeze(1).float(),
                            size=et_logits_32.shape[-3:],
                            mode='nearest'
                        ).squeeze(1)
                        et_target_32 = (lbl_32 == 3).float().unsqueeze(1)
                        et_loss = F.binary_cross_entropy_with_logits(
                            et_logits_32, et_target_32
                        )
                        loss = loss + args.et_aux_weight * et_loss
                        # Track in components for logging
                        comp = dict(comp) if comp else {}
                        comp['et_aux'] = et_loss.item()
                        comp['et_aux_w'] = args.et_aux_weight
                    
                    logits = out[0]
                else:
                    logits = out
                    loss, comp = loss_fn(logits, lbl, epoch=epoch)

            scaler.scale(loss / args.accum).backward()

            if (step + 1) % args.accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_losses.append(loss.item())
            epoch_components.append(comp)

            if (step + 1) % 100 == 0:
                print(f"  E{epoch+1} S{step+1}/{len(train_loader)} "
                      f"loss={loss.item():.4f} "
                      f"[D={comp['dice']:.3f} F={comp['focal']:.3f} "
                      f"B={comp['boundary']:.3f}*{comp['bdl_weight']:.2f}]")
            del logits, loss
            torch.cuda.empty_cache()

        # --- Epoch summary ---
        avg_loss = float(np.mean(epoch_losses))
        avg_comp = {
            k: float(np.mean([c[k] for c in epoch_components]))
            for k in epoch_components[0]
        }
        train_hist.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'lr': optimizer.param_groups[0]['lr'],
            **{f'loss_{k}': v for k, v in avg_comp.items()},
        })

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"loss={avg_loss:.4f} "
              f"[D={avg_comp['dice']:.3f} F={avg_comp['focal']:.3f} "
              f"B={avg_comp['boundary']:.3f}*{avg_comp['bdl_weight']:.2f}] | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | "
              f"{time.time()-t0:.0f}s")

        # --- Validation ---
        if (epoch + 1) % args.val_interval == 0:
            tv = time.time()
            model.eval()
            metrics = SegMetrics(
                use_postprocess=not args.no_postprocess,
                et_min=args.et_min,
                tc_min=args.tc_min,
                wt_min=args.wt_min,
            )

            with torch.no_grad():
                for batch in val_loader:
                    img = batch['image'].to(device)
                    lbl = batch['label'].to(device)
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        logits = model(img)
                    metrics.update(logits, lbl)
                    del logits
                    torch.cuda.empty_cache()

            vr = metrics.compute()
            vr['epoch'] = epoch + 1
            val_hist.append(vr)

            is_best = vr['mean_dice'] > best_dice
            if is_best:
                best_dice = vr['mean_dice']

            print(f"  Val: ET={vr['ET_dice']:.4f} TC={vr['TC_dice']:.4f} "
                  f"WT={vr['WT_dice']:.4f} Mean={vr['mean_dice']:.4f} "
                  f"{'* BEST' if is_best else ''} | {time.time()-tv:.0f}s"
                 )
            scales = [p.data.item() for n, p in model.named_parameters() if 'skip_scale' in n]
            print(f"  skip_scales: {[f'{s:.3f}' for s in scales]}")
    

            if is_best:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': {
                        k: v for k, v in model.state_dict().items()
                        if any(t in k for t in [
                            'patch_embed', 'lora', 'decoder',
                            'cnn_branch', 'bottleneck_fusion', 'blocks.'])
                        and ('image_encoder' not in k or 'lora' in k
                             or 'patch_embed' in k
                             or any(f'blocks.{j}' in k
                                    for j in range(
                                        12 - args.unfreeze_blocks, 12)))
                    },
                    'best_dice': best_dice,
                    'config': vars(args),
                    'param_stats': stats,
                }, os.path.join(ckpt_dir, 'best.pth'))

        # --- Checkpoint ---
        torch.save({
            'epoch': epoch,
            'model_state_dict': {
                k: v for k, v in model.state_dict().items()
                if any(t in k for t in [
                    'patch_embed', 'lora', 'decoder',
                    'cnn_branch', 'bottleneck_fusion', 'blocks.'])
                and ('image_encoder' not in k or 'lora' in k
                     or 'patch_embed' in k
                     or any(f'blocks.{j}' in k
                            for j in range(
                                12 - args.unfreeze_blocks, 12)))
            },
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dice': best_dice,
            'train_hist': train_hist,
            'val_hist': val_hist,
        }, os.path.join(ckpt_dir, 'latest.pth'))

        with open(os.path.join(log_dir, 'log.json'), 'w') as f:
            json.dump({
                'config': vars(args),
                'param_stats': stats,
                'train': train_hist,
                'val': val_hist,
                'best_dice': best_dice,
            }, f, indent=2)

    print(f"\nDone. Best Mean Dice: {best_dice:.4f}")


if __name__ == '__main__':
    main()
