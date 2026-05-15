#!/usr/bin/env python3
"""
Unified ablation training script for ProtoSAM-Med3D Hybrid.

All ablation dimensions controllable via command-line arguments.

Usage:
  # Full hybrid (proposed method)
  python scripts/train_ablation.py --exp_name full_hybrid

  # Ablation: no skip connections
  python scripts/train_ablation.py --exp_name no_skip --no_skip

  # Ablation: no bottleneck fusion
  python scripts/train_ablation.py --exp_name no_fusion --no_fusion

  # Ablation: no CNN branch (= v3.1 SimpleDecoder)
  python scripts/train_ablation.py --exp_name no_cnn --no_cnn

  # Ablation: LoRA rank in hybrid
  python scripts/train_ablation.py --exp_name hybrid_r4 --lora_r 4

  # Ablation: CNN width
  python scripts/train_ablation.py --exp_name cnn_ch16 --cnn_base_ch 16
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
from models.proto_sam_ablation import ProtoSAM_Ablation


# ============================================================================
# Loss functions
# ============================================================================

class DiceLoss(nn.Module):
    def __init__(self, num_classes=4, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, pred_logits, target):
        pred_soft = F.softmax(pred_logits, dim=1)
        target_oh = F.one_hot(target, self.num_classes).permute(0, 4, 1, 2, 3).float()
        dice_per_class = []
        for c in range(self.num_classes):
            p, g = pred_soft[:, c], target_oh[:, c]
            inter = (p * g).sum()
            card = p.sum() + g.sum()
            if card < self.smooth:
                continue
            dice_per_class.append(1.0 - (2 * inter + self.smooth) / (card + self.smooth))
        if not dice_per_class:
            return torch.tensor(0.0, device=pred_logits.device, requires_grad=True)
        return torch.stack(dice_per_class).mean()


class FocalLoss(nn.Module):
    def __init__(self, num_classes=4, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        freq = torch.tensor([98.93, 0.29, 0.31, 0.79])
        alpha = 1.0 / freq
        alpha = alpha / alpha.sum() * num_classes
        self.register_buffer('alpha', alpha.float())
    
    def forward(self, pred_logits, target):
        ce = F.cross_entropy(pred_logits, target, reduction='none')
        pt = F.softmax(pred_logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
        focal_w = (1.0 - pt) ** self.gamma
        alpha_w = self.alpha.to(target.device)[target]
        return (alpha_w * focal_w * ce).mean()


# ============================================================================
# Metrics
# ============================================================================

class SegMetrics:
    def __init__(self):
        self.dice = {'ET': [], 'TC': [], 'WT': []}
    
    @staticmethod
    def dc(p, g):
        if p.sum() == 0 and g.sum() == 0: return 1.0
        if p.sum() == 0 or g.sum() == 0: return 0.0
        return float(2.0 * (p & g).sum() / (p.sum() + g.sum()))
    
    def update(self, logits, target):
        p = logits.argmax(1).cpu().numpy()
        g = target.cpu().numpy()
        for b in range(p.shape[0]):
            self.dice['ET'].append(self.dc(p[b] == 3, g[b] == 3))
            self.dice['TC'].append(self.dc((p[b] == 1) | (p[b] == 3), (g[b] == 1) | (g[b] == 3)))
            self.dice['WT'].append(self.dc(p[b] >= 1, g[b] >= 1))
    
    def compute(self):
        r = {}
        for reg in ['ET', 'TC', 'WT']:
            r[f'{reg}_dice_mean'] = float(np.mean(self.dice[reg]))
        r['mean_dice'] = float(np.mean([r[f'{reg}_dice_mean'] for reg in ['ET', 'TC', 'WT']]))
        return r


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='ProtoSAM-Med3D Ablation Training')
    
    # Experiment identity
    p.add_argument('--exp_name', type=str, required=True, help='Experiment name (used for dirs)')
    
    # Architecture ablation flags
    p.add_argument('--no_cnn', action='store_true', help='Disable CNN branch entirely (= SimpleDecoder)')
    p.add_argument('--no_skip', action='store_true', help='Disable skip connections (CNN runs but skips zeroed)')
    p.add_argument('--no_fusion', action='store_true', help='Disable bottleneck fusion (ViT-only to decoder)')
    
    # Architecture hyperparameters
    p.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    p.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    p.add_argument('--unfreeze_blocks', type=int, default=2, help='ViT blocks to unfreeze')
    p.add_argument('--cnn_base_ch', type=int, default=32, help='CNN branch base channels')
    
    # Training hyperparameters
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--accum_steps', type=int, default=2)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--val_interval', type=int, default=5)
    
    # Learning rates
    p.add_argument('--lr_block', type=float, default=5e-5)
    p.add_argument('--lr_lora', type=float, default=2e-4)
    p.add_argument('--lr_cnn', type=float, default=5e-4)
    p.add_argument('--lr_dec', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=0.01)
    
    return p.parse_args()


def main():
    args = parse_args()
    
    # Derive flags
    use_cnn = not args.no_cnn
    use_skip = not args.no_skip and use_cnn  # skip requires CNN
    use_fusion = not args.no_fusion and use_cnn  # fusion requires CNN
    
    print("=" * 70)
    print(f"ProtoSAM-Med3D Ablation: {args.exp_name}")
    print("=" * 70)
    print(f"  CNN branch:  {'ON' if use_cnn else 'OFF'}")
    print(f"  Skip conn:   {'ON' if use_skip else 'OFF'}")
    print(f"  Bottleneck:  {'ON' if use_fusion else 'OFF'}")
    print(f"  LoRA rank:   {args.lora_r}")
    print(f"  Unfreeze:    {args.unfreeze_blocks} blocks")
    print(f"  CNN width:   {args.cnn_base_ch}")
    print("=" * 70)
    
    device = torch.device('cuda')
    ckpt_dir = os.path.join(PROJECT, 'checkpoints_ablation', args.exp_name)
    log_dir = os.path.join(PROJECT, 'logs_ablation', args.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # ================================================================
    # Data
    # ================================================================
    print("\n[1/4] Loading data...")
    train_loader, val_loader, _, _ = get_dataloaders(
        PROJECT, batch_size=args.batch_size, num_workers=4, strong_augment=False
    )
    print(f"  Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    
    # ================================================================
    # Model
    # ================================================================
    print("\n[2/4] Building model...")
    model = ProtoSAM_Ablation(
        sam_checkpoint=os.path.join(PROJECT, 'SAM-Med3D/ckpt/sam_med3d_turbo.pth'),
        num_classes=4,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        in_channels=4,
        unfreeze_blocks=args.unfreeze_blocks,
        use_cnn=use_cnn,
        use_skip=use_skip,
        use_fusion=use_fusion,
        cnn_base_ch=args.cnn_base_ch,
    ).to(device)
    
    stats = model.get_param_stats()
    print(f"  Config: {stats['config_name']}")
    print(f"  Total:     {stats['total']/1e6:.2f}M")
    print(f"  Trainable: {stats['trainable']/1e6:.2f}M ({stats['trainable_ratio']:.1f}%)")
    for comp in ['lora', 'patch_embed', 'unfrozen_blocks', 'cnn_branch', 'bottleneck_fusion', 'decoder']:
        if stats[comp] > 0:
            print(f"    {comp}: {stats[comp]/1e6:.2f}M")
    
    # ================================================================
    # Loss, Optimizer
    # ================================================================
    dice_loss = DiceLoss(4)
    focal_loss = FocalLoss(4).to(device)
    
    param_groups = model.get_param_groups(
        lr_enc=args.lr_lora, lr_cnn=args.lr_cnn, lr_dec=args.lr_dec,
        lr_block=args.lr_block, weight_decay=args.wd,
    )
    optimizer = torch.optim.AdamW(param_groups)
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    
    print(f"\n[3/4] Optimizer ({len(param_groups)} groups):")
    for pg in param_groups:
        n_p = sum(p.numel() for p in pg['params'])
        print(f"  {pg['name']}: {n_p/1e6:.2f}M (lr={pg['lr']:.1e})")
    
    # ================================================================
    # Resume
    # ================================================================
    start_epoch, best_dice = 0, 0.0
    train_hist, val_hist = [], []
    resume_path = os.path.join(ckpt_dir, 'latest.pth')
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        md = model.state_dict()
        loaded = {k: v for k, v in ckpt['model_state_dict'].items() if k in md and v.shape == md[k].shape}
        md.update(loaded)
        model.load_state_dict(md)
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except:
            print("  Warning: optimizer state mismatch, resetting optimizer")
        start_epoch = ckpt['epoch'] + 1
        best_dice = ckpt.get('best_dice', 0.0)
        train_hist = ckpt.get('train_hist', [])
        val_hist = ckpt.get('val_hist', [])
        print(f"  Resumed epoch {start_epoch}, best={best_dice:.4f}")
    
    # ================================================================
    # LR schedule
    # ================================================================
    def get_lr_scale(epoch):
        if epoch < args.warmup:
            return (epoch + 1) / args.warmup
        progress = (epoch - args.warmup) / max(args.epochs - args.warmup, 1)
        return max(0.5 * (1 + math.cos(math.pi * progress)), 1e-2)
    
    # ================================================================
    # Training
    # ================================================================
    print(f"\n[4/4] Training: epoch {start_epoch} → {args.epochs}")
    print(f"  Effective batch: {args.batch_size}×{args.accum_steps}={args.batch_size * args.accum_steps}")
    print("=" * 70)
    
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        scale = get_lr_scale(epoch)
        for i, pg in enumerate(optimizer.param_groups):
            pg['lr'] = base_lrs[i] * scale
        
        model.train()
        losses = []
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            img = batch['image'].to(device)
            lbl = batch['label'].to(device)
            
            with torch.amp.autocast('cuda'):
                logits = model(img)
                loss = dice_loss(logits, lbl) + focal_loss(logits, lbl)
            
            scaler.scale(loss / args.accum_steps).backward()
            
            if (step + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            losses.append(loss.item())
            
            if (step + 1) % 100 == 0:
                print(f"  E{epoch+1} S{step+1}/{len(train_loader)} | loss={loss.item():.4f}")
            
            del logits, loss
            torch.cuda.empty_cache()
        
        avg_loss = float(np.mean(losses))
        cur_lr = optimizer.param_groups[0]['lr']
        train_hist.append({'epoch': epoch + 1, 'loss': avg_loss, 'lr': cur_lr})
        print(f"Epoch {epoch+1:3d}/{args.epochs} | loss={avg_loss:.4f} | "
              f"lr={cur_lr:.2e} | {time.time()-t0:.0f}s")
        
        # Validation
        if (epoch + 1) % args.val_interval == 0:
            tv = time.time()
            model.eval()
            metrics = SegMetrics()
            with torch.no_grad():
                for batch in val_loader:
                    img = batch['image'].to(device)
                    lbl = batch['label'].to(device)
                    with torch.amp.autocast('cuda'):
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
            
            print(f"  Val: ET={vr['ET_dice_mean']:.4f} TC={vr['TC_dice_mean']:.4f} "
                  f"WT={vr['WT_dice_mean']:.4f} mean={vr['mean_dice']:.4f} "
                  f"{'★ BEST' if is_best else ''} | {time.time()-tv:.0f}s")
            
            if is_best:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': {k: v for k, v in model.state_dict().items()
                        if any(x in k for x in ['patch_embed', 'lora', 'decoder',
                               'cnn_branch', 'bottleneck_fusion', 'blocks.'])
                        and ('image_encoder' not in k or 'lora' in k or 'patch_embed' in k
                             or any(f'blocks.{i}' in k for i in range(12-args.unfreeze_blocks, 12)))},
                    'best_dice': best_dice,
                    'config': vars(args),
                    'param_stats': stats,
                }, os.path.join(ckpt_dir, 'best.pth'))
        
        # Save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': {k: v for k, v in model.state_dict().items()
                if any(x in k for x in ['patch_embed', 'lora', 'decoder',
                       'cnn_branch', 'bottleneck_fusion', 'blocks.'])
                and ('image_encoder' not in k or 'lora' in k or 'patch_embed' in k
                     or any(f'blocks.{i}' in k for i in range(12-args.unfreeze_blocks, 12)))},
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dice': best_dice,
            'train_hist': train_hist,
            'val_hist': val_hist,
        }, os.path.join(ckpt_dir, 'latest.pth'))
        
        with open(os.path.join(log_dir, 'training_log.json'), 'w') as f:
            json.dump({
                'config': vars(args), 'param_stats': stats,
                'train': train_hist, 'val': val_hist, 'best_dice': best_dice,
            }, f, indent=2)
    
    # ================================================================
    # Final summary
    # ================================================================
    print(f"\n{'='*70}")
    print(f"Experiment: {args.exp_name}")
    print(f"Config: {stats['config_name']}")
    print(f"Best Mean Dice: {best_dice:.4f}")
    print(f"Trainable params: {stats['trainable']/1e6:.2f}M")
    print(f"{'='*70}")
    
    # Append to unified results file
    results_file = os.path.join(PROJECT, 'logs_ablation', 'all_results.json')
    all_results = {}
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    
    all_results[args.exp_name] = {
        'config': vars(args),
        'config_name': stats['config_name'],
        'trainable_params': stats['trainable'],
        'best_dice': best_dice,
        'best_val': val_hist[-1] if val_hist else {},
    }
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results appended to {results_file}")


if __name__ == '__main__':
    main()
