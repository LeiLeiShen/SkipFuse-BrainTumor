#!/usr/bin/env python3
"""
ProtoSAM-Med3D Hybrid: SAM-Med3D ViT + CNN Branch + Skip-Connected Decoder
Training script for BraTS2021 GLI.

Usage:
  cd .
  mkdir -p logs_hybrid checkpoints_hybrid
  nohup python scripts/train_hybrid.py > logs_hybrid/train.log 2>&1 &

Architecture:
  Path 1: SAM-Med3D ViT encoder (frozen + LoRA) → (384, 8³)
  Path 2: Lightweight CNN branch → skips at 64³, 32³, 16³ + bottleneck at 8³
  Fusion: ViT + CNN bottleneck → fused (384, 8³)
  Decoder: U-Net style with skip connections from CNN → (4, 128³)
"""

import sys, os
PROJECT = '.'
os.chdir(PROJECT)
sys.path.insert(0, os.path.join(PROJECT, 'SAM-Med3D'))
sys.path.insert(0, PROJECT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json, time, math

from data_processing.dataset import get_dataloaders
from models.proto_sam_hybrid import ProtoSAM_Hybrid


# ============================================================================
# Loss functions (same as v3.1)
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
        # BraTS class frequency: ~99% BG, 0.29% NCR, 0.31% ED, 0.79% ET
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
# Metrics (same as v3.1)
# ============================================================================

class SegMetrics:
    def __init__(self):
        self.dice = {'ET': [], 'TC': [], 'WT': []}
    
    @staticmethod
    def dc(p, g):
        if p.sum() == 0 and g.sum() == 0:
            return 1.0
        if p.sum() == 0 or g.sum() == 0:
            return 0.0
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
# Training
# ============================================================================

def main():
    print("=" * 70)
    print("ProtoSAM-Med3D Hybrid: ViT + CNN + Skip-Connected Decoder")
    print("=" * 70)
    
    device = torch.device('cuda')
    ckpt_dir = os.path.join(PROJECT, 'checkpoints_hybrid')
    log_dir = os.path.join(PROJECT, 'logs_hybrid')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # ================================================================
    # Hyperparameters
    # ================================================================
    MAX_EPOCHS = 100
    BATCH = 2          # 128³ volumes
    ACCUM = 2          # Effective batch = 4
    WARMUP = 5
    WD = 0.01
    VAL_INT = 5        # Validate every N epochs
    
    # Differentiated learning rates (4 groups)
    BLOCK_LR = 5e-5    # Unfrozen ViT blocks: most conservative
    LORA_LR = 2e-4     # LoRA + PatchEmbed: moderate
    CNN_LR = 5e-4      # CNN branch: moderate-high (training from scratch)
    DEC_LR = 1e-3      # Fusion + Decoder: highest (training from scratch)
    
    # Architecture config
    LORA_R = 16         # LoRA rank (16 found optimal in ablation)
    LORA_ALPHA = 32
    UNFREEZE_BLOCKS = 2 # Unfreeze last 2 ViT blocks (best from ablation)
    CNN_BASE_CH = 32    # CNN branch: 32 → 64 → 128 → 256
    
    # ================================================================
    # Data
    # ================================================================
    print("\n[1/4] Loading data...")
    train_loader, val_loader, _, _ = get_dataloaders(
        PROJECT, batch_size=BATCH, num_workers=4, strong_augment=False
    )
    print(f"  Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    
    # ================================================================
    # Model
    # ================================================================
    print("\n[2/4] Building hybrid model...")
    model = ProtoSAM_Hybrid(
        sam_checkpoint=os.path.join(PROJECT, 'SAM-Med3D/ckpt/sam_med3d_turbo.pth'),
        num_classes=4,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        in_channels=4,
        unfreeze_blocks=UNFREEZE_BLOCKS,
        cnn_base_ch=CNN_BASE_CH,
    ).to(device)
    
    stats = model.get_param_stats()
    print(f"  Total params:     {stats['total']/1e6:.2f}M")
    print(f"  Trainable params: {stats['trainable']/1e6:.2f}M ({stats['trainable_ratio']:.1f}%)")
    print(f"  Breakdown:")
    print(f"    LoRA:             {stats['lora']/1e6:.2f}M")
    print(f"    PatchEmbed:       {stats['patch_embed']/1e6:.2f}M")
    print(f"    Unfrozen blocks:  {stats['unfrozen_blocks']/1e6:.2f}M")
    print(f"    CNN branch:       {stats['cnn_branch']/1e6:.2f}M")
    print(f"    Bottleneck fuse:  {stats['bottleneck_fusion']/1e6:.2f}M")
    print(f"    Decoder:          {stats['decoder']/1e6:.2f}M")
    print(f"    Encoder frozen:   {stats['encoder_frozen']/1e6:.2f}M")
    
    # ================================================================
    # Loss, Optimizer, Scheduler
    # ================================================================
    dice_loss = DiceLoss(4)
    focal_loss = FocalLoss(4).to(device)
    
    # 4-group optimizer
    param_groups = model.get_param_groups(
        lr_enc=LORA_LR, lr_cnn=CNN_LR, lr_dec=DEC_LR,
        lr_block=BLOCK_LR, weight_decay=WD,
    )
    
    optimizer = torch.optim.AdamW(param_groups)
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    
    print(f"\n[3/4] Optimizer ({len(param_groups)} groups):")
    for pg in param_groups:
        n_params = sum(p.numel() for p in pg['params'])
        print(f"  {pg['name']}: {n_params/1e6:.2f}M (lr={pg['lr']:.1e})")
    
    # ================================================================
    # Resume from checkpoint
    # ================================================================
    start_epoch, best_dice = 0, 0.0
    train_hist, val_hist = [], []
    resume_path = os.path.join(ckpt_dir, 'latest.pth')
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        md = model.state_dict()
        loaded = {k: v for k, v in ckpt['model_state_dict'].items() if k in md}
        md.update(loaded)
        model.load_state_dict(md)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_dice = ckpt.get('best_dice', 0.0)
        train_hist = ckpt.get('train_hist', [])
        val_hist = ckpt.get('val_hist', [])
        print(f"  Resumed from epoch {start_epoch}, best={best_dice:.4f}")
    
    # ================================================================
    # LR schedule: warmup + cosine decay
    # ================================================================
    def get_lr_scale(epoch):
        if epoch < WARMUP:
            return (epoch + 1) / WARMUP
        progress = (epoch - WARMUP) / max(MAX_EPOCHS - WARMUP, 1)
        return max(0.5 * (1 + math.cos(math.pi * progress)), 1e-6 / BLOCK_LR)
    
    # ================================================================
    # Training loop
    # ================================================================
    print(f"\n[4/4] Training: epoch {start_epoch} → {MAX_EPOCHS}")
    print(f"  Effective batch: {BATCH}×{ACCUM}={BATCH * ACCUM}")
    print("=" * 70)
    
    scaler = torch.amp.GradScaler('cuda')  # Mixed precision
    
    for epoch in range(start_epoch, MAX_EPOCHS):
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
            
            # Mixed precision forward
            with torch.amp.autocast('cuda'):
                logits = model(img)
                loss = dice_loss(logits, lbl) + focal_loss(logits, lbl)
            
            scaler.scale(loss / ACCUM).backward()
            
            if (step + 1) % ACCUM == 0:
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
        lr0 = optimizer.param_groups[0]['lr']
        train_hist.append({'epoch': epoch + 1, 'loss': avg_loss, 'lr': lr0})
        print(f"Epoch {epoch+1:3d}/{MAX_EPOCHS} | loss={avg_loss:.4f} | "
              f"lr={lr0:.2e} | {time.time()-t0:.0f}s")
        
        # ============================================================
        # Validation
        # ============================================================
        if (epoch + 1) % VAL_INT == 0:
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
            
            # Save best model (only trainable params)
            if is_best:
                save_keys = ['patch_embed', 'lora', 'decoder', 'cnn_branch', 
                            'bottleneck_fusion', 'blocks.10', 'blocks.11']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': {k: v for k, v in model.state_dict().items()
                        if any(x in k for x in save_keys)},
                    'best_dice': best_dice,
                    'config': {
                        'lora_r': LORA_R, 'lora_alpha': LORA_ALPHA,
                        'unfreeze_blocks': UNFREEZE_BLOCKS, 'cnn_base_ch': CNN_BASE_CH,
                    },
                }, os.path.join(ckpt_dir, 'best.pth'))
        
        # Save latest checkpoint
        save_keys = ['patch_embed', 'lora', 'decoder', 'cnn_branch',
                    'bottleneck_fusion', 'blocks.10', 'blocks.11']
        state = {
            'epoch': epoch,
            'model_state_dict': {k: v for k, v in model.state_dict().items()
                if any(x in k for x in save_keys)},
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dice': best_dice,
            'train_hist': train_hist,
            'val_hist': val_hist,
        }
        torch.save(state, os.path.join(ckpt_dir, 'latest.pth'))
        
        # Save JSON log
        with open(os.path.join(log_dir, 'training_log.json'), 'w') as f:
            json.dump({
                'train': train_hist, 'val': val_hist, 
                'best_dice': best_dice,
                'config': {
                    'lora_r': LORA_R, 'lora_alpha': LORA_ALPHA,
                    'unfreeze_blocks': UNFREEZE_BLOCKS, 'cnn_base_ch': CNN_BASE_CH,
                    'batch': BATCH, 'accum': ACCUM,
                    'lr': {'block': BLOCK_LR, 'lora': LORA_LR, 'cnn': CNN_LR, 'dec': DEC_LR},
                },
            }, f, indent=2)
    
    print(f"\nHybrid training complete! Best Mean Dice: {best_dice:.4f}")


if __name__ == '__main__':
    main()
