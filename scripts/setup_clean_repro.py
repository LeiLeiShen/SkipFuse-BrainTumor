#!/usr/bin/env python3
"""
One-shot setup for clean c=64+DS reproducibility runs.

What this does:
  1. Restores .bak model files (original, unmodified)
  2. Adds ONLY deep supervision (aux heads + forward flag) — nothing else
  3. Creates a minimal training script (train_repro.py) with zero legacy baggage

Run once, then use train_repro.py for reproducibility experiments.
"""

import os, re

PROJECT = '.'

# ===========================================================================
# Step 1: Restore .bak files
# ===========================================================================
print("=" * 60)
print("Step 1: Restoring .bak model files")
for f in ['models/hybrid_decoder_3d.py', 'models/proto_sam_hybrid.py']:
    bak = os.path.join(PROJECT, f + '.bak')
    dst = os.path.join(PROJECT, f)
    if not os.path.exists(bak):
        print(f"  ERROR: {bak} not found!")
        exit(1)
    with open(bak) as fh:
        content = fh.read()
    with open(dst, 'w') as fh:
        fh.write(content)
    print(f"  Restored {f} from .bak")

# ===========================================================================
# Step 2: Add minimal DS to hybrid_decoder_3d.py
# ===========================================================================
print("\nStep 2: Adding deep supervision to decoder")
dec_path = os.path.join(PROJECT, 'models/hybrid_decoder_3d.py')
with open(dec_path) as f:
    src = f.read()

# 2a. Add aux heads after self.head definition
old_head = """        self.head = nn.Sequential(
            nn.Conv3d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, num_classes, 1),
        )"""
new_head = old_head + """
        
        # Deep supervision auxiliary heads (1x1 conv, ~1800 params total)
        self.aux_head_16 = nn.Conv3d(256, num_classes, 1)
        self.aux_head_32 = nn.Conv3d(128, num_classes, 1)
        self.aux_head_64 = nn.Conv3d(64,  num_classes, 1)"""
src = src.replace(old_head, new_head)

# 2b. Modify forward to support deep_supervision flag
old_fwd = """    def forward(self, x, skips):"""
new_fwd = """    def forward(self, x, skips, deep_supervision=False):"""
src = src.replace(old_fwd, new_fwd)

# 2c. Add DS return path before final return
old_return = """        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        logits = self.head(x)
        
        return logits"""
new_return = """        x = F.interpolate(x64, scale_factor=2, mode='trilinear', align_corners=False)
        logits = self.head(x)
        
        if deep_supervision:
            return [logits,
                    self.aux_head_64(x64),
                    self.aux_head_32(x32),
                    self.aux_head_16(x16)]
        return logits"""
# But first we need to capture intermediate variables x16, x32, x64
# Check original forward body and rename variables
# Original forward uses plain 'x' everywhere, need to save intermediates
old_body = """        s1, s2, s3 = skips
        
        # 8³ → 16³
        x = self.up1(x)
        x = self.fuse1(x, s3)
        
        # 16³ → 32³
        x = self.up2(x)
        x = self.fuse2(x, s2)
        
        # 32³ → 64³
        x = self.up3(x)
        x = self.fuse3(x, s1)
        
        # 64³ → 128³
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        logits = self.head(x)
        
        return logits"""

new_body = """        s1, s2, s3 = skips
        
        # 8³ → 16³
        x = self.up1(x)
        x16 = self.fuse1(x, s3)
        
        # 16³ → 32³
        x = self.up2(x16)
        x32 = self.fuse2(x, s2)
        
        # 32³ → 64³
        x = self.up3(x32)
        x64 = self.fuse3(x, s1)
        
        # 64³ → 128³
        x = F.interpolate(x64, scale_factor=2, mode='trilinear', align_corners=False)
        logits = self.head(x)
        
        if deep_supervision:
            return [logits,
                    self.aux_head_64(x64),
                    self.aux_head_32(x32),
                    self.aux_head_16(x16)]
        return logits"""

src = src.replace(old_body, new_body)
# Remove the old_fwd replacement since we already changed it
# and old_return is now covered by old_body replacement

with open(dec_path, 'w') as f:
    f.write(src)

import ast
ast.parse(src)
print("  Decoder updated and syntax verified")

# ===========================================================================
# Step 3: Add deep_supervision pass-through to ProtoSAM_Hybrid
# ===========================================================================
print("\nStep 3: Adding deep_supervision to ProtoSAM_Hybrid.forward")
model_path = os.path.join(PROJECT, 'models/proto_sam_hybrid.py')
with open(model_path) as f:
    src = f.read()

src = src.replace(
    '    def forward(self, x):',
    '    def forward(self, x, deep_supervision=False):'
)
src = src.replace(
    'logits = self.decoder(fused, cnn_skips)',
    'logits = self.decoder(fused, cnn_skips, deep_supervision=deep_supervision)'
)

with open(model_path, 'w') as f:
    f.write(src)

ast.parse(src)
print("  Model updated and syntax verified")

# ===========================================================================
# Step 4: Verify no contamination
# ===========================================================================
print("\nStep 4: Contamination check")
with open(dec_path) as f:
    dec = f.read()
with open(model_path) as f:
    mod = f.read()

checks = {
    'No skip_scale in decoder': 'skip_scale' not in dec,
    'No vit_skip in decoder': 'vit_skip' not in dec,
    'No aux_et_head in decoder': 'aux_et_head' not in dec,
    'No vit_skip in model': 'vit_skip' not in mod,
    'Has aux_head_16': 'aux_head_16' in dec,
    'Has aux_head_32': 'aux_head_32' in dec,
    'Has aux_head_64': 'aux_head_64' in dec,
    'Has deep_supervision in decoder forward': 'deep_supervision=False' in dec,
    'Has deep_supervision in model forward': 'deep_supervision=False' in mod,
}
all_ok = True
for desc, ok in checks.items():
    status = '✅' if ok else '❌'
    print(f"  {status} {desc}")
    if not ok:
        all_ok = False

if not all_ok:
    print("\n  ERROR: Contamination detected! Check manually.")
    exit(1)

# ===========================================================================
# Step 5: Create clean training script
# ===========================================================================
print("\nStep 5: Creating train_repro.py")
train_script = r'''#!/usr/bin/env python3
"""
Clean reproducibility training script for Hybrid SAM-Med3D.
No legacy code, no unused features, no contamination risk.

Usage:
  python scripts/train_repro.py --exp_name hybrid_c64_ds_rep1 --cnn_base_ch 64 --deep_supervision
  python scripts/train_repro.py --exp_name hybrid_c32_baseline --cnn_base_ch 32
"""

import sys, os
PROJECT = '.'
os.chdir(PROJECT)
sys.path.insert(0, os.path.join(PROJECT, 'SAM-Med3D'))
sys.path.insert(0, PROJECT)

import argparse, time, math, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data_processing.dataset import get_dataloaders
from models.proto_sam_hybrid import ProtoSAM_Hybrid


# === Loss ===
class DiceLoss(nn.Module):
    def __init__(self, n=4, smooth=1.0):
        super().__init__()
        self.n, self.smooth = n, smooth
    def forward(self, logits, target):
        pred = F.softmax(logits, dim=1)
        tgt = F.one_hot(target, self.n).permute(0,4,1,2,3).float()
        losses = []
        for c in range(self.n):
            p, g = pred[:,c], tgt[:,c]
            inter = (p*g).sum(); card = p.sum()+g.sum()
            if card < self.smooth: continue
            losses.append(1.0 - (2*inter+self.smooth)/(card+self.smooth))
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=logits.device, requires_grad=True)

class FocalLoss(nn.Module):
    def __init__(self, n=4, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        freq = torch.tensor([98.93, 0.29, 0.31, 0.79])
        alpha = 1.0/freq; alpha = alpha/alpha.sum()*n
        self.register_buffer('alpha', alpha.float())
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none')
        pt = F.softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
        return (self.alpha[target] * (1-pt)**self.gamma * ce).mean()


# === Metrics ===
class SegMetrics:
    def __init__(self):
        self.dice = {'ET':[], 'TC':[], 'WT':[]}
    @staticmethod
    def _d(p, g):
        if p.sum()==0 and g.sum()==0: return 1.0
        if p.sum()==0 or g.sum()==0: return 0.0
        return float(2.0*(p&g).sum()/(p.sum()+g.sum()))
    def update(self, logits, target):
        pred = logits.argmax(1).cpu().numpy()
        gt = target.cpu().numpy()
        for b in range(pred.shape[0]):
            self.dice['ET'].append(self._d(pred[b]==3, gt[b]==3))
            self.dice['TC'].append(self._d((pred[b]==1)|(pred[b]==3), (gt[b]==1)|(gt[b]==3)))
            self.dice['WT'].append(self._d(pred[b]>=1, gt[b]>=1))
    def compute(self):
        r = {f'{k}_dice': float(np.mean(v)) for k,v in self.dice.items()}
        r['mean_dice'] = float(np.mean([r['ET_dice'],r['TC_dice'],r['WT_dice']]))
        return r


# === Main ===
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_name', type=str, required=True)
    p.add_argument('--cnn_base_ch', type=int, default=32)
    p.add_argument('--deep_supervision', action='store_true')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--accum', type=int, default=2)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--val_interval', type=int, default=5)
    args = p.parse_args()

    device = torch.device('cuda')
    ckpt_dir = os.path.join(PROJECT, 'checkpoints', args.exp_name)
    log_dir = os.path.join(PROJECT, 'logs', args.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print(f"Clean Repro | {args.exp_name}")
    print(f"  CNN={args.cnn_base_ch}  DS={args.deep_supervision}")
    print("=" * 60)

    # Data
    train_loader, val_loader, _, _ = get_dataloaders(
        PROJECT, batch_size=args.batch_size, num_workers=4, strong_augment=False)
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

    # Model
    model = ProtoSAM_Hybrid(
        sam_checkpoint=os.path.join(PROJECT, 'SAM-Med3D/ckpt/sam_med3d_turbo.pth'),
        num_classes=4, lora_r=16, lora_alpha=32,
        in_channels=4, unfreeze_blocks=0, cnn_base_ch=args.cnn_base_ch,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total/1e6:.2f}M  Trainable: {trainable/1e6:.2f}M ({trainable/total*100:.1f}%)")

    # Loss & Optimizer
    dice_fn = DiceLoss(4)
    focal_fn = FocalLoss(4).to(device)

    param_groups = model.get_param_groups(
        lr_enc=2e-4, lr_cnn=5e-4, lr_dec=1e-3, lr_block=5e-5, weight_decay=0.01)
    optimizer = torch.optim.AdamW(param_groups)
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    for pg in param_groups:
        n = sum(pp.numel() for pp in pg['params'])
        print(f"  {pg['name']}: {n/1e6:.2f}M lr={pg['lr']:.1e}")

    # LR schedule
    def lr_scale(epoch):
        if epoch < args.warmup:
            return (epoch+1)/args.warmup
        t = (epoch-args.warmup)/max(args.epochs-args.warmup, 1)
        return max(0.5*(1+math.cos(math.pi*t)), 1e-2)

    # DS weights
    ds_weights = [1.0, 0.5, 0.25, 0.125]
    ds_weights = [w/sum(ds_weights) for w in ds_weights]

    # Train
    best_dice = 0.0
    train_hist, val_hist = [], []
    scaler = torch.amp.GradScaler('cuda', enabled=False)  # no AMP

    for epoch in range(args.epochs):
        t0 = time.time()
        s = lr_scale(epoch)
        for i, pg in enumerate(optimizer.param_groups):
            pg['lr'] = base_lrs[i] * s

        model.train()
        losses = []
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            img = batch['image'].to(device)
            lbl = batch['label'].to(device)

            out = model(img, deep_supervision=args.deep_supervision)

            if args.deep_supervision and isinstance(out, list):
                loss = 0.0
                for w, logits_i in zip(ds_weights, out):
                    if logits_i.shape[-1] != lbl.shape[-1]:
                        lbl_ds = F.interpolate(
                            lbl.unsqueeze(1).float(),
                            size=logits_i.shape[-3:], mode='nearest'
                        ).squeeze(1).long()
                    else:
                        lbl_ds = lbl
                    loss = loss + w * (dice_fn(logits_i, lbl_ds) + focal_fn(logits_i, lbl_ds))
            else:
                loss = dice_fn(out, lbl) + focal_fn(out, lbl)

            (loss / args.accum).backward()

            if (step+1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()

            losses.append(loss.item())
            if (step+1) % 100 == 0:
                print(f"  E{epoch+1} S{step+1}/{len(train_loader)} loss={loss.item():.4f}")
            del out, loss; torch.cuda.empty_cache()

        avg_loss = float(np.mean(losses))
        train_hist.append({'epoch': epoch+1, 'loss': avg_loss})
        print(f"Epoch {epoch+1:3d}/{args.epochs} | loss={avg_loss:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | {time.time()-t0:.0f}s")

        # Validation
        if (epoch+1) % args.val_interval == 0:
            model.eval()
            metrics = SegMetrics()
            with torch.no_grad():
                for batch in val_loader:
                    img = batch['image'].to(device)
                    lbl = batch['label'].to(device)
                    logits = model(img, deep_supervision=False)
                    metrics.update(logits, lbl)
                    del logits; torch.cuda.empty_cache()

            vr = metrics.compute()
            vr['epoch'] = epoch+1
            val_hist.append(vr)
            is_best = vr['mean_dice'] > best_dice
            if is_best: best_dice = vr['mean_dice']
            print(f"  Val: ET={vr['ET_dice']:.4f} TC={vr['TC_dice']:.4f} "
                  f"WT={vr['WT_dice']:.4f} Mean={vr['mean_dice']:.4f} "
                  f"{'* BEST' if is_best else ''}")
            if is_best:
                torch.save({'epoch': epoch, 'best_dice': best_dice,
                            'config': vars(args)},
                           os.path.join(ckpt_dir, 'best.pth'))

        # Log
        with open(os.path.join(log_dir, 'log.json'), 'w') as f:
            json.dump({'config': vars(args), 'train': train_hist,
                       'val': val_hist, 'best_dice': best_dice}, f, indent=2)

    print(f"\nDone. Best Mean Dice: {best_dice:.4f}")

if __name__ == '__main__':
    main()
'''

train_path = os.path.join(PROJECT, 'scripts/train_repro.py')
with open(train_path, 'w') as f:
    f.write(train_script)
ast.parse(train_script)
print(f"  Created {train_path}")

print("\n" + "=" * 60)
print("SETUP COMPLETE")
print("=" * 60)
print("""
To run reproducibility experiments:

  cd .

  # c=64 + DS (2 runs to supplement original 87.81%)
  mkdir -p logs/hybrid_c64_ds_rep1 logs/hybrid_c64_ds_rep2
  nohup bash -c '
  python scripts/train_repro.py --exp_name hybrid_c64_ds_rep1 --cnn_base_ch 64 --deep_supervision \\
      > logs/hybrid_c64_ds_rep1/console.log 2>&1 && \\
  python scripts/train_repro.py --exp_name hybrid_c64_ds_rep2 --cnn_base_ch 64 --deep_supervision \\
      > logs/hybrid_c64_ds_rep2/console.log 2>&1
  ' > logs/repro_orchestrator.log 2>&1 &
  echo "PID: $!"
""")
