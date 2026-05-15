#!/usr/bin/env python3
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
from models.proto_sam_v3 import ProtoSAM_v3


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
    p.add_argument('--simple_decoder', action='store_true',
                   help='Use SimpleDecoder (no CNN branch) instead of HybridDecoder')
    p.add_argument('--fold', type=int, default=-1,
                   help='5-fold CV fold index (0-4). -1 = use original split.')
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
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

    # Model
    if args.simple_decoder:
        model = ProtoSAM_v3(
            sam_checkpoint=os.path.join(PROJECT, 'SAM-Med3D/ckpt/sam_med3d_turbo.pth'),
            num_classes=4, lora_r=16, lora_alpha=32, in_channels=4,
        ).to(device)
    else:
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

    if args.simple_decoder:
        # ProtoSAM_v3 has no get_param_groups — use 2 groups: encoder (LoRA+PE) vs decoder
        enc_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and ('lora' in n or 'patch_embed' in n)]
        dec_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and 'decoder' in n]
        param_groups = [
            {'params': enc_params, 'lr': 2e-4, 'weight_decay': 0.01, 'name': 'lora+patch_embed'},
            {'params': dec_params, 'lr': 1e-3, 'weight_decay': 0.01, 'name': 'decoder'},
        ]
    else:
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

            if args.simple_decoder:
                out = model(img)
            else:
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
                    logits = model(img) if args.simple_decoder else model(img, deep_supervision=False)
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
                torch.save({'epoch': epoch, 'best_dice': best_dice, 'model_state_dict': model.state_dict(),
                            'config': vars(args)},
                           os.path.join(ckpt_dir, 'best.pth'))

        # Log
        with open(os.path.join(log_dir, 'log.json'), 'w') as f:
            json.dump({'config': vars(args), 'train': train_hist,
                       'val': val_hist, 'best_dice': best_dice}, f, indent=2)

    print(f"\nDone. Best Mean Dice: {best_dice:.4f}")

if __name__ == '__main__':
    main()
