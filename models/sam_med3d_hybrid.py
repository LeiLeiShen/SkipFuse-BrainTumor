"""
Hybrid SAM-Med3D: Dual-Encoder Architecture for 3D Brain Tumor Segmentation.

Architecture:

  Input (4, 128³)
       │
       ├───────────────────────────┐
       │                           │
  SAM-Med3D ViT Encoder       CNN Branch
  (93.7M frozen + LoRA)       (fully trainable)
       │                           │
  (384, 8³)                   Skips: s1(64, 64³)
       │                           s2(128, 32³)
       │                           s3(256, 16³)
       │                      Bottleneck: (512, 8³)
       │                           │
       └──── Bottleneck Fusion ────┘
             cat(384, 512) → Conv1x1 → (384, 8³)
                      │
               HybridDecoder3D
               with skip connections
                      │
               Output (4, 128³)

Design rationale:
  - ViT path captures global semantic context via pretrained features + LoRA
  - CNN path captures local spatial details lost in ViT's 16× downsampling
  - Skip connections restore multi-scale boundary information to the decoder
  - This addresses the decoder bottleneck identified by LoRA rank ablation:
    LoRA r=4 ≈ r=32 (0.37% diff), but adding CNN skips yields +6.2% improvement

Default config: LoRA r=16, no block unfreezing (b=0), CNN base_ch=64
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything.build_sam3D import sam_model_registry3D
from models.patch_embed import PatchEmbed4Ch
from models.lora import inject_lora_to_encoder
from models.cnn_branch import CNNBranch3D
from models.decoders import HybridDecoder3D, SimpleDecoder3D


class BottleneckFusion(nn.Module):
    """Fuse ViT and CNN features at 8³ resolution via 1×1 convolution."""
    
    def __init__(self, vit_dim=384, cnn_dim=512, out_dim=384):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv3d(vit_dim + cnn_dim, out_dim, 1, bias=False),
            nn.BatchNorm3d(out_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, vit_feat, cnn_feat):
        return self.fuse(torch.cat([vit_feat, cnn_feat], dim=1))


class HybridSAMMed3D(nn.Module):
    """
    Hybrid SAM-Med3D with dual-encoder architecture.
    
    Args:
        sam_checkpoint: path to SAM-Med3D pretrained weights (.pth)
        num_classes: segmentation classes (4 for BraTS: BG/NCR/ED/ET)
        lora_r: LoRA rank for encoder adaptation (default: 16)
        lora_alpha: LoRA scaling factor (default: 32)
        in_channels: input modalities (4 for BraTS MRI)
        unfreeze_blocks: ViT blocks to unfreeze from end (default: 0)
        cnn_base_ch: CNN branch base width (default: 64 → 128 → 256 → 512)
        use_cnn: enable CNN branch (False = SimpleDecoder baseline for ablation)
        use_skip: enable skip connections (False = CNN runs but skips ignored)
        use_fusion: enable bottleneck fusion (False = ViT-only to decoder)
    """
    
    def __init__(self, sam_checkpoint, num_classes=4, lora_r=16, lora_alpha=32,
                 in_channels=4, unfreeze_blocks=0, cnn_base_ch=64,
                 use_cnn=True, use_skip=True, use_fusion=True):
        super().__init__()
        
        self.use_cnn = use_cnn
        self.use_skip = use_skip and use_cnn
        self.use_fusion = use_fusion and use_cnn
        
        # Save config for logging/reproducibility
        self.config = dict(
            lora_r=lora_r, lora_alpha=lora_alpha,
            unfreeze_blocks=unfreeze_blocks, cnn_base_ch=cnn_base_ch,
            use_cnn=use_cnn, use_skip=use_skip, use_fusion=use_fusion,
        )
        
        # ==============================================================
        # Encoder Path 1: SAM-Med3D ViT (pretrained, LoRA-adapted)
        # ==============================================================
        sam = sam_model_registry3D['vit_b_ori'](checkpoint=None)
        if sam_checkpoint and os.path.exists(sam_checkpoint):
            ckpt = torch.load(sam_checkpoint, map_location='cpu')
            sam.load_state_dict(ckpt['model_state_dict'], strict=False)
        
        self.image_encoder = sam.image_encoder
        
        # Replace patch embedding for 4-channel input
        self.image_encoder.patch_embed = PatchEmbed4Ch(in_channels=in_channels)
        
        # Inject LoRA into all qkv projections
        self.image_encoder, self.lora_param_count = inject_lora_to_encoder(
            self.image_encoder, r=lora_r, alpha=lora_alpha
        )
        
        # Freeze all encoder params, then selectively unfreeze
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for name, param in self.image_encoder.named_parameters():
            if 'lora' in name or 'patch_embed' in name:
                param.requires_grad = True
        if unfreeze_blocks > 0:
            n_blocks = len(self.image_encoder.blocks)
            for i in range(n_blocks - unfreeze_blocks, n_blocks):
                for param in self.image_encoder.blocks[i].parameters():
                    param.requires_grad = True
        
        # ==============================================================
        # Encoder Path 2: CNN Branch (conditional)
        # ==============================================================
        if use_cnn:
            self.cnn_branch = CNNBranch3D(in_channels=in_channels, base_ch=cnn_base_ch)
            cnn_out_ch = self.cnn_branch.out_channels  # [64, 128, 256, 512]
        else:
            self.cnn_branch = None
            cnn_out_ch = [cnn_base_ch, cnn_base_ch*2, cnn_base_ch*4, cnn_base_ch*8]
        
        # ==============================================================
        # Bottleneck Fusion (conditional)
        # ==============================================================
        if self.use_fusion:
            self.bottleneck_fusion = BottleneckFusion(
                vit_dim=384, cnn_dim=cnn_out_ch[-1], out_dim=384
            )
        else:
            self.bottleneck_fusion = None
        
        # ==============================================================
        # Decoder
        # ==============================================================
        if self.use_skip:
            self.decoder = HybridDecoder3D(
                in_dim=384, skip_channels=cnn_out_ch[:3], num_classes=num_classes
            )
        else:
            self.decoder = SimpleDecoder3D(in_dim=384, num_classes=num_classes)
    
    def _encode_vit(self, x):
        """ViT encoder forward pass."""
        features = self.image_encoder(x)
        # SAM-Med3D may output (B, 8, 8, 8, 384); ensure channel-first
        if features.shape[1] != 384:
            features = features.permute(0, 4, 1, 2, 3)
        return features
    
    def forward(self, x):
        """
        Args:
            x: (B, 4, 128, 128, 128) — 4-modality MRI volume
        Returns:
            logits: (B, num_classes, 128, 128, 128)
        """
        vit_feat = self._encode_vit(x)  # (B, 384, 8, 8, 8)
        
        if self.use_cnn:
            cnn_skips, cnn_bottleneck = self.cnn_branch(x)
            
            if self.use_fusion:
                bottleneck = self.bottleneck_fusion(vit_feat, cnn_bottleneck)
            else:
                bottleneck = vit_feat
            
            if self.use_skip:
                logits = self.decoder(bottleneck, cnn_skips)
            else:
                logits = self.decoder(bottleneck)
        else:
            logits = self.decoder(vit_feat)
        
        return logits
    
    def get_param_stats(self):
        """Detailed parameter breakdown for logging and paper tables."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        def count(filter_fn):
            return sum(p.numel() for n, p in self.named_parameters()
                      if filter_fn(n) and p.requires_grad)
        
        stats = {
            'total': total,
            'trainable': trainable,
            'trainable_pct': trainable / total * 100,
            'lora': count(lambda n: 'lora' in n),
            'patch_embed': count(lambda n: 'patch_embed' in n and 'image_encoder' in n),
            'unfrozen_vit': count(lambda n: 'blocks.' in n and 'lora' not in n
                                  and 'image_encoder' in n),
            'cnn_branch': count(lambda n: 'cnn_branch' in n),
            'bottleneck_fusion': count(lambda n: 'bottleneck_fusion' in n),
            'decoder': count(lambda n: 'decoder' in n),
            'encoder_frozen': sum(p.numel() for n, p in self.named_parameters()
                                 if 'image_encoder' in n and not p.requires_grad),
        }
        return stats
    
    def get_param_groups(self, lr_lora=2e-4, lr_cnn=5e-4, lr_dec=1e-3,
                          lr_block=5e-5, weight_decay=0.01):
        """
        Parameter groups with differentiated learning rates.
        
        Strategy:
          - LoRA + PatchEmbed: lr_lora (adapting pretrained representations)
          - Unfrozen ViT blocks: lr_block (conservative, preserving pretraining)
          - CNN branch: lr_cnn (training from scratch, moderate)
          - Decoder + Fusion: lr_dec (training from scratch, aggressive)
        """
        groups = {
            'block': [], 'lora': [], 'patch': [],
            'cnn': [], 'fuse': [], 'dec': []
        }
        
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'cnn_branch' in name:
                groups['cnn'].append(p)
            elif 'bottleneck_fusion' in name:
                groups['fuse'].append(p)
            elif 'decoder' in name:
                groups['dec'].append(p)
            elif 'lora' in name:
                groups['lora'].append(p)
            elif 'patch_embed' in name:
                groups['patch'].append(p)
            elif 'blocks.' in name:
                groups['block'].append(p)
        
        param_groups = []
        if groups['block']:
            param_groups.append({
                'params': groups['block'], 'lr': lr_block,
                'weight_decay': weight_decay, 'name': 'unfrozen_vit_blocks'
            })
        if groups['lora'] or groups['patch']:
            param_groups.append({
                'params': groups['lora'] + groups['patch'], 'lr': lr_lora,
                'weight_decay': weight_decay, 'name': 'lora+patch_embed'
            })
        if groups['cnn']:
            param_groups.append({
                'params': groups['cnn'], 'lr': lr_cnn,
                'weight_decay': weight_decay, 'name': 'cnn_branch'
            })
        if groups['fuse'] or groups['dec']:
            param_groups.append({
                'params': groups['fuse'] + groups['dec'], 'lr': lr_dec,
                'weight_decay': weight_decay, 'name': 'fusion+decoder'
            })
        
        return param_groups
