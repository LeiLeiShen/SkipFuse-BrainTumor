"""
ProtoSAM-Med3D Hybrid: SAM-Med3D ViT Encoder + Lightweight CNN Branch + Skip-Connected Decoder

Architecture Overview:
                                    
  Input (4, 128³)                   
       │                            
       ├──────────────────────┐     
       │                      │     
  SAM-Med3D ViT Encoder    CNN Branch (3.5M)
  (93.7M, LoRA adapted)      │     
       │                   Skips: s1(32, 64³)
       │                          s2(64, 32³)
  (384, 8³)                       s3(128, 16³)
       │                      │    
       ├──── Bottleneck ──────┘    
       │     Fusion          (256, 8³)
       │     (384+256 → 384)       
       │                           
  HybridDecoder3D                  
  with skip connections            
       │                           
  Output (4, 128³)                 

Compared to v3.1 (SimpleDecoder, no skips):
  - Adds CNN branch: ~3.5M params
  - Replaces SimpleDecoder with HybridDecoder: ~4.8M → ~5.5M
  - Adds bottleneck fusion: ~0.25M
  - Total new cost: ~5M params
  - Expected benefit: multi-scale spatial details preserved via skip connections
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything.build_sam3D import sam_model_registry3D
from models.patch_embed import PatchEmbed4Ch
from models.lora3d import inject_lora_to_encoder
from models.cnn_branch_3d import CNNBranch3D
from models.hybrid_decoder_3d import HybridDecoder3D


class BottleneckFusion3D(nn.Module):
    """Fuse ViT encoder output (384d) with CNN bottleneck (256d) at 8³ resolution."""
    
    def __init__(self, vit_dim=384, cnn_dim=256, out_dim=384):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv3d(vit_dim + cnn_dim, out_dim, 1, bias=False),
            nn.BatchNorm3d(out_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, vit_feat, cnn_feat):
        """
        Args:
            vit_feat: (B, 384, 8, 8, 8) from SAM-Med3D encoder
            cnn_feat: (B, 256, 8, 8, 8) from CNN branch stage4
        Returns:
            (B, 384, 8, 8, 8) fused features
        """
        return self.fuse(torch.cat([vit_feat, cnn_feat], dim=1))


class ProtoSAM_Hybrid(nn.Module):
    """
    Hybrid SAM-Med3D: ViT encoder + CNN branch + skip-connected decoder.
    
    Design philosophy:
      - ViT path: global semantic features via SAM-Med3D pretrained encoder + LoRA
      - CNN path: local spatial details via lightweight 3D CNN
      - Decoder: U-Net style with skip connections from CNN branch
    
    This is a direct 3D extension of the 2D Swin-SAM-UNet architecture from paper 1,
    adapted for parameter efficiency (no Swin/SE/ConvNeXt blocks to stay lightweight).
    
    Args:
        sam_checkpoint: path to SAM-Med3D pretrained weights
        num_classes: number of segmentation classes (4 for BraTS)
        lora_r: LoRA rank for encoder adaptation
        lora_alpha: LoRA scaling factor
        in_channels: number of input modalities (4 for BraTS MRI)
        unfreeze_blocks: number of ViT blocks to unfreeze from the end (0-12)
        cnn_base_ch: base channel width of CNN branch (32 → 64 → 128 → 256)
    """
    
    def __init__(self, sam_checkpoint, num_classes=4, lora_r=16, lora_alpha=32,
                 in_channels=4, unfreeze_blocks=0, cnn_base_ch=32):
        super().__init__()
        self.unfreeze_blocks = unfreeze_blocks
        
        # ================================================================
        # Path 1: SAM-Med3D ViT Encoder (pretrained, LoRA adapted)
        # ================================================================
        sam = sam_model_registry3D['vit_b_ori'](checkpoint=None)
        if sam_checkpoint and os.path.exists(sam_checkpoint):
            ckpt = torch.load(sam_checkpoint, map_location='cpu')
            sam.load_state_dict(ckpt['model_state_dict'], strict=False)
        
        self.image_encoder = sam.image_encoder
        
        # Replace patch embedding for 4-channel MRI input
        self.image_encoder.patch_embed = PatchEmbed4Ch(
            original_patch_embed=sam.image_encoder.patch_embed,
            in_channels=in_channels,
        )
        
        # Inject LoRA adapters
        self.image_encoder, self.lora_param_count = inject_lora_to_encoder(
            self.image_encoder, r=lora_r, alpha=lora_alpha
        )
        
        # Freeze encoder, then selectively unfreeze
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        # Always trainable: LoRA params + patch embedding
        for name, param in self.image_encoder.named_parameters():
            if 'lora' in name or 'patch_embed' in name:
                param.requires_grad = True
        
        # Optionally unfreeze last N blocks
        if unfreeze_blocks > 0:
            total_blocks = len(self.image_encoder.blocks)
            unfreeze_start = total_blocks - unfreeze_blocks
            for i in range(unfreeze_start, total_blocks):
                for param in self.image_encoder.blocks[i].parameters():
                    param.requires_grad = True
        
        # ================================================================
        # Path 2: Lightweight CNN Branch (fully trainable)
        # ================================================================
        self.cnn_branch = CNNBranch3D(in_channels=in_channels, base_ch=cnn_base_ch)
        
        # ================================================================
        # Bottleneck Fusion: merge ViT (384d) + CNN (256d) at 8³
        # ================================================================
        self.bottleneck_fusion = BottleneckFusion3D(
            vit_dim=384,
            cnn_dim=self.cnn_branch.out_channels[-1],  # 256
            out_dim=384,
        )
        
        # ================================================================
        # Hybrid Decoder with skip connections from CNN branch
        # ================================================================
        self.decoder = HybridDecoder3D(
            in_dim=384,
            skip_channels=self.cnn_branch.out_channels[:3],  # [32, 64, 128]
            num_classes=num_classes,
        )
    
    def encode(self, x):
        """Run ViT encoder path only."""
        features = self.image_encoder(x)
        if features.shape[1] != 384:
            features = features.permute(0, 4, 1, 2, 3)
        return features
    
    def forward(self, x, deep_supervision=False):
        """
        Args:
            x: (B, 4, 128, 128, 128) — 4-modality MRI volume
        Returns:
            logits: (B, num_classes, 128, 128, 128)
        """
        # Path 1: ViT encoder → (B, 384, 8, 8, 8)
        vit_feat = self.encode(x)
        
        # Path 2: CNN branch → skips + bottleneck
        cnn_skips, cnn_bottleneck = self.cnn_branch(x)
        
        # Fuse at bottleneck
        fused = self.bottleneck_fusion(vit_feat, cnn_bottleneck)
        
        # Decode with skip connections
        logits = self.decoder(fused, cnn_skips, deep_supervision=deep_supervision)
        
        return logits
    
    def get_param_stats(self):
        """Detailed parameter statistics for logging."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        stats = {
            'total': total,
            'trainable': trainable,
            'trainable_ratio': trainable / total * 100,
        }
        
        # Break down by component
        components = {
            'lora': lambda n, p: 'lora' in n,
            'patch_embed': lambda n, p: 'patch_embed' in n and 'image_encoder' in n,
            'unfrozen_blocks': lambda n, p: ('blocks.' in n and 'lora' not in n 
                                              and 'image_encoder' in n and p.requires_grad),
            'cnn_branch': lambda n, p: 'cnn_branch' in n,
            'bottleneck_fusion': lambda n, p: 'bottleneck_fusion' in n,
            'decoder': lambda n, p: 'decoder' in n,
        }
        
        for comp_name, filter_fn in components.items():
            params = sum(p.numel() for n, p in self.named_parameters() if filter_fn(n, p))
            stats[comp_name] = params
        
        # Encoder frozen (for reference)
        encoder_frozen = sum(p.numel() for n, p in self.named_parameters() 
                           if 'image_encoder' in n and not p.requires_grad)
        stats['encoder_frozen'] = encoder_frozen
        
        return stats
    
    def get_param_groups(self, lr_enc=1e-4, lr_cnn=5e-4, lr_dec=1e-3, 
                          lr_block=5e-5, weight_decay=0.01):
        """
        Get parameter groups with differentiated learning rates.
        
        Strategy:
          - LoRA + PatchEmbed: lr_enc (moderate, adapting pretrained)
          - Unfrozen ViT blocks: lr_block (conservative, preserving pretrained)
          - CNN branch: lr_cnn (moderate, training from scratch but small)
          - Bottleneck + Decoder: lr_dec (aggressive, training from scratch)
        """
        block_params, lora_params, patch_params = [], [], []
        cnn_params, fusion_params, dec_params = [], [], []
        
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            
            if 'cnn_branch' in name:
                cnn_params.append(p)
            elif 'bottleneck_fusion' in name:
                fusion_params.append(p)
            elif 'decoder' in name:
                dec_params.append(p)
            elif 'lora' in name:
                lora_params.append(p)
            elif 'patch_embed' in name:
                patch_params.append(p)
            elif 'blocks.' in name:
                block_params.append(p)
        
        groups = []
        if block_params:
            groups.append({'params': block_params, 'lr': lr_block, 'weight_decay': weight_decay,
                          'name': 'unfrozen_blocks'})
        groups.extend([
            {'params': lora_params + patch_params, 'lr': lr_enc, 'weight_decay': weight_decay,
             'name': 'lora+patch_embed'},
            {'params': cnn_params, 'lr': lr_cnn, 'weight_decay': weight_decay,
             'name': 'cnn_branch'},
            {'params': fusion_params + dec_params, 'lr': lr_dec, 'weight_decay': weight_decay,
             'name': 'fusion+decoder'},
        ])
        
        return groups
