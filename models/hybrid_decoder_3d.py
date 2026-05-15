"""
Hybrid Decoder for 3D SAM-Med3D
U-Net style decoder with skip connections from CNN branch.

Architecture:
  Fused bottleneck (384, 8³) 
    → up1 + skip3(128) → (256, 16³)
    → up2 + skip2(64)  → (128, 32³)
    → up3 + skip1(32)  → (64, 64³)
    → interp 2× → (64, 128³) → head → (num_classes, 128³)

vs SimpleDecoder3D (no skips):
  (384, 8³) → (256, 16³) → (128, 32³) → (64, 64³) → interp → head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipFusion3D(nn.Module):
    """Fuse upsampled features with skip connection via concatenation + conv."""
    
    def __init__(self, up_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(up_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x_up, x_skip):
        x = torch.cat([x_up, x_skip], dim=1)
        return self.conv(x)


class HybridDecoder3D(nn.Module):
    """
    U-Net style decoder with skip connections from CNN branch.
    
    Args:
        in_dim: bottleneck channel dim (384 from ViT)
        skip_channels: list of CNN skip feature channels [32, 64, 128]
        num_classes: number of output classes
    """
    
    def __init__(self, in_dim=384, skip_channels=[32, 64, 128], num_classes=4):
        super().__init__()
        
        # Up1: 8³ → 16³, fuse with skip3 (128ch)
        self.up1 = nn.ConvTranspose3d(in_dim, 256, kernel_size=2, stride=2, bias=False)
        self.fuse1 = SkipFusion3D(256, skip_channels[2], 256)  # 256+128=384 → 256
        
        # Up2: 16³ → 32³, fuse with skip2 (64ch)
        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2, bias=False)
        self.fuse2 = SkipFusion3D(128, skip_channels[1], 128)  # 128+64=192 → 128
        
        # Up3: 32³ → 64³, fuse with skip1 (32ch)
        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, bias=False)
        self.fuse3 = SkipFusion3D(64, skip_channels[0], 64)    # 64+32=96 → 64
        
        # Final: 64³ → 128³ (trilinear) + classification head
        self.head = nn.Sequential(
            nn.Conv3d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, num_classes, 1),
        )
        
        # Deep supervision auxiliary heads (1x1 conv, ~1800 params total)
        self.aux_head_16 = nn.Conv3d(256, num_classes, 1)
        self.aux_head_32 = nn.Conv3d(128, num_classes, 1)
        self.aux_head_64 = nn.Conv3d(64,  num_classes, 1)
    
    def forward(self, x, skips, deep_supervision=False):
        """
        Args:
            x: bottleneck features (B, 384, 8, 8, 8) - fused ViT + CNN
            skips: [s1(B,32,64³), s2(B,64,32³), s3(B,128,16³)] from CNN branch
        Returns:
            logits: (B, num_classes, 128, 128, 128)
        """
        s1, s2, s3 = skips
        
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
        return logits
