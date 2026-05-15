"""
CNN Branch for 3D Hybrid SAM-Med3D
Lightweight 4-stage 3D CNN providing multi-scale skip features.

Architecture:
  Input (4, 128³) → Stage1 (32, 64³) → Stage2 (64, 32³) → Stage3 (128, 16³) → Stage4 (256, 8³)

Each stage: Conv3d(stride=2) + BN + ReLU + Conv3d + BN + ReLU (ResBlock-style)
Total params: ~3.5M
"""

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Basic double-conv block with optional residual connection."""
    
    def __init__(self, in_ch, out_ch, stride=1, residual=True):
        super().__init__()
        self.residual = residual and (in_ch == out_ch) and (stride == 1)
        
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        
        if self.residual:
            self.skip = nn.Identity()
        elif stride != 1 or in_ch != out_ch:
            # Projection shortcut for dimension mismatch
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch)
            )
            self.residual = True  # Enable residual with projection
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + self.skip(identity)
        out = self.relu(out)
        return out


class CNNBranch3D(nn.Module):
    """
    Lightweight 3D CNN branch for hybrid architecture.
    
    Produces multi-scale features matching SAM-Med3D's 16× downsampling:
      128³ → 64³(32ch) → 32³(64ch) → 16³(128ch) → 8³(256ch)
    
    Returns:
        skips: list of [s1(32,64³), s2(64,32³), s3(128,16³)]
        bottleneck: (256, 8³) for fusion with ViT features
    """
    
    def __init__(self, in_channels=4, base_ch=32):
        super().__init__()
        ch = base_ch  # 32
        
        # Stage 1: 128³ → 64³, 4 → 32 channels
        self.stage1 = ConvBlock3D(in_channels, ch, stride=2)
        
        # Stage 2: 64³ → 32³, 32 → 64 channels
        self.stage2 = ConvBlock3D(ch, ch * 2, stride=2)
        
        # Stage 3: 32³ → 16³, 64 → 128 channels
        self.stage3 = ConvBlock3D(ch * 2, ch * 4, stride=2)
        
        # Stage 4: 16³ → 8³, 128 → 256 channels
        self.stage4 = ConvBlock3D(ch * 4, ch * 8, stride=2)
        
        self.out_channels = [ch, ch * 2, ch * 4, ch * 8]  # [32, 64, 128, 256]
    
    def forward(self, x):
        """
        Args:
            x: (B, 4, 128, 128, 128)
        Returns:
            skips: [s1(B,32,64³), s2(B,64,32³), s3(B,128,16³)]
            bottleneck: (B, 256, 8, 8, 8)
        """
        s1 = self.stage1(x)   # (B, 32, 64, 64, 64)
        s2 = self.stage2(s1)  # (B, 64, 32, 32, 32)
        s3 = self.stage3(s2)  # (B, 128, 16, 16, 16)
        s4 = self.stage4(s3)  # (B, 256, 8, 8, 8)
        
        return [s1, s2, s3], s4
