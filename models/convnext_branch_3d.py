"""
Enhanced CNN Branch for 3D Hybrid SAM-Med3D
ConvNeXt-style blocks + SE channel attention.

Compared to basic CNNBranch3D (Conv-BN-ReLU):
  - Depthwise separable convolutions (larger receptive field, fewer params)
  - Inverted bottleneck (channel expansion ratio 4×)
  - LayerNorm + GELU (modern normalization and activation)
  - SE channel attention (adaptive feature recalibration)

Architecture:
  Input (4, 128³) → Stage1 (c, 64³) → Stage2 (2c, 32³) → Stage3 (4c, 16³) → Stage4 (8c, 8³)
  Default c=32: ~2.8M params (vs ~3.6M for basic branch — actually fewer due to depthwise)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm3d(nn.Module):
    """Channel-first LayerNorm for 3D feature maps. (B, C, D, H, W) format."""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, D, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[None, :, None, None, None] * x + self.bias[None, :, None, None, None]
        return x


class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block for 3D feature maps."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1, 1)
        return x * w


class ConvNeXtBlock3D(nn.Module):
    """
    ConvNeXt-style block adapted for 3D.

    Structure:
      Depthwise Conv3d(k=7) → LayerNorm → Pointwise Conv(1×, expand=4×) → GELU
      → Pointwise Conv(4×, compress=1×) → + residual

    Key differences from basic ConvBlock3D:
      - Depthwise separable: k=7 receptive field with fewer params than k=3 standard
      - Inverted bottleneck: expand channels 4× internally for richer feature mixing
      - LayerNorm + GELU: more stable training, closer to Transformer behavior
    """

    def __init__(self, dim, kernel_size=7, expand_ratio=4, drop_path=0.0):
        super().__init__()
        padding = kernel_size // 2
        mid_dim = int(dim * expand_ratio)

        # Depthwise spatial convolution (large kernel, groups=dim)
        self.dwconv = nn.Conv3d(dim, dim, kernel_size, padding=padding, groups=dim, bias=True)
        self.norm = LayerNorm3d(dim)
        # Pointwise expansion
        self.pwconv1 = nn.Conv3d(dim, mid_dim, 1, bias=True)
        self.act = nn.GELU()
        # Pointwise compression
        self.pwconv2 = nn.Conv3d(mid_dim, dim, 1, bias=True)

        # Stochastic depth (optional, for deeper networks)
        self.drop_path = drop_path

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # Drop path during training (simplified version)
        if self.drop_path > 0 and self.training:
            keep = torch.rand(x.shape[0], 1, 1, 1, 1, device=x.device) > self.drop_path
            x = x * keep / (1 - self.drop_path)
        x = residual + x
        return x


class DownsampleLayer3D(nn.Module):
    """Strided convolution for spatial downsampling + channel expansion."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm = LayerNorm3d(in_ch)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2, bias=True)

    def forward(self, x):
        return self.conv(self.norm(x))


class ConvNeXtStage3D(nn.Module):
    """One stage: downsample → N × ConvNeXtBlock → SE attention."""

    def __init__(self, in_ch, out_ch, num_blocks=1, use_se=True, drop_path=0.0):
        super().__init__()
        self.downsample = DownsampleLayer3D(in_ch, out_ch)

        blocks = []
        for i in range(num_blocks):
            blocks.append(ConvNeXtBlock3D(out_ch, drop_path=drop_path))
        self.blocks = nn.Sequential(*blocks)

        self.se = SEBlock3D(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        x = self.se(x)
        return x


class EnhancedCNNBranch3D(nn.Module):
    """
    ConvNeXt-style 3D CNN branch with SE attention.

    Produces multi-scale features matching SAM-Med3D's 16× downsampling:
      128³ → 64³(c) → 32³(2c) → 16³(4c) → 8³(8c)

    Args:
        in_channels: input channels (4 for BraTS MRI)
        base_ch: base channel width (default 32)
        blocks_per_stage: ConvNeXt blocks per stage (list of 4)
        use_se: whether to use SE attention
        drop_path: stochastic depth rate
    """

    def __init__(self, in_channels=4, base_ch=32,
                 blocks_per_stage=[1, 1, 2, 1],
                 use_se=True, drop_path=0.1):
        super().__init__()
        ch = base_ch
        channels = [ch, ch * 2, ch * 4, ch * 8]  # [32, 64, 128, 256]

        # Stem: standard conv to initial feature maps (no ConvNeXt at full res for memory)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, channels[0], kernel_size=4, stride=2, padding=1, bias=False),
            LayerNorm3d(channels[0]),
            nn.GELU(),
        )

        # Build stages
        self.stage2 = ConvNeXtStage3D(channels[0], channels[1],
                                       num_blocks=blocks_per_stage[1],
                                       use_se=use_se, drop_path=drop_path)
        self.stage3 = ConvNeXtStage3D(channels[1], channels[2],
                                       num_blocks=blocks_per_stage[2],
                                       use_se=use_se, drop_path=drop_path)
        self.stage4 = ConvNeXtStage3D(channels[2], channels[3],
                                       num_blocks=blocks_per_stage[3],
                                       use_se=use_se, drop_path=drop_path)

        # Optional: ConvNeXt block at stem resolution (64³)
        stem_blocks = []
        for _ in range(blocks_per_stage[0]):
            stem_blocks.append(ConvNeXtBlock3D(channels[0], drop_path=drop_path))
        if use_se:
            stem_blocks.append(SEBlock3D(channels[0]))
        self.stem_blocks = nn.Sequential(*stem_blocks)

        self.out_channels = channels  # [32, 64, 128, 256]

    def forward(self, x):
        """
        Args:
            x: (B, 4, 128, 128, 128)
        Returns:
            skips: [s1(B,c,64³), s2(B,2c,32³), s3(B,4c,16³)]
            bottleneck: (B, 8c, 8, 8, 8)
        """
        s1 = self.stem_blocks(self.stem(x))  # (B, 32, 64³)
        s2 = self.stage2(s1)                  # (B, 64, 32³)
        s3 = self.stage3(s2)                  # (B, 128, 16³)
        s4 = self.stage4(s3)                  # (B, 256, 8³)

        return [s1, s2, s3], s4
