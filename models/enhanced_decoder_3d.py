"""
Enhanced Decoder for 3D Hybrid SAM-Med3D
Attention-gated skip connections + deep supervision.

Compared to basic HybridDecoder3D (concat + conv):
  - Attention gates: learnable weighting of skip features before fusion
  - Deep supervision: auxiliary segmentation heads at 16³ and 32³
  - ConvNeXt-style fusion blocks: more expressive skip feature processing

Architecture:
  Fused bottleneck (384, 8³)
    → up1 + AG(skip3) → (256, 16³) → aux_head1
    → up2 + AG(skip2) → (128, 32³) → aux_head2
    → up3 + AG(skip1) → (64, 64³)
    → interp 2× → (64, 128³) → main_head → (C, 128³)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate3D(nn.Module):
    """
    Attention gate for skip connections.
    Learns to suppress irrelevant spatial regions in skip features
    based on the gating signal from the decoder path.

    gate (from decoder, lower res) + skip (from CNN, higher res)
    → attention weights → filtered skip features
    """

    def __init__(self, gate_ch, skip_ch, inter_ch=None):
        super().__init__()
        inter_ch = inter_ch or max(skip_ch // 2, 16)

        self.W_gate = nn.Conv3d(gate_ch, inter_ch, 1, bias=False)
        self.W_skip = nn.Conv3d(skip_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv3d(inter_ch, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        """
        Args:
            gate: (B, gate_ch, D', H', W') — decoder features (lower resolution)
            skip: (B, skip_ch, D, H, W) — CNN skip features (higher resolution)
        Returns:
            attended_skip: (B, skip_ch, D, H, W)
        """
        # Upsample gate to match skip resolution
        g = self.W_gate(gate)
        if g.shape[2:] != skip.shape[2:]:
            g = F.interpolate(g, size=skip.shape[2:], mode='trilinear', align_corners=False)

        s = self.W_skip(skip)
        attn = self.psi(self.relu(g + s))  # (B, 1, D, H, W)
        return skip * attn


class EnhancedSkipFusion3D(nn.Module):
    """
    Skip fusion with attention gate + dual conv processing.
    Gate → attend skip → concat with decoder features → dual conv.
    """

    def __init__(self, up_ch, skip_ch, out_ch):
        super().__init__()
        self.attn_gate = AttentionGate3D(gate_ch=up_ch, skip_ch=skip_ch)
        self.conv = nn.Sequential(
            nn.Conv3d(up_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
        )

    def forward(self, x_up, x_skip):
        x_skip = self.attn_gate(gate=x_up, skip=x_skip)
        x = torch.cat([x_up, x_skip], dim=1)
        return self.conv(x)


class EnhancedHybridDecoder3D(nn.Module):
    """
    Enhanced U-Net decoder with attention-gated skip connections
    and optional deep supervision.

    Args:
        in_dim: bottleneck channel dim (384 from ViT fusion)
        skip_channels: CNN skip feature channels [32, 64, 128]
        num_classes: segmentation classes (4 for BraTS)
        deep_supervision: whether to add auxiliary heads
    """

    def __init__(self, in_dim=384, skip_channels=[32, 64, 128],
                 num_classes=4, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision

        # Up1: 8³ → 16³, fuse with skip3 (128ch)
        self.up1 = nn.ConvTranspose3d(in_dim, 256, kernel_size=2, stride=2, bias=False)
        self.fuse1 = EnhancedSkipFusion3D(256, skip_channels[2], 256)

        # Up2: 16³ → 32³, fuse with skip2 (64ch)
        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2, bias=False)
        self.fuse2 = EnhancedSkipFusion3D(128, skip_channels[1], 128)

        # Up3: 32³ → 64³, fuse with skip1 (32ch)
        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, bias=False)
        self.fuse3 = EnhancedSkipFusion3D(64, skip_channels[0], 64)

        # Main head: 64³ → 128³
        self.head = nn.Sequential(
            nn.Conv3d(64, 64, 3, padding=1, bias=False),
            nn.InstanceNorm3d(64),
            nn.GELU(),
            nn.Conv3d(64, num_classes, 1),
        )

        # Deep supervision auxiliary heads
        if deep_supervision:
            self.aux_head1 = nn.Conv3d(256, num_classes, 1)  # at 16³
            self.aux_head2 = nn.Conv3d(128, num_classes, 1)  # at 32³

    def forward(self, x, skips):
        """
        Args:
            x: bottleneck features (B, 384, 8, 8, 8)
            skips: [s1(B,32,64³), s2(B,64,32³), s3(B,128,16³)]
        Returns:
            if training and deep_supervision:
                (main_logits, [aux1_logits, aux2_logits])
            else:
                main_logits: (B, C, 128, 128, 128)
        """
        s1, s2, s3 = skips

        # 8³ → 16³
        d1 = self.up1(x)
        d1 = self.fuse1(d1, s3)

        # 16³ → 32³
        d2 = self.up2(d1)
        d2 = self.fuse2(d2, s2)

        # 32³ → 64³
        d3 = self.up3(d2)
        d3 = self.fuse3(d3, s1)

        # 64³ → 128³
        out = F.interpolate(d3, scale_factor=2, mode='trilinear', align_corners=False)
        main_logits = self.head(out)

        if self.deep_supervision and self.training:
            # Auxiliary outputs upsampled to 128³ for loss computation
            aux1 = F.interpolate(self.aux_head1(d1), size=main_logits.shape[2:],
                                 mode='trilinear', align_corners=False)
            aux2 = F.interpolate(self.aux_head2(d2), size=main_logits.shape[2:],
                                 mode='trilinear', align_corners=False)
            return main_logits, [aux1, aux2]

        return main_logits
