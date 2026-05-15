"""
ViTSkipExtractor: extracts intermediate ViT block features and upsamples them
to match decoder skip connection resolutions, providing high-semantic features
at the resolutions where ET is still preserved.

Design rationale:
  - SAM-Med3D's 16× downsampling causes ET to vanish at 8³ (25.7% of cases)
  - Intermediate ViT blocks (3, 6, 9) provide progressive semantic features
    all at 8³ but with different abstraction levels
  - Upsampling them via stacked ConvTranspose3d to 16³/32³/64³ creates a
    pseudo-feature-pyramid, similar to ViT-Det (Li et al., ECCV 2022)
  - These ViT skips are concatenated with existing CNN skips at the decoder

Architecture:
  block_3 output (B, 8, 8, 8, 768) → 3× ConvT3d → (B, 64,  64³)
  block_6 output (B, 8, 8, 8, 768) → 2× ConvT3d → (B, 128, 32³)
  block_9 output (B, 8, 8, 8, 768) → 1× ConvT3d → (B, 256, 16³)

Outputs match CNN branch skip channels [64, 128, 256] for direct concat fusion.

Hook-based extraction: registers forward hooks on the target ViT blocks,
captures their outputs in a dict during the encoder forward pass, then
processes them after the encoder completes. No modification to SAM-Med3D.
"""

import torch
import torch.nn as nn


class ConvTUpBlock3D(nn.Module):
    """Single ConvTranspose3d + BN + ReLU upsampling block (stride=2)."""
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2,
                                        bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ViTSkipBranch(nn.Module):
    """
    One branch: takes a single ViT block output (8³, 768d, channels-last)
    and progressively upsamples to target resolution.
    
    Args:
        in_dim: input channel dim (768 for ViT-B)
        out_ch: output channel count (matches corresponding CNN skip)
        n_upsample: number of 2× upsampling stages (1=16³, 2=32³, 3=64³)
    """
    
    def __init__(self, in_dim=768, out_ch=64, n_upsample=1):
        super().__init__()
        
        # Progressive channel reduction across upsampling stages
        # 768 → 384 → 192 → 96 etc., final stage maps to out_ch
        layers = []
        cur = in_dim
        for i in range(n_upsample):
            if i == n_upsample - 1:
                nxt = out_ch  # last stage: match target channels
            else:
                nxt = max(cur // 2, out_ch * 2)  # halve channels
            layers.append(ConvTUpBlock3D(cur, nxt))
            cur = nxt
        self.up_layers = nn.Sequential(*layers)
    
    def forward(self, x_chl):
        """
        Args:
            x_chl: (B, D, H, W, C) channels-last from ViT block
        Returns:
            (B, out_ch, D*2^n, H*2^n, W*2^n) channels-first
        """
        # Convert channels-last → channels-first for Conv3d
        x = x_chl.permute(0, 4, 1, 2, 3).contiguous()
        return self.up_layers(x)


class ViTSkipExtractor(nn.Module):
    """
    Extracts and upsamples intermediate ViT block features.
    
    Args:
        encoder: SAM-Med3D image_encoder (assumed to have .blocks ModuleList)
        block_indices: which blocks to extract (default [3, 6, 9], 0-indexed
                       so this means the 4th, 7th, 10th blocks)
        target_channels: output channels at each scale, matches CNN branch
                         skip channels [s1_ch, s2_ch, s3_ch]
        target_upsamples: number of 2× upsampling stages for each branch,
                          should be [3, 2, 1] for 64³/32³/16³ outputs
    """
    
    def __init__(self, encoder, block_indices=(3, 6, 9),
                 target_channels=(64, 128, 256),
                 target_upsamples=(3, 2, 1),
                 vit_dim=768):
        super().__init__()
        assert len(block_indices) == len(target_channels) == len(target_upsamples)
        
        self.block_indices = list(block_indices)
        self._captured = {}
        self._hooks = []
        
        # Register forward hooks on selected ViT blocks
        for idx in self.block_indices:
            blk = encoder.blocks[idx]
            hook = blk.register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)
        
        # Build upsampling branches (one per extracted block)
        self.branches = nn.ModuleList([
            ViTSkipBranch(in_dim=vit_dim, out_ch=ch, n_upsample=n_up)
            for ch, n_up in zip(target_channels, target_upsamples)
        ])
    
    def _make_hook(self, idx):
        def hook(module, input, output):
            # output shape: (B, D, H, W, C) channels-last from Block3D
            self._captured[idx] = output
        return hook
    
    def clear_cache(self):
        """Call after each forward pass to free captured tensors."""
        self._captured = {}
    
    def forward(self, _ignored=None):
        """
        Returns three upsampled ViT skip tensors at progressively higher
        resolutions. Must be called AFTER the encoder forward pass that
        populated self._captured via hooks.
        
        Returns:
            [vit_skip_64, vit_skip_32, vit_skip_16]
            shapes: [(B,64,64³), (B,128,32³), (B,256,16³)]
        """
        outputs = []
        for idx, branch in zip(self.block_indices, self.branches):
            assert idx in self._captured, (
                f"Block {idx} output not captured — did encoder forward run?")
            x = self._captured[idx]
            outputs.append(branch(x))
        return outputs
    
    def remove_hooks(self):
        """Cleanup: remove forward hooks (call before deleting model)."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
