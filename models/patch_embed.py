"""
4-Channel Patch Embedding for SAM-Med3D.

Confirmed from diagnostic:
  - embed_dim = 768 (vit_b_ori)
  - Original PatchEmbed3D.forward() does: proj(x) → permute(0,2,3,4,1)
  - Output format: (B, D', H', W', 768) — channels-last
  - pos_embed shape: (1, 8, 8, 8, 768) — also channels-last

This module must:
  1. Replace Conv3d(1→768) with Conv3d(4→768)
  2. Transfer pretrained weights via repeat÷4
  3. Apply the same permute as original so pos_embed addition works
"""

import torch
import torch.nn as nn


class PatchEmbed4Ch(nn.Module):
    """
    4-channel patch embedding that exactly replicates original PatchEmbed3D behavior.
    
    Conv3d(4, 768, k=16, s=16) → permute(0, 2, 3, 4, 1) → (B, D', H', W', 768)
    """
    
    def __init__(self, original_patch_embed, in_channels=4):
        super().__init__()
        
        # Read actual config from original
        self.embed_dim = original_patch_embed.proj.out_channels    # 768
        patch_size = original_patch_embed.proj.kernel_size          # (16,16,16)
        stride = original_patch_embed.proj.stride                   # (16,16,16)
        
        # New 4-channel projection (same embed_dim, patch_size, stride)
        self.proj = nn.Conv3d(
            in_channels, self.embed_dim,
            kernel_size=patch_size, stride=stride
        )
        
        # Transfer pretrained weights: (768, 1, 16, 16, 16) → (768, 4, 16, 16, 16) / 4
        with torch.no_grad():
            orig_weight = original_patch_embed.proj.weight.data
            new_weight = orig_weight.repeat(1, in_channels, 1, 1, 1) / in_channels
            self.proj.weight.copy_(new_weight)
            if original_patch_embed.proj.bias is not None:
                self.proj.bias.copy_(original_patch_embed.proj.bias.data)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C X Y Z -> B X Y Z C  (must match original for pos_embed addition)
        x = x.permute(0, 2, 3, 4, 1)
        return x
