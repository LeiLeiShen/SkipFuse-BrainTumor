"""
HierMaskDecoder3D: Hierarchical mask decoder with multi-class output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierMaskDecoder3D(nn.Module):
    
    def __init__(self, feat_dim=384, num_classes=4, output_size=128):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_size = output_size
        
        self.upscale = nn.Sequential(
            nn.ConvTranspose3d(feat_dim, feat_dim // 4, kernel_size=2, stride=2),
            nn.GroupNorm(32, feat_dim // 4), nn.GELU(),
            nn.ConvTranspose3d(feat_dim // 4, feat_dim // 8, kernel_size=2, stride=2),
            nn.GroupNorm(16, feat_dim // 8), nn.GELU())
        
        self.mask_head_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 4), nn.GELU(),
                nn.Linear(feat_dim // 4, feat_dim // 8))
            for _ in range(num_classes)])
        
        self.cross_attn = nn.MultiheadAttention(feat_dim, 8, batch_first=True)
        self.cross_norm_q = nn.LayerNorm(feat_dim)
        self.cross_norm_kv = nn.LayerNorm(feat_dim)
        self.self_attn = nn.MultiheadAttention(feat_dim, 8, batch_first=True)
        self.self_norm = nn.LayerNorm(feat_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(), nn.Linear(feat_dim * 2, feat_dim))
        self.iou_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2), nn.GELU(),
            nn.Linear(feat_dim // 2, num_classes))
    
    def forward(self, dense_embed, sparse_tokens, image_embed):
        B, C, D, H, W = image_embed.shape
        src = dense_embed + image_embed
        src_flat = src.flatten(2).permute(0, 2, 1)
        
        sparse_normed = self.self_norm(sparse_tokens)
        sparse_tokens = sparse_tokens + self.self_attn(sparse_normed, sparse_normed, sparse_normed)[0]
        q = self.cross_norm_q(sparse_tokens)
        kv = self.cross_norm_kv(src_flat)
        sparse_tokens = sparse_tokens + self.cross_attn(q, kv, kv)[0]
        sparse_tokens = sparse_tokens + self.ffn(sparse_tokens)
        
        upscaled = self.upscale(src)
        
        masks_lowres = []
        for cls_idx in range(self.num_classes):
            token = sparse_tokens[:, cls_idx, :] if cls_idx < sparse_tokens.shape[1] else sparse_tokens.mean(dim=1)
            hyper = self.mask_head_mlps[cls_idx](token)
            mask = torch.einsum('bcdhw,bc->bdhw', upscaled, hyper)
            masks_lowres.append(mask)
        
        masks_lowres = torch.stack(masks_lowres, dim=1)
        masks = F.interpolate(masks_lowres, size=self.output_size, mode='trilinear', align_corners=False)
        
        iou_token = sparse_tokens.mean(dim=1)
        iou_pred = self.iou_head(iou_token)
        
        return masks, iou_pred
