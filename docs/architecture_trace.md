# SAM-Med3D Turbo Architecture Trace

> Auto-generated on Day 2
> Model: sam_med3d_turbo.pth (100.5M params)
> Project: /storage/main/users/leishen/Modify_SAM_Med3D

## 1. Overall Pipeline
```
Input (B, 1, 128, 128, 128)
    │
    ▼ Patch Embed: Conv3d(1→768, k=16, s=16)
    │
(B, 512, 768)  ←  8³=512 tokens, dim=768
    │
    ▼ 12x ViT Blocks (attention + MLP)
    │
(B, 512, 768)
    │
    ▼ Reshape → (B, 768, 8, 8, 8)
    │
    ▼ Neck: 768 → 384
    │
(B, 384, 8, 8, 8)  ←  核心特征
    │
    ├──→ [Prompt Encoder] → sparse (B, N, 384) + dense (B, 384, 8, 8, 8)
    │
    ▼ Mask Decoder
    │
(B, 1, 32, 32, 32) → trilinear → (B, 1, 128, 128, 128)
```

## 2. Image Encoder

| Layer | Input | Output | Notes |
|-------|-------|--------|-------|
| PatchEmbed3D | (B, 1, 128³) | (B, 512, 768) | Conv3d(1→768, k=16, s=16) |
| ViT Block x12 | (B, 512, 768) | (B, 512, 768) | Attention(num_heads=12) + MLP |
| Reshape | (B, 512, 768) | (B, 768, 8, 8, 8) | |
| Neck | (B, 768, 8, 8, 8) | (B, 384, 8, 8, 8) | Conv3d 768→384 |

## 3. Prompt Encoder

| Input | Shape | Notes |
|-------|-------|-------|
| Point coords | (B, N, 3) | xyz |
| Point labels | (B, N) | 0=neg, 1=pos |
| Sparse output | (B, N+1, 384) | +1 for no-mask token |
| Dense output | (B, 384, 8, 8, 8) | Zeros if no mask prompt |
| Positional encoding | (1, 384, 8, 8, 8) | |

## 4. Mask Decoder

| Stage | Shape | Notes |
|-------|-------|-------|
| Input features | (B, 384, 8, 8, 8) | From encoder |
| Transformer layers | 2 | Self-attn + cross-attn |
| Low-res output | (B, 1, 32, 32, 32) | |
| Final mask | (B, 1, 128, 128, 128) | Trilinear upsampling |
| IoU prediction | (B, 1) | Confidence score |
| num_mask_tokens | 4 | For multi-mask mode |

## 5. Parameter Count

| Component | Parameters | % |
|-----------|-----------|---|
| Image Encoder (total) | 92,924,928 | 92.5% |
|   Patch Embed | 3,146,496 | 3.1% |
|   ViT Blocks (x12) | 85,107,456 | 84.7% |
|   Neck | 4,277,760 | 4.3% |
| Prompt Encoder | 8,668 | 0.0% |
| Mask Decoder | 7,575,636 | 7.5% |
| **TOTAL** | **100,509,232** | **100%** |

## 6. Key Dimensions (ACTUAL - turbo version)

- **ViT dim**: 768
- **Neck output dim**: 384 (NOT 256!)
- **Spatial**: 128³ → 8³ (16x downsample) → 32³ (decoder) → 128³
- **Tokens**: 8³ = 512
- **ViT blocks**: 12
- **Decoder transformer layers**: 2

## 7. Modification Points for ProtoSAM

1. **Input**: Conv3d(1→768) 改为 Conv3d(4→768) — BraTS 4 modalities
2. **LoRA**: 注入到 12 个 ViT blocks 的 qkv projection
3. **CFM3D**: 在 encoder 输出 (B, 384, 8, 8, 8) 上做 contextual modulation
4. **PPR3D**: 接收 prototypes (N_cls, 384)，输出 refined prototypes
5. **PPG3D**: prototypes → sparse (B, N, 384) + dense (B, 384, 8, 8, 8)
6. **Mask Decoder**: 输出从 1 类 → 5 类 (BraTS multi-class)

⚠️ 所有通道数用 384，不是计划中的 256！
