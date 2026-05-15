# PGP-SAM Architecture Analysis

> Auto-generated on Day 3
> Source: /storage/main/users/leishen/Modify_SAM_Med3D/PGP-SAM
> Purpose: Guide 2D→3D adaptation for ProtoSAM-Med3D

## 1. Project Structure

See Cell 1 output for full directory listing.

## 2. Class Inventory

| Class | File | Line | __init__ args | forward() args |
|-------|------|------|---------------|----------------|
| TestingDataset | dataset.py | L18 | data_root_dir, mode, image_size | N/A |
| TrainingDataset | dataset.py | L66 | data_root_dir, image_size, scale | N/A |
| PGP_SAM | model.py | L15 | sam_checkpoint, sam_mode, model_type, stage, mask_ | images |
| GlobalPrototypes | model.py | L135 | feat_dim, num_classes, num_tokens |  |
| Trainer | train.py | L72 | args, model, train_dataloader, val_dataloader, log | N/A |
| SamAutomaticMaskGenerator | segment_anything/automatic_mask_generator.py | L35 | model, points_per_side, points_per_batch, pred_iou | N/A |
| SamPredictor | segment_anything/predictor.py | L17 | sam_model | N/A |
| MLPBlock | segment_anything/modeling/common.py | L13 | embedding_dim, mlp_dim, act | x |
| LayerNorm2d | segment_anything/modeling/common.py | L31 | num_channels, eps | x |
| ImageEncoderViT | segment_anything/modeling/image_encoder.py | L19 | model_type, img_size, patch_size, in_chans, embed_ | x |
| Block | segment_anything/modeling/image_encoder.py | L132 | dim, num_heads, mlp_ratio, qkv_bias, norm_layer, a | x |
| LoraBlock | segment_anything/modeling/image_encoder.py | L198 | dim, num_heads, mlp_ratio, qkv_bias, norm_layer, a | x |
| Attention | segment_anything/modeling/image_encoder.py | L263 | dim, num_heads, qkv_bias, use_rel_pos, rel_pos_zer | x |
| LoraAttention | segment_anything/modeling/image_encoder.py | L321 | dim, num_heads, qkv_bias, use_rel_pos, rel_pos_zer | x |
| PatchEmbed | segment_anything/modeling/image_encoder.py | L503 | kernel_size, stride, padding, in_chans, embed_dim | x |
| MaskDecoder | segment_anything/modeling/mask_decoder.py | L20 |  | image_embeddings, image_pe, sparse_prompt_embeddin |
| MLP | segment_anything/modeling/mask_decoder.py | L159 | input_dim, hidden_dim, output_dim, num_layers, sig | x |
| PromptEncoder | segment_anything/modeling/prompt_encoder.py | L16 | embed_dim, image_embedding_size, input_image_size, | points, boxes, masks |
| PositionEmbeddingRandom | segment_anything/modeling/prompt_encoder.py | L171 | num_pos_feats, scale | size |
| Sam | segment_anything/modeling/sam.py | L18 | image_encoder, prompt_encoder, mask_decoder, pixel | batched_input, multimask_output |
| TwoWayTransformer | segment_anything/modeling/transformer.py | L16 | depth, embedding_dim, num_heads, mlp_dim, activati | image_embedding, image_pe, point_embedding |
| TwoWayAttentionBlock | segment_anything/modeling/transformer.py | L109 | embedding_dim, num_heads, mlp_dim, activation, att | queries, keys, query_pe, key_pe |
| Attention | segment_anything/modeling/transformer.py | L186 | embedding_dim, num_heads, downsample_rate, dropout | q, k, v |
| Baseline | segment_anything/ppm/baseline.py | L21 | stage, embed_dim, feat_dim, num_heads, num_classes | idx, interm_embed, out_embed, mask_feats, prototyp |
| FeatureRefinement | segment_anything/ppm/baseline.py | L70 | in_dim, conv_dim, out_dim, norm | interm_embed, out_embed |
| PromptGenerator | segment_anything/ppm/baseline.py | L93 | feat_dim, feat_size, num_classes, num_tokens, norm | image_embed, prototypes, class_prototypes, query_p |
| PositionEmbeddingSine | segment_anything/ppm/common.py | L16 | num_pos_feats, temperature, normalize, scale | x, mask |
| PositionEmbeddingRandom | segment_anything/ppm/common.py | L58 | num_pos_feats, scale | size |
| MLPBlock | segment_anything/ppm/common.py | L104 | embedding_dim, mlp_dim, act | x |
| MLP | segment_anything/ppm/common.py | L122 | input_dim, hidden_dim, output_dim, num_layers, sig | x |
| LayerNorm2d | segment_anything/ppm/common.py | L157 | num_channels, eps | x |
| ConvLayer2d | segment_anything/ppm/module.py | L8 | in_channels, out_channels, kernel_size, stride, di | x |
| ConvLayer1d | segment_anything/ppm/module.py | L55 | in_channels, out_channels, norm_channels, kernel_s | x |
| UpSample2d | segment_anything/ppm/module.py | L102 | in_planes, out_planes, norm_channels, kernel_size, | x |
| HierMaskDecoder | segment_anything/ppm/prototype_mask_decoder.py | L18 |  | image_embeddings, dense_prompt_embeddings, sparse_ |
| Attention | segment_anything/ppm/prototype_mask_decoder.py | L162 | embedding_dim, num_heads, downsample_rate | q, k, v |
| PrePrompt | segment_anything/ppm/prototype_prompt_encoder.py | L20 | feat_dim, num_classes, num_tokens | image_embed, inter_prototypes, intra_prototypes, i |
| CSM | segment_anything/ppm/prototype_prompt_encoder.py | L58 | hidden_dim, factor | x |
| PrototypePromptEncoder | segment_anything/ppm/prototype_prompt_encoder.py | L79 | stage, embed_dim, feat_dim, num_heads, num_classes | idx, interm_embed, out_embed, mask_embed, inter_pr |
| FeatureRefinement | segment_anything/ppm/prototype_prompt_encoder.py | L141 | in_dim, out_dim | interm_embed, out_embed |
| SCM | segment_anything/ppm/prototype_prompt_encoder.py | L171 | dim | x |
| FFC | segment_anything/ppm/prototype_prompt_encoder.py | L230 | in_dim, out_dim | x |
| PrototypeRefinement | segment_anything/ppm/prototype_prompt_encoder.py | L256 | feat_dim, num_heads, num_classes, num_tokens | image_embed, mask_embed, inter_prototypes, intra_p |
| ClassAttention | segment_anything/ppm/prototype_prompt_encoder.py | L344 | embedding_dim, in_feature, out_feature, num_heads, | query_prototypes, image_embed, mask_embed, masks |
| FFL | segment_anything/ppm/prototype_prompt_encoder.py | L436 | in_dim, out_dim | prototypes |
| PromptGenerator | segment_anything/ppm/prototype_prompt_encoder.py | L470 | feat_dim, num_classes, num_tokens | image_embed, inter_prototypes, intra_prototypes, i |
| DensePromptGenerator | segment_anything/ppm/prototype_prompt_encoder.py | L497 | feat_dim, num_classes, num_tokens | image_embed, inter_prototypes, intra_prototypes, m |
| DeformAttn | segment_anything/ppm/prototype_prompt_encoder.py | L555 | dim | image_embed |
| DeformConv | segment_anything/ppm/prototype_prompt_encoder.py | L594 | d_modal, kernel_size, groups, dilation | x |
| SparsePromptGenerator | segment_anything/ppm/prototype_prompt_encoder.py | L637 | feat_dim, num_classes, num_tokens | image_embed, inter_prototypes, intra_prototypes, m |
| PrototypeAdapter | segment_anything/ppm/prototype_prompt_encoder.py | L686 | feat_dim, num_classes, num_tokens | inter_prototypes, intra_prototypes |
| TwoWayTransformer | segment_anything/ppm/transformer.py | L19 | depth, embedding_dim, num_heads, mlp_dim, activati | image_embedding, image_pe, token, token_pe, ps_mas |
| TwoWayAttentionBlock | segment_anything/ppm/transformer.py | L114 | embedding_dim, num_heads, mlp_dim, activation, att | queries, keys, query_pe, key_pe, ps_masks |
| Attention | segment_anything/ppm/transformer.py | L190 | embedding_dim, num_heads, downsample_rate | q, k, v |
| MaskAttention | segment_anything/ppm/transformer.py | L249 | embedding_dim, num_heads, downsample_rate | q, k, v, ps_masks |
| PrototypeAttention | segment_anything/ppm/transformer.py | L316 | embedding_dim, num_heads, downsample_rate | q, k, v, ps_masks |
| MaskData | segment_anything/utils/amg.py | L16 |  | N/A |
| SamOnnxModel | segment_anything/utils/onnx.py | L17 | model, return_single_mask, use_stability_score, re | image_embeddings, point_coords, point_labels, mask |
| ResizeLongestSide | segment_anything/utils/transforms.py | L16 | target_length | N/A |
| DiceLoss | utils/loss.py | L27 | ignore_index, smooth | preds, masks |
| FocalLoss | utils/loss.py | L57 | alpha, gamma, num_classes, ignore_index | preds, masks |
| LogLR | utils/loss.py | L108 | optimizer, warmup_iters, total_iters, lr, last_epo | N/A |
| WarmupCosineLR | utils/loss.py | L127 | optimizer, warmup_iters, total_iters, base_lr, bas | N/A |
| RunTsne | utils/tsne.py | L23 | dataset_name, num_class, output_dir, extention, du | N/A |


## 3. Core Module Analysis

### 3.1 CFM (Contextual Feature Modulation)
- **Role**: Enhance encoder features with contextual information
- **Key components**: Strip pooling (directional pooling) + channel attention
- **2D operations**: AdaptiveAvgPool2d, Conv2d
- **3D adaptation**: 3 orthogonal strip pools, Conv2d→Conv3d

### 3.2 PPR (Prototype-based Prompt Refinement)
- **Role**: Refine class prototypes using image features
- **Key components**: Learnable prototypes, cosine similarity, cross-attention
- **Input**: Prototypes (N_cls, dim) + Image features (B, dim, H, W)
- **3D adaptation**: dim 256→384, spatial 64²→8³, N_cls 1→5

### 3.3 PPG (Prototype-based Prompt Generator)
- **Role**: Convert refined prototypes into SAM-compatible prompts
- **Output**: Sparse embeddings + Dense embeddings
- **3D adaptation**: Dense (B,384,8,8,8), Sparse (B,N,384)

## 4. 2D→3D Adaptation Summary

| Operation | 2D | 3D |
|-----------|----|----|
| Conv | Conv2d | Conv3d |
| Pool | AdaptiveAvgPool2d | AdaptiveAvgPool3d |
| Spatial | (H,W) = (64,64) | (D,H,W) = (8,8,8) |
| Tokens | 4096 | 512 |
| Embedding | 256 | 384 |
| Strip pool dirs | 2 (H,W) | 3 (D,H,W) |
| Output classes | 1 | 5 |
| Input modalities | 3 (RGB) | 4 (MRI) |

## 5. Key Design Decisions for ProtoSAM-Med3D

1. **Freeze strategy**: Freeze ViT blocks + inject LoRA (r=8) into qkv
2. **Prototype count**: 5 learnable prototypes (BG + 4 tumor classes)
3. **Multi-class output**: Modify decoder for 5-channel sigmoid output
4. **Loss**: Dice + Focal per class + Contrastive between prototypes
5. **Few-shot**: K-shot support → masked average pooling → prototype init

## 6. Implementation Priority (Week 2)

| Day | Task |
|-----|------|
| Day 8 | Modify patch_embed Conv3d(1→4) for 4-channel input |
| Day 9 | Implement LoRA3D for ViT qkv projections |
| Day 10 | Build CFM3D module |
| Day 11 | Build PPR3D module |
| Day 12 | Build PPG3D module |
| Day 13 | Modify MaskDecoder3D for 5-class output |
| Day 14 | Integration test: full forward pass |
