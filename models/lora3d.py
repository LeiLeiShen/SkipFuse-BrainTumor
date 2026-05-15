"""
LoRA3D: 低秩适配层 for SAM-Med3D

在冻结的 ViT qkv 投影旁注入可训练的低秩分支
output = frozen_linear(x) + (alpha/r) * B(A(x))
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    
    def __init__(self, original_linear, r=8, alpha=16):
        super().__init__()
        
        self.original = original_linear
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        for param in self.original.parameters():
            param.requires_grad = False
        
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scale = alpha / r
        
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
        
        self.r = r
        self.alpha = alpha
    
    def forward(self, x):
        return self.original(x) + self.scale * self.lora_B(self.lora_A(x))


def inject_lora_to_encoder(image_encoder, r=8, alpha=16, target_blocks=None):
    if target_blocks is None:
        target_blocks = list(range(len(image_encoder.blocks)))
    
    lora_params = 0
    
    for idx in target_blocks:
        block = image_encoder.blocks[idx]
        attn = block.attn
        
        original_qkv = attn.qkv
        lora_qkv = LoRALinear(original_qkv, r=r, alpha=alpha)
        attn.qkv = lora_qkv
        
        lora_params += sum(p.numel() for p in lora_qkv.lora_A.parameters())
        lora_params += sum(p.numel() for p in lora_qkv.lora_B.parameters())
    
    return image_encoder, lora_params


def freeze_encoder_with_lora(image_encoder):
    """冻结 encoder 全部参数，只解冻 LoRA"""
    for param in image_encoder.parameters():
        param.requires_grad = False
    
    for block in image_encoder.blocks:
        if hasattr(block.attn.qkv, 'lora_A'):
            for param in block.attn.qkv.lora_A.parameters():
                param.requires_grad = True
            for param in block.attn.qkv.lora_B.parameters():
                param.requires_grad = True
    
    return image_encoder
