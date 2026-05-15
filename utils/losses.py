"""
Loss functions for ProtoSAM-Med3D
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    
    def __init__(self, num_classes=4, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, pred_logits, target):
        pred_soft = F.softmax(pred_logits, dim=1)
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 4, 1, 2, 3).float()
        dice_per_class = []
        for c in range(self.num_classes):
            pred_c = pred_soft[:, c]
            target_c = target_onehot[:, c]
            intersection = (pred_c * target_c).sum()
            cardinality = pred_c.sum() + target_c.sum()
            if cardinality < self.smooth:
                continue
            dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            dice_per_class.append(1.0 - dice)
        if len(dice_per_class) == 0:
            return torch.tensor(0.0, device=pred_logits.device, requires_grad=True)
        return torch.stack(dice_per_class).mean()


class FocalLoss(nn.Module):
    
    def __init__(self, num_classes=4, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        if alpha is None:
            freq = torch.tensor([98.93, 0.29, 0.31, 0.79])
            alpha = 1.0 / freq
            alpha = alpha / alpha.sum() * num_classes
        self.register_buffer('alpha', alpha.float())
    
    def forward(self, pred_logits, target):
        ce = F.cross_entropy(pred_logits, target, reduction='none')
        pred_soft = F.softmax(pred_logits, dim=1)
        pt = pred_soft.gather(1, target.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - pt) ** self.gamma

        alpha_weight = self.alpha.to(target.device)[target]  # (B, D, H, W)
        return (alpha_weight * focal_weight * ce).mean()


class MultiStageLoss(nn.Module):
    
    def __init__(self, num_classes=4, num_stages=3,
                 dice_weight=1.0, focal_weight=1.0, iou_weight=0.5):
        super().__init__()
        self.dice_loss = DiceLoss(num_classes)
        self.focal_loss = FocalLoss(num_classes)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.iou_weight = iou_weight
        stage_weights = [0.2 + 0.15 * i for i in range(num_stages)]
        total = sum(stage_weights)
        self.stage_weights = [w / total for w in stage_weights]
    
    def forward(self, all_masks, all_ious, target):
        total_loss = 0.0
        loss_dict = {}
        for i, (masks, ious) in enumerate(zip(all_masks, all_ious)):
            dice = self.dice_loss(masks, target)
            focal = self.focal_loss(masks, target)
            with torch.no_grad():
                pred_classes = masks.argmax(dim=1)
                actual_dice = []
                for c in range(masks.shape[1]):
                    pred_c = (pred_classes == c).float()
                    gt_c = (target == c).float()
                    inter = (pred_c * gt_c).sum(dim=(1, 2, 3))
                    union = pred_c.sum(dim=(1, 2, 3)) + gt_c.sum(dim=(1, 2, 3))
                    actual_dice.append((2 * inter + 1.0) / (union + 1.0))
                actual_dice = torch.stack(actual_dice, dim=1)
            iou_loss = F.mse_loss(ious.sigmoid(), actual_dice)
            stage_loss = self.dice_weight * dice + self.focal_weight * focal + self.iou_weight * iou_loss
            total_loss = total_loss + self.stage_weights[i] * stage_loss
            loss_dict[f'stage{i}_dice'] = dice.item()
            loss_dict[f'stage{i}_focal'] = focal.item()
            loss_dict[f'stage{i}_iou'] = iou_loss.item()
            loss_dict[f'stage{i}_total'] = stage_loss.item()
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict
