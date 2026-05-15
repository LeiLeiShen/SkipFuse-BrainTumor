"""
Loss functions for Hybrid SAM-Med3D on BraTS2021.

Compound loss: Dice + Focal + Boundary (with linear warmup schedule).

Components:
  - DiceLoss: per-class Dice with optional class weighting
  - FocalLoss: frequency-balanced Focal with per-region gamma option
  - BoundaryLoss: signed distance transform based boundary-aware loss
    (Kervadec et al., "Boundary loss for highly unbalanced segmentation",
     Medical Image Analysis, 2019)

Boundary Loss rationale:
  - Dice/Focal capture region overlap but are insensitive to boundary precision
  - BoundaryLoss uses signed distance transforms to directly penalize
    predictions that deviate from GT boundaries
  - Particularly effective for ET (enhancing tumor) which has the most
    irregular and hard-to-delineate boundaries
  - MUNet ablation (2025): removing Boundary Loss → ET DSC drops from
    0.836 to 0.642 (-23%), HD95 quadruples

Schedule:
  - Epochs [0, bdl_warmup): only Dice + Focal (coarse region learning)
  - Epochs [bdl_warmup, bdl_warmup + bdl_ramp): linearly ramp Boundary weight
  - Epochs [bdl_warmup + bdl_ramp, ...): full Boundary weight
  This follows the standard practice: boundary loss is unstable when
  predictions are far from correct, so Dice/Focal must first establish
  a reasonable segmentation before boundary refinement kicks in.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


# ===========================================================================
# Dice Loss
# ===========================================================================

class DiceLoss(nn.Module):
    """
    Soft Dice Loss over all classes.

    Skips classes with zero GT presence to avoid division artifacts.
    Uses per-sample Dice (not batch Dice) for stable gradients.
    """

    def __init__(self, num_classes=4, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, target):
        """
        Args:
            logits: (B, C, D, H, W)
            target: (B, D, H, W) long
        Returns:
            scalar loss
        """
        pred = F.softmax(logits, dim=1)
        target_oh = F.one_hot(target, self.num_classes)        # (B,D,H,W,C)
        target_oh = target_oh.permute(0, 4, 1, 2, 3).float()  # (B,C,D,H,W)

        losses = []
        for c in range(self.num_classes):
            p, g = pred[:, c], target_oh[:, c]
            inter = (p * g).sum()
            card = p.sum() + g.sum()
            if card < self.smooth:      # class not present
                continue
            losses.append(1.0 - (2 * inter + self.smooth) / (card + self.smooth))

        if not losses:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return torch.stack(losses).mean()


# ===========================================================================
# Focal Loss
# ===========================================================================

class FocalLoss(nn.Module):
    """
    Frequency-balanced Focal Loss.

    Alpha weights inversely proportional to class frequencies in BraTS2021:
      BG ~98.93%, NCR ~0.29%, ED ~0.31%, ET ~0.79%

    Args:
        num_classes: number of classes (4 for BraTS)
        gamma: focusing parameter (higher = more focus on hard examples)
        et_gamma_boost: additional gamma for ET class (class 3)
            ET is the hardest sub-region; slightly higher gamma
            forces the model to focus more on difficult ET voxels.
            Set to 0 to disable (default behavior = uniform gamma).
    """

    def __init__(self, num_classes=4, gamma=2.0, et_gamma_boost=0.0):
        super().__init__()
        self.gamma = gamma
        self.et_gamma_boost = et_gamma_boost

        # Inverse-frequency class weights, normalized
        freq = torch.tensor([98.93, 0.29, 0.31, 0.79])
        alpha = 1.0 / freq
        alpha = alpha / alpha.sum() * num_classes
        self.register_buffer('alpha', alpha.float())

    def forward(self, logits, target):
        """
        Args:
            logits: (B, C, D, H, W)
            target: (B, D, H, W) long
        """
        ce = F.cross_entropy(logits, target, reduction='none')  # (B,D,H,W)
        pt = F.softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)

        # Per-voxel gamma: boost for ET class if configured
        if self.et_gamma_boost > 0:
            gamma_map = torch.full_like(ce, self.gamma)
            gamma_map[target == 3] = self.gamma + self.et_gamma_boost
            focal_w = (1.0 - pt) ** gamma_map
        else:
            focal_w = (1.0 - pt) ** self.gamma

        alpha_w = self.alpha[target]
        return (alpha_w * focal_w * ce).mean()


# ===========================================================================
# Boundary Loss
# ===========================================================================

class BoundaryLoss(nn.Module):
    """
    Boundary Loss using signed distance transforms.

    For each foreground class c:
      1. Compute signed distance transform of GT mask:
         negative inside GT, positive outside GT
      2. Loss_c = mean(softmax_c * dist_map_c)
         → minimizing this pushes high-confidence predictions inside GT
           and penalizes predictions that spill outside

    Only computed on tumor classes (1=NCR, 2=ED, 3=ET), not background.
    Background boundary is implicitly handled by the foreground classes.

    Distance maps are normalized per-sample to [-1, 1] for numerical
    stability with mixed precision training.

    Reference:
        Kervadec et al., "Boundary loss for highly unbalanced segmentation",
        Medical Image Analysis, 2019
    """

    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes

    @staticmethod
    def _signed_dist(binary_mask):
        """
        Compute normalized signed distance transform.

        Args:
            binary_mask: (D, H, W) numpy bool array
        Returns:
            (D, H, W) float32, values in [-1, 1]
            negative inside mask, positive outside
        """
        if binary_mask.sum() == 0:
            # No GT voxels: all positive (penalize any false positive)
            return np.ones_like(binary_mask, dtype=np.float32)
        if binary_mask.all():
            # All GT: all negative (reward prediction everywhere)
            return -np.ones_like(binary_mask, dtype=np.float32)

        # Unsigned distances
        dist_out = distance_transform_edt(~binary_mask)   # outside GT
        dist_in = distance_transform_edt(binary_mask)     # inside GT

        # Signed: negative inside, positive outside
        signed = dist_out - dist_in

        # Normalize to [-1, 1]
        abs_max = max(np.abs(signed).max(), 1e-6)
        signed = signed / abs_max

        return signed.astype(np.float32)

    def forward(self, logits, target):
        """
        Args:
            logits: (B, C, D, H, W)
            target: (B, D, H, W) long
        Returns:
            scalar boundary loss (only on tumor classes 1,2,3)
        """
        pred = F.softmax(logits, dim=1)         # (B, C, D, H, W)
        target_np = target.detach().cpu().numpy()

        B = target.shape[0]

        # Build distance maps on CPU, then move to GPU
        # Shape: (B, C, D, H, W) but we only fill tumor classes
        dist_maps = torch.zeros(
            (B, self.num_classes) + target.shape[1:],
            dtype=torch.float32, device=logits.device
        )

        for b in range(B):
            for c in range(1, self.num_classes):    # skip background (c=0)
                gt_c = (target_np[b] == c)
                dist_maps[b, c] = torch.from_numpy(self._signed_dist(gt_c))

        # Loss = mean of (softmax * signed_distance) for tumor classes
        # Minimizing: pushes predictions to align with GT boundaries
        boundary_loss = (pred[:, 1:] * dist_maps[:, 1:]).mean()

        return boundary_loss


# ===========================================================================
# Compound Loss with Scheduling
# ===========================================================================

class RegionDiceLoss(nn.Module):
    """
    Region-based Dice Loss aligned with BraTS evaluation metric (ET/TC/WT).

    Derives 3 nested binary regions from 4-class softmax:
        P(ET) = p3
        P(TC) = p1 + p3      (NCR + ET)
        P(WT) = p1 + p2 + p3 (all tumor)

    Directly optimizes evaluation metric. Avoids wasting capacity on
    NCR vs ED distinction which BraTS evaluation does not measure.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        prob = F.softmax(logits, dim=1)
        p_et = prob[:, 3]
        p_tc = prob[:, 1] + prob[:, 3]
        p_wt = prob[:, 1] + prob[:, 2] + prob[:, 3]

        g_et = (target == 3).float()
        g_tc = ((target == 1) | (target == 3)).float()
        g_wt = (target >= 1).float()

        losses = []
        for p, g in [(p_et, g_et), (p_tc, g_tc), (p_wt, g_wt)]:
            inter = (p * g).sum()
            card = p.sum() + g.sum()
            if card < self.smooth:
                continue
            losses.append(1.0 - (2 * inter + self.smooth) / (card + self.smooth))

        if not losses:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return torch.stack(losses).mean()


class CompoundLoss(nn.Module):
    """
    Compound loss: Dice (per-class OR region) + Focal + Boundary.

    Args:
        use_region_dice: use RegionDiceLoss (ET/TC/WT) instead of per-class
        focal_weight: multiplier for Focal Loss (0.5 recommended with region)
        bdl_weight/warmup/ramp: Boundary Loss scheduling
        et_gamma_boost: extra Focal gamma on ET class
    """

    def __init__(self, num_classes=4, focal_gamma=2.0, et_gamma_boost=0.0,
                 bdl_weight=0.0, bdl_warmup=20, bdl_ramp=20,
                 use_region_dice=False, focal_weight=1.0):
        super().__init__()
        if use_region_dice:
            self.dice_fn = RegionDiceLoss()
        else:
            self.dice_fn = DiceLoss(num_classes)
        self.focal_fn = FocalLoss(num_classes, gamma=focal_gamma,
                                  et_gamma_boost=et_gamma_boost)
        self.boundary_fn = BoundaryLoss(num_classes)
        self.bdl_weight = bdl_weight
        self.bdl_warmup = bdl_warmup
        self.bdl_ramp = bdl_ramp
        self.focal_weight = focal_weight
        self.use_region_dice = use_region_dice

    def _get_bdl_weight(self, epoch):
        if epoch < self.bdl_warmup:
            return 0.0
        ramp_progress = (epoch - self.bdl_warmup) / max(self.bdl_ramp, 1)
        return self.bdl_weight * min(ramp_progress, 1.0)

    def forward(self, logits, target, epoch):
        dice_loss = self.dice_fn(logits, target)
        focal_loss = self.focal_fn(logits, target)

        w_bdl = self._get_bdl_weight(epoch)
        if w_bdl > 0:
            bdl_loss = self.boundary_fn(logits, target)
            total = dice_loss + self.focal_weight * focal_loss + w_bdl * bdl_loss
        else:
            bdl_loss = torch.tensor(0.0, device=logits.device)
            total = dice_loss + self.focal_weight * focal_loss

        components = {
            'dice': dice_loss.item(),
            'focal': focal_loss.item(),
            'boundary': bdl_loss.item(),
            'bdl_weight': w_bdl,
        }
        return total, components


# ===========================================================================
# ET Post-Processing (inference only)
# ===========================================================================

def brats_postprocess(pred_vol, et_min=200, tc_min=200, wt_min=200):
    """
    BraTS-standard post-processing on predicted segmentation volume.

    Rules (following nnU-Net BraTS conventions):
      1. If ET region (label 3) has fewer than et_min voxels,
         replace ET with NCR (label 1) — small ET is likely noise
      2. If TC region (labels 1+3) has fewer than tc_min voxels,
         replace TC with ED (label 2) or background
      3. If WT region (labels 1+2+3) has fewer than wt_min voxels,
         remove all tumor labels

    Args:
        pred_vol: (D, H, W) numpy int array with labels {0,1,2,3}
        et_min: minimum voxel count for ET region
        tc_min: minimum voxel count for TC region (NCR+ET)
        wt_min: minimum voxel count for WT region (all tumor)
    Returns:
        processed: (D, H, W) numpy int array
    """
    out = pred_vol.copy()

    # Rule 1: small ET → NCR
    et_mask = (out == 3)
    if et_mask.sum() < et_min:
        out[et_mask] = 1  # relabel ET as NCR

    # Rule 2: small TC → remove (set to background or ED)
    tc_mask = (out == 1) | (out == 3)
    if tc_mask.sum() < tc_min:
        out[tc_mask] = 0

    # Rule 3: small WT → remove all tumor
    wt_mask = (out >= 1)
    if wt_mask.sum() < wt_min:
        out[wt_mask] = 0

    return out
