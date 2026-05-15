"""
Segmentation evaluation metrics (BraTS standard)
"""

import numpy as np


class SegmentationMetrics:
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.dice_scores = {'ET': [], 'TC': [], 'WT': []}
        self.per_class_dice = {0: [], 1: [], 2: [], 3: []}
    
    @staticmethod
    def dice_coefficient(pred, gt):
        if pred.sum() == 0 and gt.sum() == 0:
            return 1.0
        if pred.sum() == 0 or gt.sum() == 0:
            return 0.0
        intersection = (pred & gt).sum()
        return 2.0 * intersection / (pred.sum() + gt.sum())
    
    def update(self, pred_logits, target):
        pred = pred_logits.argmax(dim=1).cpu().numpy()
        gt = target.cpu().numpy()
        for b in range(pred.shape[0]):
            p, g = pred[b], gt[b]
            self.dice_scores['ET'].append(self.dice_coefficient(p == 3, g == 3))
            self.dice_scores['TC'].append(self.dice_coefficient((p == 1) | (p == 3), (g == 1) | (g == 3)))
            self.dice_scores['WT'].append(self.dice_coefficient(p >= 1, g >= 1))
            for c in range(4):
                self.per_class_dice[c].append(self.dice_coefficient(p == c, g == c))
    
    def compute(self):
        results = {}
        for region in ['ET', 'TC', 'WT']:
            scores = self.dice_scores[region]
            results[f'{region}_dice_mean'] = np.mean(scores)
            results[f'{region}_dice_std'] = np.std(scores)
            results[f'{region}_dice_median'] = np.median(scores)
        results['mean_dice'] = np.mean([results[f'{r}_dice_mean'] for r in ['ET', 'TC', 'WT']])
        for c in range(4):
            name = {0: 'BG', 1: 'NCR', 2: 'ED', 3: 'ET'}[c]
            results[f'class_{name}_dice'] = np.mean(self.per_class_dice[c])
        return results
