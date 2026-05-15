"""
BraTS2021 Dataset with strong augmentation for v2
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import rotate, gaussian_filter, map_coordinates


class BraTS2021Dataset(Dataset):
    
    def __init__(self, data_dir, case_names, augment=False, strong_augment=False):
        self.data_dir = data_dir
        self.case_names = case_names
        self.augment = augment
        self.strong_augment = strong_augment
    
    def __len__(self):
        return len(self.case_names)
    
    def __getitem__(self, idx):
        case = self.case_names[idx]
        image = np.load(os.path.join(self.data_dir, f'{case}_image.npy'))  # (4,128,128,128)
        label = np.load(os.path.join(self.data_dir, f'{case}_label.npy'))  # (128,128,128)
        
        if self.augment:
            image, label = self._augment(image, label)
        
        return {
            'image': torch.from_numpy(image.copy()).float(),
            'label': torch.from_numpy(label.copy()).long(),
            'case_name': case,
        }
    
    def _augment(self, image, label):
        # 1. Random flip (3 axes)
        for axis in [1, 2, 3]:  # spatial axes of (C,D,H,W)
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=axis).copy()
                label = np.flip(label, axis=axis - 1).copy()
        
        # 2. Intensity augmentation
        if np.random.rand() > 0.3:
            for c in range(4):
                shift = np.random.uniform(-0.1, 0.1)
                scale = np.random.uniform(0.9, 1.1)
                image[c] = image[c] * scale + shift
        
        if self.strong_augment:
            # 3. Random rotation (small angle, 1 random axis)
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-15, 15)
                axes_choices = [(0, 1), (0, 2), (1, 2)]
                rot_axes = axes_choices[np.random.randint(3)]
                for c in range(4):
                    image[c] = rotate(image[c], angle, axes=rot_axes, 
                                      reshape=False, order=1, mode='constant', cval=0)
                label = rotate(label.astype(np.float32), angle, axes=rot_axes,
                              reshape=False, order=0, mode='constant', cval=0).astype(np.int8)
            
            # 4. Gaussian noise
            if np.random.rand() > 0.5:
                noise_std = np.random.uniform(0.01, 0.05)
                noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
                image = image + noise
            
            # 5. Gaussian blur (random channel)
            if np.random.rand() > 0.7:
                sigma = np.random.uniform(0.5, 1.0)
                c = np.random.randint(4)
                image[c] = gaussian_filter(image[c], sigma=sigma)
            
            # 6. Random crop and pad back to 128 (simulate scale variation)
            if np.random.rand() > 0.5:
                crop_size = np.random.randint(96, 128)
                if crop_size < 128:
                    # Find tumor center for guided crop
                    tumor_mask = label > 0
                    if tumor_mask.sum() > 0:
                        coords = np.where(tumor_mask)
                        center = [int(np.median(c)) for c in coords]
                    else:
                        center = [64, 64, 64]
                    
                    # Compute crop start (centered on tumor)
                    starts = []
                    for dim in range(3):
                        s = max(0, min(center[dim] - crop_size // 2, 128 - crop_size))
                        starts.append(s)
                    
                    # Crop
                    d, h, w = starts
                    image_crop = image[:, d:d+crop_size, h:h+crop_size, w:w+crop_size]
                    label_crop = label[d:d+crop_size, h:h+crop_size, w:w+crop_size]
                    
                    # Resize back to 128
                    import torch.nn.functional as Fnn
                    img_t = torch.from_numpy(image_crop.copy()).float().unsqueeze(0)
                    image = Fnn.interpolate(img_t, size=128, mode='trilinear', 
                                           align_corners=False)[0].numpy()
                    lbl_t = torch.from_numpy(label_crop.copy().astype(np.float32)).unsqueeze(0).unsqueeze(0)
                    label = Fnn.interpolate(lbl_t, size=128, mode='nearest')[0, 0].numpy().astype(np.int8)
        
        return image, label


class FewShotSampler:
    
    def __init__(self, dataset, num_classes=4):
        self.dataset = dataset
        self.num_classes = num_classes
        self.complete_indices = []
        
        for idx in range(len(dataset)):
            case = dataset.case_names[idx]
            label = np.load(os.path.join(dataset.data_dir, f'{case}_label.npy'))
            unique = set(np.unique(label).tolist())
            if {1, 2, 3}.issubset(unique):
                fg_ratio = (label > 0).sum() / label.size
                self.complete_indices.append((idx, fg_ratio))
        
        self.complete_indices.sort(key=lambda x: x[1], reverse=True)
    
    def sample_support_set(self, K=5):
        indices = [idx for idx, _ in self.complete_indices[:K]]
        cases, images, labels = [], [], []
        for idx in indices:
            sample = self.dataset[idx]
            cases.append(sample['case_name'])
            images.append(sample['image'])
            labels.append(sample['label'])
        return cases, torch.stack(images), torch.stack(labels)


def get_dataloaders(project_dir, batch_size=2, num_workers=4, strong_augment=False):
    data_dir = os.path.join(project_dir, 'data/processed_BraTS2021')
    split_file = os.path.join(project_dir, 'data/splits/brats2021_split.json')
    
    with open(split_file) as f:
        splits = json.load(f)
    
    train_dataset = BraTS2021Dataset(data_dir, splits['train'], 
                                      augment=True, strong_augment=strong_augment)
    val_dataset = BraTS2021Dataset(data_dir, splits['val'], augment=False)
    test_dataset = BraTS2021Dataset(data_dir, splits['test'], augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader, splits
