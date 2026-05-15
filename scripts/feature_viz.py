#!/usr/bin/env python3
"""
Feature space visualization: GLI vs MEN in SAM-Med3D encoder
t-SNE + UMAP analysis at encoder and decoder levels
"""

import sys, os
PROJECT = '.'
os.chdir(PROJECT)
sys.path.insert(0, os.path.join(PROJECT, 'SAM-Med3D'))
sys.path.insert(0, PROJECT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# pip install umap-learn if needed
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("umap-learn not available, skipping UMAP")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

device = torch.device('cuda')
os.makedirs('results/viz', exist_ok=True)

# ── Load model ────────────────────────────────────────────

from models.proto_sam_v31 import ProtoSAM_v3_1

model = ProtoSAM_v3_1(
    sam_checkpoint=os.path.join(PROJECT, 'SAM-Med3D/ckpt/sam_med3d_turbo.pth'),
    num_classes=4, lora_r=32, lora_alpha=64, in_channels=4, unfreeze_blocks=2,
).to(device)
ckpt = torch.load(os.path.join(PROJECT, 'checkpoints_v31/best.pth'), map_location=device)
md = model.state_dict()
md.update({k: v for k, v in ckpt['model_state_dict'].items() if k in md})
model.load_state_dict(md)
model.eval()
print(f"v3.1 loaded: {ckpt['best_dice']:.4f}")

# ── Load data ─────────────────────────────────────────────

GLI_DIR = os.path.join(PROJECT, 'data/processed_BraTS2021')
MEN_DIR = os.path.join(PROJECT, 'data/processed_MEN')

with open('data/splits/men_split.json') as f:
    men_splits = json.load(f)

# 找 GLI cases
gli_cases = sorted([f.replace('_image.npy', '') for f in os.listdir(GLI_DIR) 
                    if f.endswith('_image.npy')])
men_cases = men_splits['test']

np.random.seed(42)
gli_sample = np.random.choice(gli_cases, size=min(30, len(gli_cases)), replace=False).tolist()
men_sample = np.random.choice(men_cases, size=min(30, len(men_cases)), replace=False).tolist()

print(f"GLI samples: {len(gli_sample)}, MEN samples: {len(men_sample)}")

# ── Extract features ──────────────────────────────────────

def extract_features(model, data_dir, case_list, max_voxels_per_case=200):
    """
    Extract per-voxel features at encoder level (384d, 8³) and decoder level (64d, 128³)
    Subsample voxels for visualization
    """
    enc_features = []  # (N, 384)
    dec_features = []  # (N, 64)
    labels = []        # (N,) — 0=BG, 1=NCR, 2=ED, 3=ET
    domains = []       # (N,) — 0=GLI, 1=MEN
    case_ids = []      # (N,)
    
    domain_label = 0 if 'GLI' in case_list[0] or 'BraTS-GLI' in case_list[0] else 1
    # Auto-detect domain from filename
    
    for case in case_list:
        img = np.load(os.path.join(data_dir, f'{case}_image.npy'))
        lbl = np.load(os.path.join(data_dir, f'{case}_label.npy'))
        
        img_t = torch.from_numpy(img).float().unsqueeze(0).to(device)  # (1, 4, 128³)
        
        with torch.no_grad():
            # Encoder features
            enc_feat = model.image_encoder(img_t)
            if enc_feat.shape[1] != 384:
                enc_feat = enc_feat.permute(0, 4, 1, 2, 3)
            # enc_feat: (1, 384, 8, 8, 8)
            
            # Decoder features (before classification head)
            x = model.decoder.up1(enc_feat)
            x = model.decoder.up2(x)
            x = model.decoder.up3(x)
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
            x = model.decoder.head[0](x)
            x = model.decoder.head[1](x)
            dec_feat = model.decoder.head[2](x)
            # dec_feat: (1, 64, 128, 128, 128)
        
        # ── Encoder-level sampling (8³ = 512 voxels) ──
        enc_np = enc_feat.squeeze(0).cpu().numpy()  # (384, 8, 8, 8)
        lbl_down = torch.from_numpy(lbl).float().unsqueeze(0).unsqueeze(0)
        lbl_8 = F.interpolate(lbl_down, size=(8,8,8), mode='nearest').squeeze().numpy().astype(int)
        
        # Sample voxels per class (balanced)
        for c in range(4):
            coords = np.argwhere(lbl_8 == c)
            if len(coords) == 0:
                continue
            n_sample = min(max_voxels_per_case // 4, len(coords))
            if n_sample == 0:
                continue
            idx = np.random.choice(len(coords), n_sample, replace=(n_sample > len(coords)))
            for j in idx:
                d, h, w = coords[j]
                enc_features.append(enc_np[:, d, h, w])  # (384,)
                labels.append(c)
                domains.append(domain_label)
                case_ids.append(case)
        
        # ── Decoder-level sampling (128³, subsample heavily) ──
        dec_np = dec_feat.squeeze(0).cpu().numpy()  # (64, 128, 128, 128)
        
        for c in range(4):
            coords = np.argwhere(lbl == c)
            if len(coords) == 0:
                continue
            n_sample = min(max_voxels_per_case // 4, len(coords))
            if n_sample == 0:
                continue
            idx = np.random.choice(len(coords), n_sample, replace=(n_sample > len(coords)))
            for j in idx:
                d, h, w = coords[j]
                dec_features.append(dec_np[:, d, h, w])  # (64,)
        
        del img_t, enc_feat, dec_feat, x
        torch.cuda.empty_cache()
    
    return {
        'enc': np.array(enc_features),
        'dec': np.array(dec_features),
        'labels': np.array(labels),
        'domains': np.array(domains),
        'cases': case_ids,
    }


print("\nExtracting GLI features...")
gli_data = extract_features(model, GLI_DIR, gli_sample, max_voxels_per_case=200)
gli_data['domains'][:] = 0
print(f"  Encoder: {gli_data['enc'].shape}, Decoder: {gli_data['dec'].shape}")

print("Extracting MEN features...")
men_data = extract_features(model, MEN_DIR, men_sample, max_voxels_per_case=200)
men_data['domains'][:] = 1
print(f"  Encoder: {men_data['enc'].shape}, Decoder: {men_data['dec'].shape}")

# ── Combine ───────────────────────────────────────────────

enc_all = np.concatenate([gli_data['enc'], men_data['enc']], axis=0)
dec_all = np.concatenate([gli_data['dec'], men_data['dec']], axis=0)
labels_all = np.concatenate([gli_data['labels'], men_data['labels']], axis=0)
domains_all = np.concatenate([gli_data['domains'], men_data['domains']], axis=0)

print(f"\nTotal encoder voxels: {enc_all.shape[0]}")
print(f"Total decoder voxels: {dec_all.shape[0]}")
print(f"Labels: {np.unique(labels_all, return_counts=True)}")
print(f"Domains: GLI={np.sum(domains_all==0)}, MEN={np.sum(domains_all==1)}")

# ── t-SNE ─────────────────────────────────────────────────

print("\nRunning PCA (384d → 50d)...")
from sklearn.preprocessing import StandardScaler

scaler_enc = StandardScaler().fit(enc_all)
enc_scaled = scaler_enc.transform(enc_all)

pca_enc = PCA(n_components=50).fit_transform(enc_scaled)
print(f"  PCA explained variance: {PCA(n_components=50).fit(enc_scaled).explained_variance_ratio_.sum():.3f}")

print("Running t-SNE on encoder features...")
tsne_enc = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42).fit_transform(pca_enc)

if dec_all.shape[0] > 0:
    scaler_dec = StandardScaler().fit(dec_all)
    dec_scaled = scaler_dec.transform(dec_all)
    print("Running t-SNE on decoder features...")
    tsne_dec = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42).fit_transform(dec_scaled)

# ── UMAP ──────────────────────────────────────────────────

if HAS_UMAP:
    print("Running UMAP on encoder features...")
    umap_enc = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, 
                          random_state=42).fit_transform(enc_scaled)
    if dec_all.shape[0] > 0:
        print("Running UMAP on decoder features...")
        umap_dec = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, 
                              random_state=42).fit_transform(dec_scaled)

# ── Plotting ──────────────────────────────────────────────

CLASS_NAMES = ['BG', 'NCR', 'ED', 'ET']
CLASS_COLORS = ['#808080', '#e74c3c', '#2ecc71', '#3498db']
DOMAIN_NAMES = ['GLI', 'MEN']
DOMAIN_MARKERS = ['o', '^']

def plot_by_class_and_domain(coords, labels, domains, title, save_path):
    """2x2 grid: (by class, by domain, class×domain overlay, domain-only fg)"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # (0,0): Color by class
    ax = axes[0, 0]
    for c in range(4):
        mask = labels == c
        ax.scatter(coords[mask, 0], coords[mask, 1], c=CLASS_COLORS[c],
                   s=8, alpha=0.5, label=CLASS_NAMES[c])
    ax.legend(fontsize=10, markerscale=3)
    ax.set_title('By Tissue Class', fontsize=14)
    ax.set_xticks([]); ax.set_yticks([])
    
    # (0,1): Color by domain
    ax = axes[0, 1]
    domain_colors = ['#e74c3c', '#3498db']
    for d in range(2):
        mask = domains == d
        ax.scatter(coords[mask, 0], coords[mask, 1], c=domain_colors[d],
                   s=8, alpha=0.5, label=DOMAIN_NAMES[d])
    ax.legend(fontsize=10, markerscale=3)
    ax.set_title('By Domain (GLI vs MEN)', fontsize=14)
    ax.set_xticks([]); ax.set_yticks([])
    
    # (1,0): Class × Domain (shape=domain, color=class)
    ax = axes[1, 0]
    for d in range(2):
        for c in range(4):
            mask = (labels == c) & (domains == d)
            if mask.sum() == 0: continue
            ax.scatter(coords[mask, 0], coords[mask, 1], c=CLASS_COLORS[c],
                       marker=DOMAIN_MARKERS[d], s=12, alpha=0.4,
                       label=f'{DOMAIN_NAMES[d]}-{CLASS_NAMES[c]}')
    ax.legend(fontsize=7, markerscale=2, ncol=2, loc='best')
    ax.set_title('Class × Domain', fontsize=14)
    ax.set_xticks([]); ax.set_yticks([])
    
    # (1,1): Foreground only (label > 0), color by domain
    ax = axes[1, 1]
    fg_mask = labels > 0
    for d in range(2):
        mask = fg_mask & (domains == d)
        if mask.sum() == 0: continue
        ax.scatter(coords[mask, 0], coords[mask, 1], c=domain_colors[d],
                   s=12, alpha=0.5, label=f'{DOMAIN_NAMES[d]} tumor')
    ax.legend(fontsize=10, markerscale=3)
    ax.set_title('Tumor Voxels Only: GLI vs MEN', fontsize=14)
    ax.set_xticks([]); ax.set_yticks([])
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# Plot t-SNE encoder
plot_by_class_and_domain(
    tsne_enc, labels_all, domains_all,
    't-SNE: Encoder Features (384d, 8³)',
    'results/viz/tsne_encoder.png'
)

# Plot t-SNE decoder
if dec_all.shape[0] > 0:
    plot_by_class_and_domain(
        tsne_dec, labels_all, domains_all,
        't-SNE: Decoder Features (64d, 128³)',
        'results/viz/tsne_decoder.png'
    )

# Plot UMAP
if HAS_UMAP:
    plot_by_class_and_domain(
        umap_enc, labels_all, domains_all,
        'UMAP: Encoder Features (384d, 8³)',
        'results/viz/umap_encoder.png'
    )
    if dec_all.shape[0] > 0:
        plot_by_class_and_domain(
            umap_dec, labels_all, domains_all,
            'UMAP: Decoder Features (64d, 128³)',
            'results/viz/umap_decoder.png'
        )

# ── Quantitative domain separation metrics ────────────────

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

print("\n" + "="*60)
print("Quantitative Feature Analysis")
print("="*60)

# 1. Linear probe: can a linear classifier separate GLI vs MEN?
print("\n1. Domain classification (GLI vs MEN) — linear probe:")
for name, feat in [('encoder', enc_scaled), ('decoder', StandardScaler().fit_transform(dec_all) if dec_all.shape[0]>0 else None)]:
    if feat is None: continue
    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, feat, domains_all, cv=5, scoring='accuracy')
    print(f"  {name}: {scores.mean():.4f} ± {scores.std():.4f}")
    # 50% = domains indistinguishable (good for transfer)
    # 100% = completely separable (bad for transfer)

# 2. Per-class domain separability
print("\n2. Per-class domain separability:")
for c in range(4):
    mask = labels_all == c
    if mask.sum() < 20: continue
    feat_c = enc_scaled[mask]
    dom_c = domains_all[mask]
    if len(np.unique(dom_c)) < 2: continue
    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, feat_c, dom_c, cv=min(5, min(np.sum(dom_c==0), np.sum(dom_c==1))),
                              scoring='accuracy')
    print(f"  {CLASS_NAMES[c]}: {scores.mean():.4f} ± {scores.std():.4f} "
          f"(GLI={np.sum(dom_c==0)}, MEN={np.sum(dom_c==1)})")

# 3. Class separability within each domain
print("\n3. Tumor class classification — linear probe:")
for d, dname in [(0, 'GLI'), (1, 'MEN')]:
    mask = (domains_all == d) & (labels_all > 0)
    if mask.sum() < 20: continue
    feat_d = enc_scaled[mask]
    lbl_d = labels_all[mask]
    if len(np.unique(lbl_d)) < 2: continue
    clf = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    scores = cross_val_score(clf, feat_d, lbl_d, cv=5, scoring='accuracy')
    print(f"  {dname} tumor classes: {scores.mean():.4f} ± {scores.std():.4f}")

# 4. Cross-domain class separability (train on GLI, test on MEN)
print("\n4. Cross-domain transfer — train on GLI, test on MEN:")
gli_mask = (domains_all == 0) & (labels_all > 0)
men_mask = (domains_all == 1) & (labels_all > 0)
if gli_mask.sum() > 0 and men_mask.sum() > 0:
    clf = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    clf.fit(enc_scaled[gli_mask], labels_all[gli_mask])
    acc = clf.score(enc_scaled[men_mask], labels_all[men_mask])
    print(f"  Encoder: {acc:.4f}")
    
    if dec_all.shape[0] > 0:
        dec_scaled_all = StandardScaler().fit_transform(dec_all)
        clf2 = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
        clf2.fit(dec_scaled_all[gli_mask], labels_all[gli_mask])
        acc2 = clf2.score(dec_scaled_all[men_mask], labels_all[men_mask])
        print(f"  Decoder: {acc2:.4f}")

# 5. Inter-class and inter-domain distances
print("\n5. Feature distances (cosine similarity between class centroids):")
for level, feat in [('encoder', enc_scaled), ('decoder', StandardScaler().fit_transform(dec_all) if dec_all.shape[0]>0 else None)]:
    if feat is None: continue
    print(f"  [{level}]")
    centroids = {}
    for d in range(2):
        for c in range(4):
            mask = (domains_all == d) & (labels_all == c)
            if mask.sum() == 0: continue
            centroids[(d, c)] = feat[mask].mean(axis=0)
    
    # Same class, across domains
    print(f"  Same class, GLI vs MEN (higher = more transferable):")
    for c in range(4):
        if (0, c) in centroids and (1, c) in centroids:
            cos = np.dot(centroids[(0,c)], centroids[(1,c)]) / (
                np.linalg.norm(centroids[(0,c)]) * np.linalg.norm(centroids[(1,c)]) + 1e-8)
            print(f"    {CLASS_NAMES[c]}: cosine={cos:.4f}")

print("\nDone! Check results/viz/ for plots.")
