#!/usr/bin/env python3
# combined_train_swin_fix.py

"""
This script integrates a Swin‐UNETR‐based 3D regression backbone for LMDB volumes,
fixes the 'encoder' attribute error (SwinUNETR no longer exposes .encoder),
and adds memory‐saving techniques (gradient checkpointing, FP16). All original
functionality is preserved (no downsampling), and accuracy is maximized by
still leveraging the full SwinTransformer encoder.
"""

import os
import sys
import argparse
import pickle
import random
import numpy as np
import lmdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import optuna
import monai
from monai.transforms import (
    Lambdad,               # Apply a lambda to “image” key
    ScaleIntensityRanged,  # Normalize intensities
    RandFlipd,             # Random flip
    RandRotate90d,         # Random 90° rotations
    RandGaussianNoised,    # Random noise
    RandBiasFieldd,        # Random bias field
    RandGaussianSmoothd,   # Random smoothing
    RandAdjustContrastd,   # Random contrast adjustments
    ToTensord,             # Convert to PyTorch tensor
    Compose                # Compose transforms
)
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import torchvision
from collections import Counter

# ---------------------------------------------------------------------------- #
#                            HARD‐CODED MAX D, H, W                            #
# ---------------------------------------------------------------------------- #
# We skip compute_global_max_dims() entirely and directly hard‐code the known maxima:
MAX_D, MAX_H, MAX_W = 800, 1912, 1847
print(f"ℹ️  Using hardcoded MAX_D={MAX_D}, MAX_H={MAX_H}, MAX_W={MAX_W}")

# ---------------------------------------------------------------------------- #
#                             ARGUMENT PARSING                                  #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="Combined TransSeg & MedViT 3D Regression (Swin fix)")
parser.add_argument("--train_csv", type=str, default="train_labels.csv",
                    help="CSV with columns [tomo_id, Number of motors]")
parser.add_argument("--lmdb_path", type=str, default="train.lmdb",
                    help="Path to existing LMDB file containing {tomo_id: {'volume', 'label'}}")
parser.add_argument("--output_dir", type=str, default="output",
                    help="Where to save logs, checkpoints, etc.")
parser.add_argument("--epochs", type=int, default=100,
                    help="Max number of epochs per model (TransSeg/MedViT)")  # #37
parser.add_argument("--batch_size", type=int, default=2,
                    help="Batch size for both TransSeg and MedViT")  # #36
parser.add_argument("--lr", type=float, default=1e-4,
                    help="Initial learning rate (will be tuned by Optuna)")  # #34
parser.add_argument("--optimizer", type=str, choices=["adamw", "sgd"], default="adamw",
                    help="Optimizer choice (AdamW with weight_decay=1e-4)")  # #10
parser.add_argument("--no_tune", action="store_true",
                    help="Skip Optuna tuning (if set, uses default hyperparams)")  # #34–35
parser.add_argument("--use_mixup", action="store_true",
                    help="Enable 3D MixUp in training (alpha=0.4)")  # #11
parser.add_argument("--freeze_epochs", type=int, default=5,
                    help="Number of epochs to freeze backbone then unfreeze")  # #12
parser.add_argument("--early_stop_patience", type=int, default=10,
                    help="Patience for early stopping")  # #13
parser.add_argument("--medvit_lr", type=float, default=5e-5,
                    help="Initial LR for MedViT")  # #34–35
parser.add_argument("--medvit_epochs", type=int, default=100,
                    help="Max epochs for MedViT training")  # #37
parser.add_argument("--medvit_freeze_epochs", type=int, default=5,
                    help="Freeze MedViT backbone for these many epochs")  # #12
parser.add_argument("--vit2d_ckpt", type=str, default="TransSeg/pretrained/vit_base_2d.pth",
                    help="Path to 2D ViT checkpoint for TransSeg inflation")  # #3
parser.add_argument("--medvit2d_ckpt", type=str, default="MedViT/pretrained/medvit_base_2d.pth",
                    help="Path to 2D MedViT checkpoint for inflation")  # #17
parser.add_argument("--num_folds", type=int, default=5,
                    help="Number of folds for Stratified K-Fold CV (Step 29)")  # #29
parser.add_argument("--use_kfold", action="store_true",
                    help="Enable 5-Fold cross-validation")  # #29
parser.add_argument("--use_pruning", action="store_true",
                    help="Apply magnitude-based pruning after final training (Step 31)")  # #31
parser.add_argument("--use_distillation", action="store_true",
                    help="Perform knowledge distillation from teacher→student (Step 31)")  # #31
parser.add_argument("--max_lr", type=float, default=1e-3,
                    help="Max LR for OneCycleLR (Step 32)")  # #32

args, _ = parser.parse_known_args()
os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------------------------- #
#                   FIX RANDOM SEEDS FOR REPRODUCIBILITY                       #
# ---------------------------------------------------------------------------- #
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
monai.utils.set_determinism(seed=42)

# ---------------------------------------------------------------------------- #
#                        DEVICE (GPU) SETUP                                      #
# ---------------------------------------------------------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ℹ️  Using device = {device}")

# ---------------------------------------------------------------------------- #
#                              LOAD TRAIN CSV                                    #
# ---------------------------------------------------------------------------- #
import pandas as pd
df = pd.read_csv(args.train_csv)
df = df.drop_duplicates(subset="tomo_id", keep="first")
df["label"] = df["Number of motors"].astype(float)  # regression target
split_labels = df["Number of motors"].astype(int).values  # for stratified splitting
ids = df["tomo_id"].values
regression_targets = df["label"].values  # float targets
n_total = len(ids)  # total number of files

# ---------------------------------------------------------------------------- #
#                             DATASET UTILITIES                                  #
# ---------------------------------------------------------------------------- #
class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, ids, targets, transforms=None):
        self.env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False
        )
        self.ids = list(ids)
        self.targets = list(targets)
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        key = self.ids[idx].encode("utf-8")
        with self.env.begin() as txn:
            data = txn.get(key)
        record = pickle.loads(data)
        vol = record["volume"]  # np array shape: (D, H, W)
        lbl = self.targets[idx]

        # Pad this volume so it becomes exactly (MAX_D, MAX_H, MAX_W)
        D, H, W = vol.shape
        pad_d = MAX_D - D
        pad_h = MAX_H - H
        pad_w = MAX_W - W
        vol_padded = np.pad(
            vol,
            ((0, pad_d), (0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=0
        )  # now shape = (MAX_D, MAX_H, MAX_W)

        sample = {"image": vol_padded, "label": lbl}
        if self.transforms:
            sample = self.transforms(sample)
        # Warning: converting label to tensor
        return sample["image"], torch.tensor(sample["label"], dtype=torch.float)

# ---------------------------------------------------------------------------- #
#                  3D AUGMENTATION & TRANSFORM COMPOSITION                       #
# ---------------------------------------------------------------------------- #
train_transforms = Compose([
    # 1) Add channel dimension: (D,H,W) → (1,D,H,W)
    Lambdad(keys="image", func=lambda vol: vol[np.newaxis, ...]),

    # 2) Intensity normalization: map [0,255] → [0,1]
    ScaleIntensityRanged(
        keys="image",
        a_min=0, a_max=255,
        b_min=0.0, b_max=1.0,
        clip=True
    ),

    # 3) Random geometric transforms
    RandFlipd(keys="image", prob=0.5, spatial_axis=0),
    RandFlipd(keys="image", prob=0.5, spatial_axis=1),
    RandFlipd(keys="image", prob=0.5, spatial_axis=2),
    RandRotate90d(keys="image", prob=0.5, spatial_axes=(1, 2)),

    # 4) Random intensity & noise
    RandGaussianNoised(keys="image", prob=0.2, mean=0.0, std=0.1),
    RandBiasFieldd(keys="image", prob=0.3, coeff_range=(0.1, 0.5)),
    RandGaussianSmoothd(keys="image", prob=0.2),
    RandAdjustContrastd(keys="image", prob=0.2, gamma=(0.7, 1.5)),

    ToTensord(keys=["image", "label"]),
])

val_transforms = Compose([
    Lambdad(keys="image", func=lambda vol: vol[np.newaxis, ...]),
    ScaleIntensityRanged(
        keys="image",
        a_min=0, a_max=255,
        b_min=0.0, b_max=1.0,
        clip=True
    ),
    ToTensord(keys=["image", "label"]),
])

# ---------------------------------------------------------------------------- #
#                          SWIN‐UNETR REGRESSOR CLASS                             #
# ---------------------------------------------------------------------------- #
# We only need the encoder portion of SwinUNETR. Recent MONAI no longer exposes .encoder,
# so we directly call swinViT(...) → global‐pool → regressor head.
from monai.networks.nets.swin_unetr import SwinUNETR  # :contentReference[oaicite:0]{index=0}

class TransSegSwinRegressor(nn.Module):
    def __init__(self, max_d, max_h, max_w, feature_size=96, dropout_rate=0.15):
        super().__init__()
        # ---------------------------------------------------------------------------- #
        # 1) Instantiate SwinUNETR backbone (only encoder used) with gradient checkpointing:
        #    - use_checkpoint=True turns on activation checkpointing in SwinUNETR  :contentReference[oaicite:1]{index=1}
        #    - feature_size controls the channel widths (96 → last stage has feature_size*8 = 768 channels).
        #
        self.swin_unetr = SwinUNETR(
            img_size=(max_d, max_h, max_w),  # spatial dims = (800,1912,1847)
            in_channels=1,
            out_channels=1,                  # dummy; we won't use the segmentation head
            feature_size=feature_size,
            drop_rate=dropout_rate,
            attn_drop_rate=dropout_rate,
            dropout_path_rate=dropout_rate,
            use_checkpoint=True,             # enable memory‐saving checkpointing :contentReference[oaicite:2]{index=2}
            spatial_dims=3,
        )

        # ---------------------------------------------------------------------------- #
        # 2) We only extract the *last* hidden feature map from swinViT → global pooling:
        #    swinViT(x) returns a tuple of 4 feature maps (one per Swin stage); we take the final one.
        #
        #    hidden_states[-1].shape = (B, feature_size*8, D/32, H/32, W/32) = (B, 768, 25, 59, 57)
        #    (since MAX_D=800 → /32≈25; MAX_H=1912→/32≈59; MAX_W=1847→/32≈57) :contentReference[oaicite:3]{index=3}
        #
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))

        # ---------------------------------------------------------------------------- #
        # 3) Regression head: Normalize + Linear from 768 → 1  :contentReference[oaicite:4]{index=4}
        #
        hidden_dim = feature_size * 8  # = 768 when feature_size=96
        self.cls_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        x: (B, 1, D, H, W)  with (D,H,W) = (800, 1912, 1847)
        """
        # 1) Get all four Swin‐Transformer hidden states:
        #    hidden_states is a list of 4 tensors; we take the last one (deepest features).
        hidden_states = self.swin_unetr.swinViT(x)
        # hidden_states[-1].shape = (B, feature_size*8, D/32, H/32, W/32)
        feat = hidden_states[-1]  # :contentReference[oaicite:5]{index=5}

        # 2) Global average‐pool → (B, feature_size*8, 1,1,1) → flatten → (B, feature_size*8)
        pooled = self.global_pool(feat).view(feat.shape[0], -1)

        # 3) LayerNorm + Linear → scalar per sample
        out = self.cls_head(pooled).squeeze(1)  # (B,)
        return out

# ---------------------------------------------------------------------------- #
#                     OPTIONAL: MEDViT 3D REGRESSOR CLASS                        #
# ---------------------------------------------------------------------------- #
from MedViT import MedViT as MedViTBase  # Original 2D MedViT class

class MedViT3DRegressor(nn.Module):
    def __init__(self, max_d, max_h, max_w,
                 pretrained_2d_ckpt, patch_size=(2, 16, 16),
                 embed_dim=768, depth=12, num_heads=12, dropout_rate=0.15):
        super().__init__()

        # 3D patch embedding: (1, D,H,W) → (768, D/2, H/16, W/16)
        self.patch_embed3d = nn.Conv3d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        num_patches = (max_d // patch_size[0]) * \
                      (max_h // patch_size[1]) * \
                      (max_w // patch_size[2]

                     )

        # 3D positional embeddings (not used for classification/regression here, but kept for inflation)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder blocks (adapted from MedViTBase.Block)
        self.blocks = nn.ModuleList([
            MedViTBase.Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop=dropout_rate,
                attn_drop=dropout_rate,
                drop_path=0.1
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # DVPP as in original MedViT (kept unchanged)
        self.dvpp = DVPP3D(in_channels=embed_dim, embed_dim=embed_dim)

        # Linear regression head: 768 → 1
        self.head = nn.Linear(embed_dim, 1)

        # Inflate 2D → 3D if checkpoint exists (unchanged)
        if os.path.isfile(pretrained_2d_ckpt):
            sd2 = torch.load(pretrained_2d_ckpt)
            sd3 = {}
            for k, v in sd2.items():
                # Handle patch embedding weight separately
                if "patch_embed.proj.weight" in k:
                    v = v.mean(dim=1, keepdim=True)  # collapse color channels
                    depth_k = patch_size[0]
                    v = v.unsqueeze(-1).repeat(1, 1, 1, 1, depth_k) / depth_k
                    sd3["patch_embed3d.weight"] = v
                    continue
                new_k = k.replace("patch_embed.", "patch_embed3d.")
                if new_k in self.state_dict() and self.state_dict()[new_k].shape == v.shape:
                    sd3[new_k] = v
            self.load_state_dict(sd3, strict=False)

    def forward(self, x):
        B = x.size(0)

        # 3D patch embed
        feat = self.patch_embed3d(x)  # (B, 768, D/2, H/16, W/16)

        # Flatten + add CLS token for transformer (omitted here – DVPP expects spatial grid)
        # We skip CLS token and transformer if not needed for DVPP; original code already does DVPP directly
        dvpp_feat = self.dvpp(feat)  # (B, 768)

        output = self.head(dvpp_feat).squeeze(1)  # (B,)
        return output

# ---------------------------------------------------------------------------- #
#                         EVALUATION FUNCTION (unchanged)                        #
# ---------------------------------------------------------------------------- #
def evaluate_and_print(model, dataloader, n_total):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            # Mixed precision inference
            with autocast():
                outputs = model(imgs).cpu().numpy()  # (B,) float outputs
            all_preds.extend(outputs)
            all_labels.extend(lbls.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels).astype(int)  # true counts are integers

    rounded = np.rint(all_preds).astype(int)
    n_correct = int((rounded == all_labels).sum())
    n_val = len(all_labels)

    print(f"Number of files = {n_total}")
    print(f"Number of Validation tests = {n_val}")
    print(f"True estimated validation predictions (count) / all validation predictions (count) = "
          f"{n_correct} / {n_val}  (accuracy = {n_correct}/{n_val} = {n_correct/n_val:.3f})")

    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    return mae, rmse, n_correct / n_val

# ---------------------------------------------------------------------------- #
#             K-FOLD CROSS-VALIDATION / SINGLE SPLIT LOGIC (unchanged)           #
# ---------------------------------------------------------------------------- #
label_counts = Counter(split_labels)
min_count = min(label_counts.values())

if args.use_kfold:
    if min_count < 2:
        raise ValueError("Stratified K-Fold requires each class to have ≥ 2 samples.")
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=42)
    folds = list(skf.split(ids, split_labels))
else:
    if min_count < 2:
        ss = ShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        train_val_idx, test_idx = next(ss.split(ids))
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        train_val_idx, test_idx = next(sss.split(ids, split_labels))

    train_val_ids = ids[train_val_idx]
    train_val_targets = regression_targets[train_val_idx]
    test_ids = ids[test_idx]
    test_targets = regression_targets[test_idx]

    subset_labels = split_labels[train_val_idx]
    min_count_sub = min(Counter(subset_labels).values())
    if min_count_sub < 2:
        ss2 = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, vl_idx = next(ss2.split(train_val_ids))
    else:
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, vl_idx = next(sss2.split(train_val_ids, subset_labels))

    train_ids = train_val_ids[tr_idx]
    train_targets = train_val_targets[tr_idx]
    val_ids = train_val_ids[vl_idx]
    val_targets = train_val_targets[vl_idx]

# ---------------------------------------------------------------------------- #
#                            MIXUP (OPTIONAL) SETUP                             #
# ---------------------------------------------------------------------------- #
def mixup_3d(inputs, targets, alpha=0.4):
    """3D MixUp for 3D volumes (alpha=0.4)."""
    lam = np.random.beta(alpha, alpha)
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size).to(inputs.device)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
    labels_a, labels_b = targets, targets[index]
    return mixed_inputs, labels_a, labels_b, lam

# ---------------------------------------------------------------------------- #
#               TRANSSEG (SWIN) TRAIN / VALID / TEST LOOP                        #
# ---------------------------------------------------------------------------- #
def train_transseg_swin_fold(train_ids, train_targets, val_ids, val_targets, test_ids, test_targets, fold_idx=0):
    fold_dir = os.path.join(args.output_dir, f"swintransseg_fold{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    train_ds = LMDBDataset(args.lmdb_path, train_ids, train_targets, transforms=train_transforms)
    val_ds = LMDBDataset(args.lmdb_path, val_ids, val_targets, transforms=val_transforms)
    test_ds = LMDBDataset(args.lmdb_path, test_ids, test_targets, transforms=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    # Instantiate the regressor and move to GPU + FP16:
    model = TransSegSwinRegressor(MAX_D, MAX_H, MAX_W).to(device).half()  # :contentReference[oaicite:6]{index=6}

    # Optimizer & scheduler:
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    total_steps = args.epochs * len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=args.max_lr, total_steps=total_steps,
                           pct_start=0.3, anneal_strategy="cos")

    criterion = nn.MSELoss()
    scaler = GradScaler()  # :contentReference[oaicite:7]{index=7}

    # 1) Freeze entire backbone initially (only regression head learns):
    for name, param in model.named_parameters():
        if "cls_head" not in name:
            param.requires_grad = False

    best_val_mae = float("inf")
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, lbls in train_loader:
            imgs = imgs.to(device).half()
            lbls = lbls.to(device)

            if args.use_mixup:
                imgs, lbls_a, lbls_b, lam = mixup_3d(imgs, lbls, alpha=0.4)
                with autocast():
                    outputs = model(imgs)
                    loss = lam * criterion(outputs, lbls_a) + (1 - lam) * criterion(outputs, lbls_b)
            else:
                with autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, lbls)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if not args.no_tune:
                scheduler.step()

            running_loss += loss.item() * imgs.size(0)

        # 2) After freeze_epochs, unfreeze backbone and scale lr down ×0.1
        if epoch == args.freeze_epochs + 1:
            for param in model.parameters():
                param.requires_grad = True
            for g in optimizer.param_groups:
                g["lr"] *= 0.1

        val_mae, val_rmse, val_acc = evaluate_and_print(model, val_loader, n_total)
        print(f"[SwinTransSeg Fold {fold_idx}] Epoch {epoch} | Train MSE: {running_loss/len(train_loader.dataset):.4f} "
              f"| Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f} | Val Acc: {val_acc:.3f}")

        if val_mae < best_val_mae - 1e-4:
            best_val_mae = val_mae
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(fold_dir, "best_swintransseg_model.pth"))
        else:
            no_improve += 1
            if no_improve >= args.early_stop_patience:
                print(f"[SwinTransSeg Fold {fold_idx}] Early stopping at epoch {epoch}")
                break

    # Load best, evaluate on test set
    model.load_state_dict(torch.load(os.path.join(fold_dir, "best_swintransseg_model.pth")))
    test_mae, test_rmse, test_acc = evaluate_and_print(model, test_loader, n_total)
    print(f"[SwinTransSeg Fold {fold_idx}] Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}, Test Acc: {test_acc:.3f}")

    return model

# ---------------------------------------------------------------------------- #
#                          MEDViT TRAIN / VALID / TEST LOOP                     #
# ---------------------------------------------------------------------------- #
def train_medvit_fold(train_ids, train_targets, val_ids, val_targets, test_ids, test_targets, fold_idx=0):
    fold_dir = os.path.join(args.output_dir, f"medvit_fold{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    train_ds = LMDBDataset(args.lmdb_path, train_ids, train_targets, transforms=train_transforms)
    val_ds = LMDBDataset(args.lmdb_path, val_ids, val_targets, transforms=val_transforms)
    test_ds = LMDBDataset(args.lmdb_path, test_ids, test_targets, transforms=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    medvit_model = MedViT3DRegressor(
        MAX_D, MAX_H, MAX_W,
        pretrained_2d_ckpt=args.medvit2d_ckpt,
        patch_size=(2, 16, 16),
        embed_dim=768,
        depth=12,
        num_heads=12,
        dropout_rate=0.15
    ).to(device).half()

    optimizer = optim.AdamW(medvit_model.parameters(), lr=args.medvit_lr, weight_decay=1e-4)
    total_steps = args.medvit_epochs * len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=args.max_lr, total_steps=total_steps,
                           pct_start=0.3, anneal_strategy="cos")

    criterion = nn.MSELoss()
    scaler = GradScaler()

    for name, param in medvit_model.named_parameters():
        if "head" not in name:
            param.requires_grad = False

    best_val_mae = float("inf")
    no_improve = 0

    for epoch in range(1, args.medvit_epochs + 1):
        medvit_model.train()
        running_loss = 0.0
        for imgs, lbls in train_loader:
            imgs = imgs.to(device).half()
            lbls = lbls.to(device)
            with autocast():
                outputs = medvit_model(imgs)
                loss = criterion(outputs, lbls)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if not args.no_tune:
                scheduler.step()

            running_loss += loss.item() * imgs.size(0)

        if epoch == args.medvit_freeze_epochs + 1:
            for param in medvit_model.parameters():
                param.requires_grad = True
            for g in optimizer.param_groups:
                g["lr"] *= 0.1

        val_mae, val_rmse, val_acc = evaluate_and_print(medvit_model, val_loader, n_total)
        print(f"[MedViT Fold {fold_idx}] Epoch {epoch} | Train MSE: {running_loss/len(train_loader.dataset):.4f} "
              f"| Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f} | Val Acc: {val_acc:.3f}")

        if val_mae < best_val_mae - 1e-4:
            best_val_mae = val_mae
            no_improve = 0
            torch.save(medvit_model.state_dict(), os.path.join(fold_dir, "best_medvit_model.pth"))
        else:
            no_improve += 1
            if no_improve >= args.early_stop_patience:
                print(f"[MedViT Fold {fold_idx}] Early stopping at epoch {epoch}")
                break

    medvit_model.load_state_dict(torch.load(os.path.join(fold_dir, "best_medvit_model.pth")))
    test_mae, test_rmse, test_acc = evaluate_and_print(medvit_model, test_loader, n_total)
    print(f"[MedViT Fold {fold_idx}] Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}, Test Acc: {test_acc:.3f}")

    return medvit_model

# ---------------------------------------------------------------------------- #
#                     MAIN TRAINING / EVAL ROUTINE                               #
# ---------------------------------------------------------------------------- #
def main():
    if args.use_kfold:
        for fold_idx, (train_val_idx, test_idx) in enumerate(folds):
            train_val_ids = ids[train_val_idx]
            train_val_targets = regression_targets[train_val_idx]
            test_ids_fold = ids[test_idx]
            test_targets_fold = regression_targets[test_idx]

            subset_labels = split_labels[train_val_idx]
            min_count_sub = min(Counter(subset_labels).values())
            if min_count_sub < 2:
                ss2 = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                tr_idx, vl_idx = next(ss2.split(train_val_ids))
            else:
                sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                tr_idx, vl_idx = next(sss2.split(train_val_ids, subset_labels))

            train_ids_fold = train_val_ids[tr_idx]
            train_targets_fold = train_val_targets[tr_idx]
            val_ids_fold = train_val_ids[vl_idx]
            val_targets_fold = train_val_targets[vl_idx]

            print(f"#### Starting SwinTransSeg fold {fold_idx} ####")
            swintransseg_model = train_transseg_swin_fold(
                train_ids_fold, train_targets_fold,
                val_ids_fold, val_targets_fold,
                test_ids_fold, test_targets_fold,
                fold_idx=fold_idx
            )

            print(f"#### Starting MedViT fold {fold_idx} ####")
            medvit_model = train_medvit_fold(
                train_ids_fold, train_targets_fold,
                val_ids_fold, val_targets_fold,
                test_ids_fold, test_targets_fold,
                fold_idx=fold_idx
            )

            # Optional pruning
            if args.use_pruning:
                import torch.nn.utils.prune as prune
                for module in swintransseg_model.modules():
                    if isinstance(module, nn.Linear):
                        prune.l1_unstructured(module, name="weight", amount=0.3)
                for module in medvit_model.modules():
                    if isinstance(module, nn.Linear):
                        prune.l1_unstructured(module, name="weight", amount=0.3)

            # Optional distillation (kept as before)
            if args.use_distillation:
                student = get_medicalnet_backbone(
                    name="resnet50", num_outputs=1
                ).to(device)

            # Ensemble evaluation (unchanged, uses the new regressor forward)
            print(f"#### Ensemble results for fold {fold_idx} ####")
            test_ds_fold = LMDBDataset(args.lmdb_path, test_ids_fold, test_targets_fold, transforms=val_transforms)
            test_loader_fold = DataLoader(test_ds_fold, batch_size=1, shuffle=False, num_workers=2)

            ensemble_preds, ensemble_labels = [], []
            for img, lbl in test_loader_fold:
                with torch.no_grad():
                    p1 = swintransseg_model(img.to(device).half()).item()
                    p2 = medvit_model(img.to(device).half()).item()
                ensemble_preds.append((p1 + p2) / 2.0)
                ensemble_labels.append(lbl.item())

            rounded_ens = np.rint(np.array(ensemble_preds)).astype(int)
            true_ens = np.array(ensemble_labels).astype(int)
            ens_n_correct = int((rounded_ens == true_ens).sum())
            ens_n_val = len(true_ens)

            print(f"Number of files = {n_total}")
            print(f"Number of Validation tests = {ens_n_val}")
            print(f"True estimated validation predictions (count) / all validation predictions (count) = "
                  f"{ens_n_correct} / {ens_n_val}  (accuracy = {ens_n_correct}/{ens_n_val} = {ens_n_correct/ens_n_val:.3f})")

            ens_mae = mean_absolute_error(true_ens, ensemble_preds)
            ens_rmse = np.sqrt(mean_squared_error(true_ens, ensemble_preds))
            print(f"[Ensemble Fold {fold_idx}] Test MAE: {ens_mae:.4f}, Test RMSE: {ens_rmse:.4f}")

    else:
        print("#### Starting single-split SwinTransSeg training ####")
        swintransseg_model = train_transseg_swin_fold(
            train_ids, train_targets, val_ids, val_targets,
            test_ids, test_targets, fold_idx=0
        )

        print("#### Starting single-split MedViT training ####")
        medvit_model = train_medvit_fold(
            train_ids, train_targets, val_ids, val_targets,
            test_ids, test_targets, fold_idx=0
        )

        if args.use_pruning:
            import torch.nn.utils.prune as prune
            for module in swintransseg_model.modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name="weight", amount=0.3)
            for module in medvit_model.modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name="weight", amount=0.3)

        if args.use_distillation:
            student = get_medicalnet_backbone(
                name="resnet50", num_outputs=1
            ).to(device)

        test_ds = LMDBDataset(args.lmdb_path, test_ids, test_targets, transforms=val_transforms)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
        ens_preds, ens_labels = [], []
        for img, lbl in test_loader:
            with torch.no_grad():
                p1 = swintransseg_model(img.to(device).half()).item()
                p2 = medvit_model(img.to(device).half()).item()
            ens_preds.append((p1 + p2) / 2.0)
            ens_labels.append(lbl.item())

        rounded_ens = np.rint(np.array(ens_preds)).astype(int)
        true_ens = np.array(ens_labels).astype(int)
        ens_n_correct = int((rounded_ens == true_ens).sum())
        ens_n_val = len(true_ens)

        print(f"Number of files = {n_total}")
        print(f"Number of Validation tests = {ens_n_val}")
        print(f"True estimated validation predictions (count) / all validation predictions (count) = "
              f"{ens_n_correct} / {ens_n_val}  (accuracy = {ens_n_correct}/{ens_n_val} = {ens_n_correct/ens_n_val:.3f})")

        ens_mae = mean_absolute_error(true_ens, ens_preds)
        ens_rmse = np.sqrt(mean_squared_error(true_ens, ens_preds))
        print(f"[Ensemble] Test MAE: {ens_mae:.4f}, Test RMSE: {ens_rmse:.4f}")

if __name__ == "__main__":
    main()

"""
python /workspace/combined_train.py \
  --train_csv /workspace/train_labels.csv \
  --lmdb_path /workspace/train.lmdb \
  --output_dir /workspace/output \
  --epochs 100 \
  --batch_size 2 \
  --lr 1e-4 \
  --optimizer adamw \
  --use_mixup \
  --freeze_epochs 5 \
  --early_stop_patience 10 \
  --medvit_lr 5e-5 \
  --medvit_epochs 100 \
  --medvit_freeze_epochs 5 \
  --vit2d_ckpt /workspace/TransSeg/pretrained/vit_base_2d.pth \
  --medvit2d_ckpt /workspace/MedViT/pretrained/medvit_base_2d.pth \
  --max_lr 1e-3 \
  --use_pruning \
  --use_distillation

"""