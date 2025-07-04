#!/usr/bin/env python3
# combined_train.py
# This script integrates TransSeg‐inflated‐ViT and MedViT for 3D regression on LMDB volumes.
# Annotated with step numbers (1–40) for each new or modified section.

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
# ---------------------------------------------------------------------------- #
# REPLACED StratifiedShuffleSplit, StratifiedKFold (classification) 
# WITH ShuffleSplit, KFold (regression)                                      [CHANGED]
# ---------------------------------------------------------------------------- #
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import optuna
import monai
from monai.transforms import (
    Lambdad, ScaleIntensityRanged, RandFlipd,
    RandRotate90d, RandGaussianNoised,
    RandBiasFieldd, RandGaussianSmoothd,
    RandAdjustContrastd, ToTensord, Compose
)
from torchvision.ops import StochasticDepth  # Step 9: for stochastic depth
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import torchvision

# ---------------------------------------------------------------------------- #
#                              DVPP MODULE DEFINITION                            #
# ---------------------------------------------------------------------------- #
class DVPP3D(nn.Module):
    """
    # DVPP: Dual‐View Pyramid Pooling for 3D volumes (spatial + cross‐channel).
    Aggregates multi‐scale adaptive pools at [1,2,4] bins in 3D, then concatenates.
    This replaces a fixed cls_token+transformer head, enabling variable (D,H,W).
    """
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        # We will adaptively pool to (1,1,1), (2,2,2), (4,4,4) and then flatten & concat
        # Resulting DVPP vector size = in_channels * (1^3 + 2^3 + 4^3) = in_channels * (1 + 8 + 64) = in_channels * 73
        self.pool1 = nn.AdaptiveAvgPool3d((1, 1, 1))   # 1×1×1
        self.pool2 = nn.AdaptiveAvgPool3d((2, 2, 2))   # 2×2×2
        self.pool3 = nn.AdaptiveAvgPool3d((4, 4, 4))   # 4×4×4
        # Final linear layer → embed_dim (compressed back to transformer‐style hidden dim)
        self.fc = nn.Linear(in_channels * (1 + 8 + 64), embed_dim)
        # LayerNorm + final regression head will be defined in the calling class

    def forward(self, x):
        """
        x: (B, C, D', H', W')  # feature‐map from patch_embed or similar
        returns: (B, embed_dim) → a fixed‐length vector
        """
        B, C, Dp, Hp, Wp = x.shape
        # 1×1×1 pooling → shape (B, C, 1, 1, 1)
        p1 = self.pool1(x).view(B, C * 1 * 1 * 1)
        # 2×2×2 pooling → shape (B, C, 2, 2, 2)
        p2 = self.pool2(x).view(B, C * 2 * 2 * 2)
        # 4×4×4 pooling → shape (B, C, 4, 4, 4)
        p3 = self.pool3(x).view(B, C * 4 * 4 * 4)
        # Concatenate (B, C*(1+8+64))
        cat = torch.cat([p1, p2, p3], dim=1)
        # Project to embed_dim
        out = self.fc(cat)
        return out

# Step 1 & 14: It's assumed that TransSeg/ and MedViT/ have been cloned (see Section 1).
#   TransSeg is at ./TransSeg
#   MedViT is at ./MedViT

# ---------------------------------------------------------------------------- #
#                          ARGPARSE CONFIGURATION                                #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="Combined TransSeg & MedViT 3D Regression")
parser.add_argument("--train_csv", type=str, default="train_labels.csv",
                    help="CSV with columns [tomo_id, Number of motors]")
parser.add_argument("--lmdb_path", type=str, default="train.lmdb",
                    help="Path to existing LMDB file (from Code 1) containing {tomo_id: {'volume', 'label'}}")  # #7
parser.add_argument("--output_dir", type=str, default="output",
                    help="Where to save logs, checkpoints, etc.")
parser.add_argument("--epochs", type=int, default=100,
                    help="Maximum number of epochs per model (TransSeg/MedViT)")  # #37
parser.add_argument("--batch_size", type=int, default=2,
                    help="Batch size for both TransSeg and MedViT (step 36 allows >1)")  # #36
parser.add_argument("--lr", type=float, default=1e-4,
                    help="Initial learning rate (will be tuned by Optuna)")  # #34
parser.add_argument("--optimizer", type=str, choices=["adamw","sgd"], default="adamw",
                    help="Optimizer choice (AdamW with weight_decay=1e-4)")  # #10
parser.add_argument("--no_tune", action="store_true",
                    help="Skip Optuna tuning (if set, uses default hyperparams)")  # #34–35
parser.add_argument("--use_mixup", action="store_true",
                    help="Enable 3D MixUp in training (alpha=0.4)")  # #11
parser.add_argument("--freeze_epochs", type=int, default=5,
                    help="Number of epochs to freeze backbone then unfreeze (Step 12)")  # #12
parser.add_argument("--early_stop_patience", type=int, default=10,
                    help="Patience for early stopping (Step 13)")  # #13
parser.add_argument("--medvit_lr", type=float, default=5e-5,
                    help="Initial LR for MedViT (will be tuned if no_tune=False)")  # #34–35
parser.add_argument("--medvit_epochs", type=int, default=100,
                    help="Max epochs for MedViT training")  # #37
parser.add_argument("--medvit_freeze_epochs", type=int, default=5,
                    help="Freeze MedViT backbone for these many epochs (Step 12)")  # #12
parser.add_argument("--vit2d_ckpt", type=str, default="TransSeg/pretrained/vit_base_2d.pth",
                    help="Path to 2D ViT checkpoint for TransSeg inflation")  # #3
parser.add_argument("--medvit2d_ckpt", type=str, default="MedViT/pretrained/medvit_base_2d.pth",
                    help="Path to 2D MedViT checkpoint (if available) for inflation")  # #17
parser.add_argument("--num_folds", type=int, default=5,
                    help="Number of folds for K-Fold CV (Step 29)")  # #29
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

# Fix random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
monai.utils.set_determinism(seed=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------- #
#                              LOAD GLOBAL LABELS                               #
# ---------------------------------------------------------------------------- #
df = __import__("pandas").read_csv(args.train_csv)  # Step 39 ensures positive integers
# Keep only one row per tomo_id
df = df.drop_duplicates(subset="tomo_id", keep="first")
# ------------------------------------------------------------------------------ #
# REPLACE classification labels with regression targets (number of motors)     [CHANGED]
# ------------------------------------------------------------------------------ #
df["label"] = df["Number of motors"].astype(float)  # regression target
# For splitting only, cast to int (continuous stratification not supported)
split_labels = df["Number of motors"].astype(int).values   # used by ShuffleSplit/KFold
ids = df["tomo_id"].values
regression_targets = df["label"].values   # float targets
# Number of total files (e.g., 700)
n_total_files = len(ids)

# ---------------------------------------------------------------------------- #
#                             DATASET: LMDB LOADER                              #
# ---------------------------------------------------------------------------- #
class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, ids, targets, transforms=None):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False,
                             readahead=True, meminit=False)
        self.ids = list(ids)
        # Now 'targets' is float array of number-of-motors
        self.targets = list(targets)
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        key = self.ids[idx].encode("utf-8")
        with self.env.begin() as txn:
            data = txn.get(key)
        record = pickle.loads(data)
        img = record["volume"]           # np array (D,H,W)
        # IGNORE record["label"] from LMDB; use targets passed in
        lbl = self.targets[idx]          # float target
        sample = {"image": img, "label": lbl}
        if self.transforms:
            sample = self.transforms(sample)
        # Return image tensor and float label
        return sample["image"], torch.tensor(sample["label"], dtype=torch.float)

# ---------------------------------------------------------------------------- #
#                         3D DATA AUGMENTATION & TRANSFORMS                      #
# ---------------------------------------------------------------------------- #
train_transforms = Compose([
    # Step 8: Add channel dimension: (D,H,W) → (1,D,H,W)
    Lambdad(keys="image", func=lambda vol: vol[np.newaxis, ...]),

    # Intensity normalization
    ScaleIntensityRanged(keys="image", a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),

    # Random geometric transforms
    RandFlipd(keys="image", prob=0.5, spatial_axis=0),         # Step 8
    RandFlipd(keys="image", prob=0.5, spatial_axis=1),
    RandFlipd(keys="image", prob=0.5, spatial_axis=2),
    RandRotate90d(keys="image", prob=0.5, spatial_axes=(1,2)),

    # Random intensity & noise
    RandGaussianNoised(keys="image", prob=0.2, mean=0.0, std=0.1),  # Step 8
    RandBiasFieldd(keys="image", prob=0.3, coef_range=(0.1,0.5)),   # Step 8, #38
    RandGaussianSmoothd(keys="image", prob=0.2),                   # Step 8, #38
    RandAdjustContrastd(keys="image", prob=0.2, gamma=(0.7,1.5)),   # Step 8, #38

    ToTensord(keys=["image","label"]),
])

val_transforms = Compose([
    Lambdad(keys="image", func=lambda vol: vol[np.newaxis, ...]),
    ScaleIntensityRanged(keys="image", a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    ToTensord(keys=["image","label"]),
])

# ---------------------------------------------------------------------------- #
#                 OPTIONAL: K-FOLD CROSS-VALIDATION SPLITS                     #
# ---------------------------------------------------------------------------- #
if args.use_kfold:
    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=42)  # Step 29 [CHANGED]
    folds = list(kf.split(ids))
else:
    # Single split using ShuffleSplit (no stratification for regression)     [CHANGED]
    ss = ShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_idx, test_idx = next(ss.split(ids, split_labels))
    train_val_ids = ids[train_val_idx]
    train_val_targets = regression_targets[train_val_idx]
    test_ids = ids[test_idx]
    test_targets = regression_targets[test_idx]

    # Within train_val, split 80/20 → train/val
    ss2 = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)       # [CHANGED]
    train_idx, val_idx = next(ss2.split(train_val_ids, split_labels[train_val_idx]))
    train_ids = train_val_ids[train_idx]
    train_targets = train_val_targets[train_idx]
    val_ids = train_val_ids[val_idx]
    val_targets = train_val_targets[val_idx]

# ---------------------------------------------------------------------------- #
#                            MIXUP (OPTIONAL) SETUP                            #
# ---------------------------------------------------------------------------- #
def mixup_3d(inputs, targets, alpha=0.4):
    """Step 11: MixUp for 3D volumes (alpha=0.4)."""
    lam = np.random.beta(alpha, alpha)
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size).to(inputs.device)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
    labels_a, labels_b = targets, targets[index]
    return mixed_inputs, labels_a, labels_b, lam

# ---------------------------------------------------------------------------- #
#                             MODEL DEFINITION (TransSeg)                       #
# ---------------------------------------------------------------------------- #
# We import the TransSeg VisionTransformer3D and modify the head for regression.
sys.path.append("TransSeg/src")
from backbones.vit3d import VisionTransformer3D  # Step 5 requires editing this file

class TransSegRegressor(nn.Module):
    def __init__(self, pretrained_3d_ckpt, dropout_rate=0.15):
        super().__init__()
        # Step 5: Load VisionTransformer3D backbone
        self.backbone = VisionTransformer3D(
            img_size=(None,None,None),      # handled internally by LMDB loader
            patch_size=(2,16,16),
            in_chans=1,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0
        )
        # Step 9: Insert dropout + stochastic depth in backend blocks
        for block in self.backbone.blocks:
            block.drop1 = nn.Dropout(p=dropout_rate)  # after attention
            block.drop2 = nn.Dropout(p=dropout_rate)  # after MLP
            block.stochastic_depth = StochasticDepth(p=0.1, mode="row")

        # DVPP (#DVPP: Step 1.1)  
        # We will pool from the patch‐embedding feature map (pre‐flatten), which has 'embed_dim' channels.
        self.dvpp = DVPP3D(in_channels=768, embed_dim=768)  # DVPP → output is (B, 768)

        # Step 5: Replace classification head → regression head (now outputs 1 value)     [CHANGED]
        self.cls_head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 1)  # single float output
        )

        if os.path.isfile(pretrained_3d_ckpt):
            # Step 4: Load inflated 3D weights
            state_dict = torch.load(pretrained_3d_ckpt)
            self.backbone.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        # x: (B, 1, D, H, W)
        B = x.shape[0]

        # Step 5: Extract patch tokens + CLS token (UNCHANGED)  
        # BUT we intercept immediately after patch_embed to get a 3D feature map (before flatten).
        feat = self.backbone.patch_embed(x)                 # DVPP needs this 3D tensor → shape (B, 768, D//2, H//16, W//16)

        # DVPP (#DVPP: Step 2.1)  
        dvpp_feat = self.dvpp(feat)                         # (B, 768) fixed‐length

        # DVPP path (#DVPP: Step 2.2)  
        # Bypass the transformer CLStoken path entirely—feed DVPP vector into regression head:
        output = self.cls_head(dvpp_feat).squeeze(1)         # (B,) float outputs
        return output

# ---------------------------------------------------------------------------- #
#                              MODEL DEFINITION (MedViT)                         #
# ---------------------------------------------------------------------------- #
# We import MedViT code, modify its PatchEmbed to 3D, then swap its head.

sys.path.append("MedViT")
from MedViT import MedViT as MedViTBase  # Original 2D MedViT class

class MedViT3DRegressor(nn.Module):
    def __init__(self, pretrained_2d_ckpt, image_size=(1,64,256,256),
                 patch_size=(2,16,16), embed_dim=768, depth=12, num_heads=12, dropout_rate=0.15):
        super().__init__()
        D, H, W = image_size[1], image_size[2], image_size[3]
        # Step 16: Replace PatchEmbed with 3D version
        self.patch_embed3d = nn.Conv3d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        num_patches = (D // patch_size[0]) * (H // patch_size[1]) * (W // patch_size[2])

        # Step 18: 3D Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder layers (copied/adapted from MedViT)
        self.blocks = nn.ModuleList([
            MedViTBase.Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop=dropout_rate,              # Step 9: dropout in Transformer
                attn_drop=dropout_rate,
                drop_path=0.1                   # Step 9: stochastic depth
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # DVPP (#DVPP: Step 1.2)  
        # Apply DVPP to the raw 3D features from patch_embed3d (B, embed_dim, D/2, H/16, W/16)
        self.dvpp = DVPP3D(in_channels=embed_dim, embed_dim=embed_dim)  # (B, 768)

        # Step 19: Replace classification head → regression head (output 1 value)     [CHANGED]
        self.head = nn.Linear(embed_dim, 1)

        # Step 17: Inflate 2D → 3D if 2D checkpoint exists
        if os.path.isfile(pretrained_2d_ckpt):
            sd2 = torch.load(pretrained_2d_ckpt)
            sd3 = {}
            for k, v in sd2.items():
                # Handle patch embedding separately
                if "patch_embed.proj.weight" in k:
                    # v shape: (embed_dim, 3, P, P) → avg channels → (embed_dim, 1, P, P)
                    v = v.mean(dim=1, keepdim=True)  # collapse color channels
                    # inflate depth dimension
                    depth_k = patch_size[0]
                    v = v.unsqueeze(-1).repeat(1, 1, 1, 1, depth_k) / depth_k
                    sd3["patch_embed3d.weight"] = v
                    continue
                # Map other keys 1:1 if shape matches
                new_k = k.replace("patch_embed.", "patch_embed3d.")
                if new_k in self.state_dict() and self.state_dict()[new_k].shape == v.shape:
                    sd3[new_k] = v
            self.load_state_dict(sd3, strict=False)

    def forward(self, x):
        # x: (B, 1, D, H, W)
        B = x.size(0)

        # Step 16: Extract raw 3D features from patch_embed3d
        feat = self.patch_embed3d(x)                # (B, embed_dim, D/2, H/16, W/16)

        # DVPP (#DVPP: Step 2.3)  
        dvpp_feat = self.dvpp(feat)                  # (B, 768)

        # DVPP path (#DVPP: Step 2.4)  
        output = self.head(dvpp_feat).squeeze(1)      # (B,) float outputs
        return output

# ---------------------------------------------------------------------------- #
#                         MEDICALNET PRETRAINED 3D CNN OPTION                   #
# ---------------------------------------------------------------------------- #
def get_medicalnet_backbone(name="resnet101", num_outputs=1):
    """Step 26: Return MedicalNet 3D pretrained model adapted for regression."""
    if name == "resnet101":
        from monai.networks.nets import resnet
        model = resnet.resnet101(
            spatial_dims=3, in_channels=1, num_classes=num_outputs, pretrained=True
        )
    elif name == "resnext50":
        from monai.networks.nets import resnet
        model = resnet.resnext50_32x4d(
            spatial_dims=3, in_channels=1, num_classes=num_outputs, pretrained=True
        )
    else:
        raise ValueError("Unsupported MedicalNet backbone")
    return model

# ---------------------------------------------------------------------------- #
#                              EVALUATION FUNCTIONS                              #
# ---------------------------------------------------------------------------- #
def evaluate_model(model, dataloader):
    """Evaluate model on dataloader → returns (MAE, RMSE, Accuracy)."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            outputs = model(imgs)           # (B,) floats
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels).astype(int)

    mae  = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))

    # Compute rounded‐to‐integer accuracy
    rounded = np.rint(all_preds).astype(int)
    acc     = accuracy_score(all_labels, rounded)

    return mae, rmse, acc

# ---------------------------------------------------------------------------- #
#                      TRANSSEG TRAIN / VALID / TEST LOOP                        #
# ---------------------------------------------------------------------------- #
def train_transseg_fold(train_ids, train_targets, val_ids, val_targets, test_ids, test_targets, fold_idx=0):
    fold_dir = os.path.join(args.output_dir, f"transseg_fold{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    # Create datasets & loaders
    train_ds = LMDBDataset(args.lmdb_path, train_ids, train_targets, transforms=train_transforms)
    val_ds   = LMDBDataset(args.lmdb_path, val_ids, val_targets, transforms=val_transforms)
    test_ds  = LMDBDataset(args.lmdb_path, test_ids, test_targets, transforms=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    # Initialize model
    model = TransSegRegressor(
        pretrained_3d_ckpt="TransSeg/pretrained/vit_base_3d.pth"  # Step 4
    ).to(device)

    # Step 10: Choose AdamW or SGD with weight decay
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)  # Step 10
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # Step 32: One-Cycle LR
    total_steps = args.epochs * len(train_loader)
    scheduler   = OneCycleLR(optimizer, max_lr=args.max_lr, total_steps=total_steps,
                             pct_start=0.3, anneal_strategy="cos")

    # ---------------------------------------------------------------------------- #
    # REPLACE CrossEntropyLoss with MSELoss for regression                         [CHANGED]
    # ---------------------------------------------------------------------------- #
    criterion = nn.MSELoss()

    # Step 12: mixed precision
    scaler = GradScaler()

    # Step 12: freeze backbone
    for name, param in model.named_parameters():
        if "cls_head" not in name:
            param.requires_grad = False

    best_val_mae = float("inf")
    no_improve   = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, lbls in train_loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            # Step 11: MixUp logic (labels are floats, so mixture is straightforward)
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
                scheduler.step()  # Step 32: scheduler update

            running_loss += loss.item() * imgs.size(0)

        # Unfreeze after freeze_epochs (Step 12)
        if epoch == args.freeze_epochs + 1:
            for param in model.parameters():
                param.requires_grad = True
            for g in optimizer.param_groups:
                g["lr"] *= 0.1

        # Validation
        model.eval()
        val_preds, val_labels_list = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                outputs = model(imgs)
                val_preds.extend(outputs.cpu().numpy())
                val_labels_list.extend(lbls.cpu().numpy())

        val_preds        = np.array(val_preds)
        val_labels_array = np.array(val_labels_list).astype(int)
        val_mae          = mean_absolute_error(val_labels_array, val_preds)
        val_rmse         = np.sqrt(mean_squared_error(val_labels_array, val_preds))

        # ---------------------------------------------------------------------------- #
        # INSERTED SNIPPET: print total files, validation count, rounded accuracy      [NEW]
        # ---------------------------------------------------------------------------- #
        rounded_preds = np.rint(val_preds).astype(int)
        n_val         = len(val_labels_array)
        n_correct     = int((rounded_preds == val_labels_array).sum())
        print(f"Number of files = {n_total_files}")
        print(f"Number of Validation tests = {n_val}")
        print(f"True estimated validation predictions (count) / all validation predictions (count) = "
              f"{n_correct} / {n_val}  "
              f"(accuracy = {n_correct}/{n_val} = {n_correct/n_val:.3f})")
        # ---------------------------------------------------------------------------- #

        print(f"[TransSeg Fold {fold_idx}] Epoch {epoch} | Train MSE: {running_loss/len(train_loader.dataset):.4f} "
              f"| Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}")

        # Early stopping based on MAE
        if val_mae < best_val_mae - 1e-4:
            best_val_mae = val_mae
            no_improve   = 0
            torch.save(model.state_dict(), os.path.join(fold_dir, "best_transseg_model.pth"))
        else:
            no_improve += 1
            if no_improve >= args.early_stop_patience:
                print(f"[TransSeg Fold {fold_idx}] Early stopping at epoch {epoch}")
                break

    # Load best model for final evaluation on test split
    model.load_state_dict(torch.load(os.path.join(fold_dir, "best_transseg_model.pth")))
    test_mae, test_rmse, test_acc = evaluate_model(model, test_loader)
    print(f"[TransSeg Fold {fold_idx}] Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}, Test Accuracy: {test_acc:.3f}")

    return model  # Return trained TransSeg model

# ---------------------------------------------------------------------------- #
#                          MEDVIT TRAIN / VALID / TEST LOOP                     #
# ---------------------------------------------------------------------------- #
def train_medvit_fold(train_ids, train_targets, val_ids, val_targets, test_ids, test_targets, fold_idx=0):
    fold_dir = os.path.join(args.output_dir, f"medvit_fold{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    # Create datasets & loaders
    train_ds = LMDBDataset(args.lmdb_path, train_ids, train_targets, transforms=train_transforms)
    val_ds   = LMDBDataset(args.lmdb_path, val_ids, val_targets, transforms=val_transforms)
    test_ds  = LMDBDataset(args.lmdb_path, test_ids, test_targets, transforms=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    # Step 22: Initialize MedViT3DRegressor
    medvit_model = MedViT3DRegressor(
        pretrained_2d_ckpt=args.medvit2d_ckpt,  # Step 17
        image_size=(1, 64, 256, 256),             # update to actual volume dims
        patch_size=(2, 16, 16),
        embed_dim=768,
        depth=12,
        num_heads=12,
        dropout_rate=0.15  # Step 9
    ).to(device)

    # Step 10: Optimizer for MedViT
    optimizer = optim.AdamW(medvit_model.parameters(), lr=args.medvit_lr, weight_decay=1e-4)

    # Step 32: One-Cycle LR for MedViT
    total_steps = args.medvit_epochs * len(train_loader)
    scheduler   = OneCycleLR(optimizer, max_lr=args.max_lr, total_steps=total_steps,
                             pct_start=0.3, anneal_strategy="cos")

    # ---------------------------------------------------------------------------- #
    # REPLACE classification loss with MSELoss for regression                       [CHANGED]
    # ---------------------------------------------------------------------------- #
    criterion = nn.MSELoss()

    # Step 12: mixed precision
    scaler = GradScaler()

    # Step 12: freeze head only for first medvit_freeze_epochs
    for name, param in medvit_model.named_parameters():
        if "head" not in name:
            param.requires_grad = False

    best_val_mae = float("inf")
    no_improve   = 0

    for epoch in range(1, args.medvit_epochs + 1):
        medvit_model.train()
        running_loss = 0.0
        for imgs, lbls in train_loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            with autocast():
                outputs = medvit_model(imgs)
                loss = criterion(outputs, lbls)  # regression loss
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if not args.no_tune:
                scheduler.step()

            running_loss += loss.item() * imgs.size(0)

        # Unfreeze after medvit_freeze_epochs (Step 12)
        if epoch == args.medvit_freeze_epochs + 1:
            for param in medvit_model.parameters():
                param.requires_grad = True
            for g in optimizer.param_groups:
                g["lr"] *= 0.1

        # Validation
        medvit_model.eval()
        val_preds, val_labels_list = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                outputs = medvit_model(imgs)
                val_preds.extend(outputs.cpu().numpy())
                val_labels_list.extend(lbls.cpu().numpy())

        val_preds        = np.array(val_preds)
        val_labels_array = np.array(val_labels_list).astype(int)
        val_mae          = mean_absolute_error(val_labels_array, val_preds)
        val_rmse         = np.sqrt(mean_squared_error(val_labels_array, val_preds))

        # ---------------------------------------------------------------------------- #
        # INSERTED SNIPPET: print total files, validation count, rounded accuracy      [NEW]
        # ---------------------------------------------------------------------------- #
        rounded_preds = np.rint(val_preds).astype(int)
        n_val         = len(val_labels_array)
        n_correct     = int((rounded_preds == val_labels_array).sum())
        print(f"Number of files = {n_total_files}")
        print(f"Number of Validation tests = {n_val}")
        print(f"True estimated validation predictions (count) / all validation predictions (count) = "
              f"{n_correct} / {n_val}  "
              f"(accuracy = {n_correct}/{n_val} = {n_correct/n_val:.3f})")
        # ---------------------------------------------------------------------------- #

        print(f"[MedViT Fold {fold_idx}] Epoch {epoch} | Train MSE: {running_loss/len(train_loader.dataset):.4f} "
              f"| Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}")

        # Early stopping based on MAE
        if val_mae < best_val_mae - 1e-4:
            best_val_mae = val_mae
            no_improve   = 0
            torch.save(medvit_model.state_dict(), os.path.join(fold_dir, "best_medvit_model.pth"))
        else:
            no_improve += 1
            if no_improve >= args.early_stop_patience:
                print(f"[MedViT Fold {fold_idx}] Early stopping at epoch {epoch}")
                break

    # Load best model for final eval on test
    medvit_model.load_state_dict(torch.load(os.path.join(fold_dir, "best_medvit_model.pth")))
    test_mae, test_rmse, test_acc = evaluate_model(medvit_model, test_loader)
    print(f"[MedViT Fold {fold_idx}] Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}, Test Accuracy: {test_acc:.3f}")

    return medvit_model  # Return trained MedViT model

# ---------------------------------------------------------------------------- #
#                          ENSEMBLE UTILITY (Step 25)                            #
# ---------------------------------------------------------------------------- #
def ensemble_predict(img, model1, model2):
    """
    Step 25: Average regression outputs of TransSeg & MedViT → final prediction.
    """
    with torch.no_grad():
        out1 = model1(img.unsqueeze(0).to(device)).cpu().numpy().item()
        out2 = model2(img.unsqueeze(0).to(device)).cpu().numpy().item()
    # Simple average of floats
    final = (out1 + out2) / 2.0
    return final

# ---------------------------------------------------------------------------- #
#                           MAIN TRAINING / EVAL ROUTINE                         #
# ---------------------------------------------------------------------------- #
def main():
    if args.use_kfold:
        # Step 29: 5-Fold CV
        for fold_idx, (train_val_idx, test_idx) in enumerate(folds):
            train_val_ids       = ids[train_val_idx]
            train_val_targets   = regression_targets[train_val_idx]
            test_ids_fold       = ids[test_idx]
            test_targets_fold   = regression_targets[test_idx]

            # Further split train_val into train/val (80/20)
            ss2 = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  # [CHANGED]
            tr_idx, vl_idx = next(ss2.split(train_val_ids, split_labels[train_val_idx]))
            train_ids_fold     = train_val_ids[tr_idx]
            train_targets_fold = train_val_targets[tr_idx]
            val_ids_fold       = train_val_ids[vl_idx]
            val_targets_fold   = train_val_targets[vl_idx]

            # Train & evaluate TransSeg and MedViT on this fold
            print(f"#### Starting TransSeg fold {fold_idx} ####")
            transseg_model = train_transseg_fold(
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

            # Step 31: Optional pruning
            if args.use_pruning:
                import torch.nn.utils.prune as prune
                for module in transseg_model.modules():
                    if isinstance(module, nn.Linear):
                        prune.l1_unstructured(module, name="weight", amount=0.3)
                for module in medvit_model.modules():
                    if isinstance(module, nn.Linear):
                        prune.l1_unstructured(module, name="weight", amount=0.3)
                # Fine-tune pruned models briefly (omitted for brevity)

            # Step 31: Optional Distillation
            if args.use_distillation:
                # Teacher: transseg_model; Student: smaller ResNet50_3D
                student = get_medicalnet_backbone(
                    name="resnet50", num_outputs=1
                ).to(device)
                # Distill for a few epochs (omitted for brevity)

            # Step 25: Example single-sample ensemble on test set
            print(f"#### Ensemble results for fold {fold_idx} ####")
            test_ds_fold   = LMDBDataset(args.lmdb_path, test_ids_fold, test_targets_fold, transforms=val_transforms)
            test_loader_fold = DataLoader(test_ds_fold, batch_size=1, shuffle=False, num_workers=2)

            ensemble_preds, ensemble_labels = [], []
            for img, lbl in test_loader_fold:
                pred = ensemble_predict(img, transseg_model, medvit_model)
                ensemble_preds.append(pred)
                ensemble_labels.append(lbl.item())

            # Compute ensemble metrics (MAE, RMSE, Accuracy)
            ens_mae  = mean_absolute_error(ensemble_labels, ensemble_preds)
            ens_rmse = np.sqrt(mean_squared_error(ensemble_labels, ensemble_preds))
            rounded  = np.rint(np.array(ensemble_preds)).astype(int)
            ens_acc  = accuracy_score(np.array(ensemble_labels).astype(int), rounded)

            print(f"[Ensemble Fold {fold_idx}] Test MAE: {ens_mae:.4f}, Test RMSE: {ens_rmse:.4f}, Test Accuracy: {ens_acc:.3f}")

    else:
        # Single‐split scenario
        print("#### Starting single-split TransSeg training ####")
        transseg_model = train_transseg_fold(
            train_ids, train_targets, val_ids, val_targets,
            test_ids, test_targets, fold_idx=0
        )

        print("#### Starting single-split MedViT training ####")
        medvit_model = train_medvit_fold(
            train_ids, train_targets, val_ids, val_targets,
            test_ids, test_targets, fold_idx=0
        )

        # Step 31: Optional pruning
        if args.use_pruning:
            import torch.nn.utils.prune as prune
            for module in transseg_model.modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name="weight", amount=0.3)
            for module in medvit_model.modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name="weight", amount=0.3)
            # Fine-tune pruned models briefly

        # Step 31: Optional distillation (omitted for brevity)

        # Step 25: Ensemble on test set
        test_ds    = LMDBDataset(args.lmdb_path, test_ids, test_targets, transforms=val_transforms)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
        ens_preds, ens_labels = [], []
        for img, lbl in test_loader:
            pred = ensemble_predict(img, transseg_model, medvit_model)
            ens_preds.append(pred)
            ens_labels.append(lbl.item())

        ens_mae  = mean_absolute_error(ens_labels, ens_preds)
        ens_rmse = np.sqrt(mean_squared_error(ens_labels, ens_preds))
        rounded  = np.rint(np.array(ens_preds)).astype(int)
        ens_acc  = accuracy_score(np.array(ens_labels).astype(int), rounded)

        print(f"[Ensemble] Test MAE: {ens_mae:.4f}, Test RMSE: {ens_rmse:.4f}, Test Accuracy: {ens_acc:.3f}")

if __name__ == "__main__":
    main()
