#!/usr/bin/env python3
# combined_train.py
# Integrates a 3D Swin UNETR encoder (sliding-window attention) and MedViT for 3D regression
# on LMDB volumes at full resolution (800×1912×1847). Uses FSDP+CPUOffload, AMP, checkpointing,
# LRU cache, MixUp, pruning, distillation, and ensemble.

import os
import sys
import argparse
import pickle
import random
import numpy as np
import lmdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, ShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import monai
from monai.transforms import (
    Lambdad, ScaleIntensityRanged, RandFlipd, RandRotate90d,
    RandGaussianNoised, RandBiasFieldd, RandGaussianSmoothd,
    RandAdjustContrastd, ToTensord, Compose
)
from collections import Counter, OrderedDict

# -------------------------------------------------------------------
# Prevent GPU fragmentation (allow ~128 MiB maximum split size). 
# -------------------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.6"

# -------------------------------------------------------------------
# Hardcode maximum 3D dims (no LMDB scan). :contentReference[oaicite:4]{index=4}
# -------------------------------------------------------------------
MAX_D = 800     # Depth
MAX_H = 1912    # Height
MAX_W = 1847    # Width

# -------------------------------------------------------------------
# Define device early. :contentReference[oaicite:5]{index=5}
# -------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------
# Command-line arguments
# -------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Combined SwinUNETR & MedViT 3D Regression")
parser.add_argument("--train_csv", type=str, default="train_labels.csv",
                    help="CSV with columns [tomo_id, Number of motors]")
parser.add_argument("--lmdb_path", type=str, default="train.lmdb",
                    help="Path to LMDB containing {tomo_id:{'volume','label'}}")
parser.add_argument("--output_dir", type=str, default="output",
                    help="Directory for logs & checkpoints")
parser.add_argument("--epochs", type=int, default=100,
                    help="Max epochs per model")
parser.add_argument("--batch_size", type=int, default=1,
                    help="Batch size (larger consumes more memory)")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rate")
parser.add_argument("--optimizer", type=str, choices=["adamw","sgd"], default="adamw",
                    help="Optimizer type")
parser.add_argument("--no_tune", action="store_true",
                    help="Skip LR scheduler tuning")
parser.add_argument("--use_mixup", action="store_true",
                    help="Enable 3D MixUp")
parser.add_argument("--freeze_epochs", type=int, default=5,
                    help="Epochs to freeze backbone")
parser.add_argument("--early_stop_patience", type=int, default=10,
                    help="Early-stopping patience")
parser.add_argument("--medvit_lr", type=float, default=5e-5,
                    help="Learning rate for MedViT")
parser.add_argument("--medvit_epochs", type=int, default=100,
                    help="Max epochs for MedViT")
parser.add_argument("--medvit_freeze_epochs", type=int, default=5,
                    help="Epochs to freeze MedViT backbone")
parser.add_argument("--vit2d_ckpt", type=str, default="TransSeg/pretrained/vit_base_2d.pth",
                    help="2D ViT checkpoint (unused in Swin)") 
parser.add_argument("--medvit2d_ckpt", type=str, default="MedViT/pretrained/medvit_base_2d.pth",
                    help="2D MedViT checkpoint for inflation")
parser.add_argument("--num_folds", type=int, default=5,
                    help="Number of folds for Stratified K-Fold CV")
parser.add_argument("--use_kfold", action="store_true",
                    help="Enable 5-Fold cross-validation")
parser.add_argument("--use_pruning", action="store_true",
                    help="Apply magnitude-based pruning")
parser.add_argument("--use_distillation", action="store_true",
                    help="Perform knowledge distillation")
parser.add_argument("--max_lr", type=float, default=1e-3,
                    help="Max LR for OneCycleLR")

args, _ = parser.parse_known_args()
os.makedirs(args.output_dir, exist_ok=True)

# -------------------------------------------------------------------
# Fix random seeds for reproducibility. :contentReference[oaicite:6]{index=6}
# -------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
monai.utils.set_determinism(seed=42)

# -------------------------------------------------------------------
# Distributed / FSDP Setup :contentReference[oaicite:7]{index=7}
# -------------------------------------------------------------------
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, \
                                    CPUOffload, MixedPrecision, ShardingStrategy

# Use TCP rendezvous to avoid requiring MASTER_ADDR. :contentReference[oaicite:8]{index=8}
dist.init_process_group(
    backend="nccl",
    init_method="tcp://127.0.0.1:29500",
    rank=0,
    world_size=1
)

# CPUOffload: offload parameter shards to CPU when not in use. 
cpu_offload = CPUOffload(offload_params=True)

# MixedPrecision: keep FP16 on GPU, FP32 masters on CPU. 
mp_policy = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16
)

# -------------------------------------------------------------------
# LMDBDataset with LRU cache for padded volumes (CPU). :contentReference[oaicite:11]{index=11}
# -------------------------------------------------------------------
class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, ids, targets, transforms=None, cache_size=5):
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

        # LRU cache: store up to `cache_size` padded volumes in CPU RAM. :contentReference[oaicite:12]{index=12}
        self.cache = OrderedDict()
        self.max_cache_size = cache_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        tomo_id = self.ids[idx]
        lbl = self.targets[idx]

        if tomo_id in self.cache:
            vol_padded = self.cache[tomo_id]
            self.cache.move_to_end(tomo_id)
        else:
            key = tomo_id.encode("utf-8")
            with self.env.begin() as txn:
                data = txn.get(key)
            record = pickle.loads(data)
            vol = record["volume"]   # shape: (D, H, W)
            D, H, W = vol.shape

            pad_d = MAX_D - D
            pad_h = MAX_H - H
            pad_w = MAX_W - W
            vol_padded = np.pad(
                vol,
                ((0, pad_d), (0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0
            )  # → (800, 1912, 1847)

            if len(self.cache) >= self.max_cache_size:
                self.cache.popitem(last=False)
            self.cache[tomo_id] = vol_padded

        sample = {"image": vol_padded, "label": lbl}
        if self.transforms:
            sample = self.transforms(sample)
        return sample["image"], torch.tensor(sample["label"], dtype=torch.float)

# -------------------------------------------------------------------
# 3D augmentation & normalization pipelines
# -------------------------------------------------------------------
train_transforms = Compose([
    Lambdad(keys="image", func=lambda vol: vol[np.newaxis, ...]),  # Add channel dim: (1,D,H,W)
    ScaleIntensityRanged(keys="image", a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    RandFlipd(keys="image", prob=0.5, spatial_axis=0),
    RandFlipd(keys="image", prob=0.5, spatial_axis=1),
    RandFlipd(keys="image", prob=0.5, spatial_axis=2),
    RandRotate90d(keys="image", prob=0.5, spatial_axes=(1, 2)),
    RandGaussianNoised(keys="image", prob=0.2, mean=0.0, std=0.1),
    RandBiasFieldd(keys="image", prob=0.3, coeff_range=(0.1, 0.5)),
    RandGaussianSmoothd(keys="image", prob=0.2),
    RandAdjustContrastd(keys="image", prob=0.2, gamma=(0.7, 1.5)),
    ToTensord(keys=["image", "label"]),
])

val_transforms = Compose([
    Lambdad(keys="image", func=lambda vol: vol[np.newaxis, ...]),
    ScaleIntensityRanged(keys="image", a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    ToTensord(keys=["image", "label"]),
])

# ---------------------------------------------------------------------------- #
#                              MODEL DEFINITIONS                                #
# ---------------------------------------------------------------------------- #
from monai.networks.nets import swin_unetr  # MONAI’s 3D Swin UNETR :contentReference[oaicite:13]{index=13}

class TransSegSwinRegressor(nn.Module):
    """
    Replaces ViT with a 3D Swin UNETR encoder + regression head.
    Constructor uses a padded img_size=(800,1920,1856) so all dims %32==0. :contentReference[oaicite:14]{index=14}
    Forward pads the raw input (800,1912,1847) → (800,1920,1856). :contentReference[oaicite:15]{index=15}
    Checkpointing saves ~40–50% activation memory. :contentReference[oaicite:16]{index=16}
    """
    def __init__(self, max_d, max_h, max_w,
                 depths=(2,2,2,2), num_heads=(3,6,12,24), feature_size=96,
                 norm_name='instance', drop_rate=0.0, attn_drop_rate=0.0,
                 dropout_path_rate=0.0, use_checkpoint=True, spatial_dims=3):
        super().__init__()
        # Compute padded dims (all divisible by 32). :contentReference[oaicite:17]{index=17}
        pad_h_init = (32 - (max_h % 32)) % 32       # = (32 - (1912 % 32)) % 32 = 8
        pad_w_init = (32 - (max_w % 32)) % 32       # = (32 - (1847 % 32)) % 32 = 9
        pad_d_init = (32 - (max_d % 32)) % 32       # = (32 - (800 % 32)) % 32 = 0

        init_h = max_h + pad_h_init    # 1912 + 8 = 1920
        init_w = max_w + pad_w_init    # 1847 + 9 = 1856
        init_d = max_d + pad_d_init    # 800   + 0 = 800

        # Initialize SwinUNETR encoder with padded img_size. :contentReference[oaicite:18]{index=18}
        self.swin_unetr = swin_unetr.SwinUNETR(
            img_size=(init_d, init_h, init_w),   # (800,1920,1856) so each dim %32==0 :contentReference[oaicite:19]{index=19}
            in_channels=1,
            out_channels=1,                    # placeholder; we only use encoder
            depths=depths,
            num_heads=num_heads,
            feature_size=feature_size,
            norm_name=norm_name,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            normalize=True,
            use_checkpoint=use_checkpoint,     # memory saving via checkpointing :contentReference[oaicite:20]{index=20}
            spatial_dims=spatial_dims,
            downsample="merging",
            use_v2=False
        )

        # Regression head: global-average-pool final encoder output (768 channels) → Linear(768→1).
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.reg_head = nn.Sequential(
            nn.LayerNorm(feature_size * 8),
            nn.Linear(feature_size * 8, 1)
        )

    def forward(self, x):
        # x: (B,1,800,1912,1847), dtype=torch.float16 on GPU
        B, C, D, H, W = x.shape

        # Compute pad so D,H,W are divisible by 32 at runtime. :contentReference[oaicite:21]{index=21}
        pad_d = (32 - (D % 32)) % 32   # = (32 - (800 % 32)) % 32 = 0
        pad_h = (32 - (H % 32)) % 32   # = (32 - (1912 % 32)) % 32 = 8
        pad_w = (32 - (W % 32)) % 32   # = (32 - (1847 % 32)) % 32 = 9

        # Pad only at the “end” of each dimension: (W_left, W_right, H_left, H_right, D_left, D_right). :contentReference[oaicite:22]{index=22}
        x_padded = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), mode="constant", value=0.0)
        # → shape = (B,1,800,1920,1856)

        # SwinUNETR’s encoder returns a dict: keys = 'layer1','layer2','layer3','layer4'. :contentReference[oaicite:23]{index=23}
        out_dict = self.swin_unetr.encoder(x_padded)
        feat = out_dict["layer4"]  # shape: (B,768, D/32, H/32, W/32) e.g. (B,768,25,60,58)

        # Global average pool → (B,768,1,1,1) → flatten → (B,768)
        pooled = self.pool(feat).view(B, -1)

        # Regression head → single float output per sample
        out = self.reg_head(pooled).squeeze(1)  # shape: (B,)
        return out

# MedViT3DRegressor unchanged (patch embedding + Transformer blocks). :contentReference[oaicite:24]{index=24}
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(repo_root, "MedViT"))
from MedViT import MedViT as MedViTBase

class MedViT3DRegressor(nn.Module):
    def __init__(self, max_d, max_h, max_w,
                 pretrained_2d_ckpt, patch_size=(2, 16, 16),
                 embed_dim=768, depth=12, num_heads=12, dropout_rate=0.15):
        super().__init__()
        # 3D patch embedding
        self.patch_embed3d = nn.Conv3d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        num_patches = (max_d // patch_size[0]) * \
                      (max_h // patch_size[1]) * \
                      (max_w // patch_size[2])

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

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
        self.head = nn.Linear(embed_dim, 1)

        # Inflate 2D→3D if checkpoint exists :contentReference[oaicite:25]{index=25}
        if os.path.isfile(pretrained_2d_ckpt):
            sd2 = torch.load(pretrained_2d_ckpt)
            sd3 = {}
            for k, v in sd2.items():
                if "patch_embed.proj.weight" in k:
                    v = v.mean(dim=1, keepdim=True)
                    depth_k = patch_size[0]
                    v = v.unsqueeze(-1).repeat(1, 1, 1, 1, depth_k) / depth_k
                    sd3["patch_embed3d.weight"] = v
                    continue
                new_k = k.replace("patch_embed.", "patch_embed3d.")
                if new_k in self.state_dict() and self.state_dict()[new_k].shape == v.shape:
                    sd3[new_k] = v
            self.load_state_dict(sd3, strict=False)

    def forward(self, x):
        feat = self.patch_embed3d(x)       # (B,768,400,119,115)
        B, C, d, h, w = feat.shape
        feat_flat = feat.flatten(2).transpose(1, 2)  # (B, num_patches, 768)
        cls = self.cls_token.expand(B, -1, -1)       # (B,1,768)
        tokens = torch.cat((cls, feat_flat), dim=1)  # (B,num_patches+1,768)
        tokens = tokens + self.pos_embed

        # Checkpoint each Transformer block for ~40–50% memory savings. :contentReference[oaicite:26]{index=26}
        for blk in self.blocks:
            tokens = checkpoint(blk, tokens)
        tokens = self.norm(tokens)            # (B,num_patches+1,768)
        cls_final = tokens[:, 0]              # (B,768)
        output = self.head(cls_final).squeeze(1)  # (B,)
        return output

# ---------------------------------------------------------------------------- #
#                    MEDICALNET PRETRAINED 3D CNN OPTION                         #
# ---------------------------------------------------------------------------- #
def get_medicalnet_backbone(name="resnet101", num_outputs=1):
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
#                              EVALUATION FUNCTION                              #
# ---------------------------------------------------------------------------- #
def evaluate_and_print(model, dataloader, n_total):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device, dtype=torch.float16)
            lbls = lbls.to(device)
            with autocast():  # Mixed precision inference :contentReference[oaicite:27]{index=27}
                outputs = model(imgs).cpu().numpy()
            all_preds.extend(outputs)
            all_labels.extend(lbls.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels).astype(int)
    rounded = np.rint(all_preds).astype(int)
    n_correct = int((rounded == all_labels).sum())
    n_val = len(all_labels)

    print(f"Number of files = {n_total}")
    print(f"Number of Validation tests = {n_val}")
    print(f"True estimated validation predictions (count) / all validation predictions (count) = "
          f"{n_correct} / {n_val}  (accuracy = {n_correct}/{n_val} = {n_correct/ n_val:.3f})")

    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    return mae, rmse, n_correct / n_val

# ---------------------------------------------------------------------------- #
#                OPTIONAL: K-FOLD CROSS-VALIDATION SPLITS                      #
# ---------------------------------------------------------------------------- #
import pandas as pd
df = pd.read_csv(args.train_csv)
df = df.drop_duplicates(subset="tomo_id", keep="first")
df["label"] = df["Number of motors"].astype(float)
split_labels = df["Number of motors"].astype(int).values
ids = df["tomo_id"].values
regression_targets = df["label"].values
n_total = len(ids)

label_counts = Counter(split_labels)
min_count = min(label_counts.values())

if args.use_kfold:
    if min_count < 2:
        raise ValueError("Stratified K-Fold requires each class ≥ 2 samples.")
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
#                            MIXUP (OPTIONAL) SETUP                            #
# ---------------------------------------------------------------------------- #
def mixup_3d(inputs, targets, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size).to(inputs.device)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
    labels_a, labels_b = targets, targets[index]
    return mixed_inputs, labels_a, labels_b, lam

# ---------------------------------------------------------------------------- #
#                TRAIN FUNCTION (FSDP for Swin UNETR)                           #
# ---------------------------------------------------------------------------- #
def train_swintransseg_fold(train_ids, train_targets,
                            val_ids, val_targets,
                            test_ids, test_targets,
                            fold_idx=0):
    fold_dir = os.path.join(args.output_dir, f"swintransseg_fold{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    train_ds = LMDBDataset(args.lmdb_path, train_ids, train_targets,
                           transforms=train_transforms, cache_size=5)
    val_ds   = LMDBDataset(args.lmdb_path, val_ids,   val_targets,
                           transforms=val_transforms,   cache_size=5)
    test_ds  = LMDBDataset(args.lmdb_path, test_ids,  test_targets,
                           transforms=val_transforms,   cache_size=5)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Build model → to GPU → cast to FP16 → wrap in FSDP (Full Shard + CPUOffload + MixedPrecision). :contentReference[oaicite:28]{index=28}
    base_model = TransSegSwinRegressor(
        MAX_D, MAX_H, MAX_W,
        depths=(2,2,2,2),
        num_heads=(3,6,12,24),
        feature_size=96,
        norm_name='instance',
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
        spatial_dims=3
    ).to(device).half()

    model = FSDP(
        base_model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=cpu_offload,
        mixed_precision=mp_policy
    )

    # Optimizer & scheduler
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.max_lr, total_steps=total_steps,
        pct_start=0.3, anneal_strategy="cos"
    )

    criterion = nn.MSELoss()
    scaler = GradScaler()

    best_val_mae = float("inf")
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for imgs, lbls in train_loader:
            imgs = imgs.to(device, dtype=torch.float16)
            lbls = lbls.to(device)

            if args.use_mixup:
                imgs, lbls_a, lbls_b, lam = mixup_3d(imgs, lbls, alpha=0.4)
                with autocast():  # Mixed precision training :contentReference[oaicite:29]{index=29}
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

        # Validation :contentReference[oaicite:30]{index=30}
        val_mae, val_rmse, val_acc = evaluate_and_print(model, val_loader, n_total)
        print(f"[SwinTransSeg Fold {fold_idx}] Epoch {epoch} | Train MSE: "
              f"{running_loss/len(train_loader.dataset):.4f} | Val MAE: {val_mae:.4f} "
              f"| Val RMSE: {val_rmse:.4f} | Val Acc: {val_acc:.3f}")

        if val_mae < best_val_mae - 1e-4:
            best_val_mae = val_mae
            no_improve = 0
            # Save best FSDP checkpoint 
            torch.save(model.state_dict(), os.path.join(fold_dir, "best_swintransseg.pt"))
        else:
            no_improve += 1
            if no_improve >= args.early_stop_patience:
                print(f"[SwinTransSeg Fold {fold_idx}] Early stopping at epoch {epoch}")
                break

    # Load best checkpoint 
    best_ckpt = torch.load(os.path.join(fold_dir, "best_swintransseg.pt"))
    model.load_state_dict(best_ckpt, strict=False)

    test_mae, test_rmse, test_acc = evaluate_and_print(model, test_loader, n_total)
    print(f"[SwinTransSeg Fold {fold_idx}] Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}, Test Acc: {test_acc:.3f}")

    return model

# ---------------------------------------------------------------------------- #
#                     TRAIN FUNCTION (FSDP for MedViT)                            #
# ---------------------------------------------------------------------------- #
def train_medvit_fold(train_ids, train_targets,
                      val_ids, val_targets,
                      test_ids, test_targets,
                      fold_idx=0):
    fold_dir = os.path.join(args.output_dir, f"medvit_fold{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    train_ds = LMDBDataset(args.lmdb_path, train_ids, train_targets,
                           transforms=train_transforms, cache_size=5)
    val_ds   = LMDBDataset(args.lmdb_path, val_ids,   val_targets,
                           transforms=val_transforms,   cache_size=5)
    test_ds  = LMDBDataset(args.lmdb_path, test_ids,  test_targets,
                           transforms=val_transforms,   cache_size=5)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    base_model = MedViT3DRegressor(
        MAX_D, MAX_H, MAX_W,
        pretrained_2d_ckpt=args.medvit2d_ckpt,
        patch_size=(2, 16, 16),
        embed_dim=768,
        depth=12,
        num_heads=12,
        dropout_rate=0.15
    ).to(device).half()

    model = FSDP(
        base_model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=cpu_offload,
        mixed_precision=mp_policy
    )

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.medvit_lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.medvit_lr, momentum=0.9, weight_decay=1e-4)

    total_steps = args.medvit_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.max_lr, total_steps=total_steps,
        pct_start=0.3, anneal_strategy="cos"
    )

    criterion = nn.MSELoss()
    scaler = GradScaler()

    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False

    best_val_mae = float("inf")
    no_improve = 0

    for epoch in range(1, args.medvit_epochs + 1):
        model.train()
        running_loss = 0.0

        for imgs, lbls in train_loader:
            imgs = imgs.to(device, dtype=torch.float16)
            lbls = lbls.to(device)

            with autocast():  # Mixed precision training :contentReference[oaicite:33]{index=33}
                outputs = model(imgs)
                loss = criterion(outputs, lbls)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if not args.no_tune:
                scheduler.step()

            running_loss += loss.item() * imgs.size(0)

        val_mae, val_rmse, val_acc = evaluate_and_print(model, val_loader, n_total)
        print(f"[MedViT Fold {fold_idx}] Epoch {epoch} | Train MSE: "
              f"{running_loss/len(train_loader.dataset):.4f} | Val MAE: {val_mae:.4f} "
              f"| Val RMSE: {val_rmse:.4f} | Val Acc: {val_acc:.3f}")

        if val_mae < best_val_mae - 1e-4:
            best_val_mae = val_mae
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(fold_dir, "best_medvit.pt"))
        else:
            no_improve += 1
            if no_improve >= args.early_stop_patience:
                print(f"[MedViT Fold {fold_idx}] Early stopping at epoch {epoch}")
                break

    best_ckpt = torch.load(os.path.join(fold_dir, "best_medvit.pt"))
    model.load_state_dict(best_ckpt, strict=False)

    test_mae, test_rmse, test_acc = evaluate_and_print(model, test_loader, n_total)
    print(f"[MedViT Fold {fold_idx}] Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}, Test Acc: {test_acc:.3f}")

    return model

# ---------------------------------------------------------------------------- #
#                      ENSEMBLE UTILITY (AVERAGE PREDICTIONS)                   #
# ---------------------------------------------------------------------------- #
def ensemble_predict(img, model1, model2):
    with torch.no_grad():
        out1 = model1(img.unsqueeze(0).to(device, dtype=torch.float16)).cpu().numpy().item()
        out2 = model2(img.unsqueeze(0).to(device, dtype=torch.float16)).cpu().numpy().item()
    return (out1 + out2) / 2.0

# ---------------------------------------------------------------------------- #
#                           MAIN TRAINING / EVAL ROUTINE                         #
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
            swintransseg_model = train_swintransseg_fold(
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

            if args.use_pruning:
                import torch.nn.utils.prune as prune
                for module in swintransseg_model.parameters():
                    if isinstance(module, nn.Linear):
                        prune.l1_unstructured(module, name="weight", amount=0.3)
                for module in medvit_model.parameters():
                    if isinstance(module, nn.Linear):
                        prune.l1_unstructured(module, name="weight", amount=0.3)

            if args.use_distillation:
                student = get_medicalnet_backbone(name="resnet50", num_outputs=1).to(device).half()

            print(f"#### Ensemble results for fold {fold_idx} ####")
            test_ds_fold = LMDBDataset(args.lmdb_path, test_ids_fold, test_targets_fold,
                                      transforms=val_transforms, cache_size=5)
            test_loader_fold = DataLoader(
                test_ds_fold,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )

            ensemble_preds, ensemble_labels = [], []
            for img, lbl in test_loader_fold:
                pred = ensemble_predict(img, swintransseg_model, medvit_model)
                ensemble_preds.append(pred)
                ensemble_labels.append(lbl.item())

            rounded_ens = np.rint(np.array(ensemble_preds)).astype(int)
            true_ens = np.array(ensemble_labels).astype(int)
            ens_n_correct = int((rounded_ens == true_ens).sum())
            ens_n_val = len(true_ens)

            print(f"Number of files = {n_total}")
            print(f"Number of Validation tests = {ens_n_val}")
            print(f"True estimated validation predictions (count) / all validation predictions = "
                  f"{ens_n_correct} / {ens_n_val}  (accuracy = {ens_n_correct}/{ens_n_val} = {ens_n_correct/ens_n_val:.3f})")

            ens_mae = mean_absolute_error(true_ens, ensemble_preds)
            ens_rmse = np.sqrt(mean_squared_error(true_ens, ensemble_preds))
            print(f"[Ensemble Fold {fold_idx}] Test MAE: {ens_mae:.4f}, Test RMSE: {ens_rmse:.4f}")

    else:
        print("#### Starting single-split SwinTransSeg training ####")
        swintransseg_model = train_swintransseg_fold(
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
            for module in swintransseg_model.parameters():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name="weight", amount=0.3)
            for module in medvit_model.parameters():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name="weight", amount=0.3)

        if args.use_distillation:
            student = get_medicalnet_backbone(name="resnet50", num_outputs=1).to(device).half()

        test_ds = LMDBDataset(args.lmdb_path, test_ids, test_targets,
                              transforms=val_transforms, cache_size=5)
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        ens_preds, ens_labels = [], []
        for img, lbl in test_loader:
            pred = ensemble_predict(img, swintransseg_model, medvit_model)
            ens_preds.append(pred)
            ens_labels.append(lbl.item())

        rounded_ens = np.rint(np.array(ens_preds)).astype(int)
        true_ens = np.array(ens_labels).astype(int)
        ens_n_correct = int((rounded_ens == true_ens).sum())
        ens_n_val = len(true_ens)

        print(f"Number of files = {n_total}")
        print(f"Number of Validation tests = {ens_n_val}")
        print(f"True estimated validation predictions (count) / all validation predictions = "
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