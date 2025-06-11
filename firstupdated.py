#!/usr/bin/env python3
import os
import argparse
import json
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import monai
from monai.transforms import (
    LoadImaged, ScaleIntensityRanged, RandFlipd,
    RandRotate90d, RandGaussianNoised, ToTensord,
    Compose, Lambdad
)
from monai.data import Dataset
import optuna

# ---------------------------------------------------------------------------- #
#                             ARGPARSE CONFIGURATION                            #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="Train 3D Tomogram Classifier")
parser.add_argument("--train_csv", type=str, default="train_labels.csv")
parser.add_argument("--train_dir", type=str, default="byu-locating-bacterial-flagellar-motors-2025/train/")
parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument("--arch", type=str, choices=["resnet50", "densenet121"], default="resnet50")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--optimizer", type=str, choices=["adam","sgd"], default="adam")
parser.add_argument("--no_tune", action="store_true", help="Skip Optuna tuning")
args, _ = parser.parse_known_args()

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
monai.utils.set_determinism(seed=42)

# ---------------------------------------------------------------------------- #
#                              LOAD LABELS & SPLIT                              #
# ---------------------------------------------------------------------------- #
df = pd.read_csv(args.train_csv)
le = LabelEncoder()
df["label"] = le.fit_transform(df["Number of motors"])
labels = df["label"].values
ids = df["tomo_id"].values

sss = StratifiedShuffleSplit(n_splits=1, test_size=140, random_state=42)
train_idx, val_idx = next(sss.split(ids, labels))
train_ids, train_labels = ids[train_idx], labels[train_idx]
val_ids, val_labels     = ids[val_idx], labels[val_idx]

with open(os.path.join(args.output_dir, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

# ---------------------------------------------------------------------------- #
#                             MONAI TRANSFORMS                                  #
# ---------------------------------------------------------------------------- #
train_transforms = Compose([
    LoadImaged(keys="image", reader="PILReader"),
    # stacks N slices into shape (1, N, H, W)
    Lambdad(
        keys="image",
        func=lambda imgs: np.expand_dims(np.stack(imgs, axis=0), axis=0)
    ),
    ScaleIntensityRanged(keys="image", a_min=0, a_max=255,
                         b_min=0.0, b_max=1.0, clip=True),
    RandFlipd(keys="image", prob=0.5, spatial_axis=0),
    RandFlipd(keys="image", prob=0.5, spatial_axis=1),
    RandFlipd(keys="image", prob=0.5, spatial_axis=2),
    RandRotate90d(keys="image", prob=0.5,
                  max_k=3, spatial_axes=(1,2)),
    RandGaussianNoised(keys="image", prob=0.2,
                       mean=0.0, std=0.1),
    ToTensord(keys=["image","label"]),
])

val_transforms = Compose([
    LoadImaged(keys="image", reader="PILReader"),
    Lambdad(
        keys="image",
        func=lambda imgs: np.expand_dims(np.stack(imgs, axis=0), axis=0)
    ),
    ScaleIntensityRanged(keys="image", a_min=0, a_max=255,
                         b_min=0.0, b_max=1.0, clip=True),
    ToTensord(keys=["image","label"]),
])

def make_dataset(tomo_ids, tomo_labels):
    data = []
    for tid, lbl in zip(tomo_ids, tomo_labels):
        img_dir = os.path.join(args.train_dir, tid)
        slices = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith(".jpg")
        ])
        data.append({"image": slices, "label": lbl})
    return data

train_ds = Dataset(data=make_dataset(train_ids, train_labels), transform=train_transforms)
val_ds   = Dataset(data=make_dataset(val_ids,   val_labels),   transform=val_transforms)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=1,               shuffle=False,
                          num_workers=2, pin_memory=True)

# ---------------------------------------------------------------------------- #
#                            MODEL CONSTRUCTION                                #
# ---------------------------------------------------------------------------- #
def build_model(arch: str, num_classes: int):
    if arch == "resnet50":
        return monai.networks.nets.resnet50(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=num_classes
        )
    else:  # densenet121
        return monai.networks.nets.DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes
        )

# ---------------------------------------------------------------------------- #
#                            TRAIN / VAL FUNCTIONS                             #
# ---------------------------------------------------------------------------- #
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    running_loss = 0.0
    for batch in loader:
        imgs, lbls = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(imgs)
            loss = criterion(logits, lbls)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def eval_one_epoch(model, loader, criterion):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            imgs, lbls = batch["image"].to(device), batch["label"].to(device)
            preds = torch.argmax(model(imgs), dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
    return correct / total

# ---------------------------------------------------------------------------- #
#                      OPTUNA HYPERPARAMETER TUNING                            #
# ---------------------------------------------------------------------------- #
def objective(trial):
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    opt_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    arch = trial.suggest_categorical("arch", ["resnet50", "densenet121"])
    model = build_model(arch, num_classes=len(le.classes_)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr) if opt_name=="adam" \
                else optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    for _ in range(5):
        train_one_epoch(model, train_loader, optimizer, criterion, scaler)
    return eval_one_epoch(model, val_loader, criterion)

if not args.no_tune:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    best = study.best_params
    print("Best hyperparameters:", best)
    args.lr, args.optimizer, args.arch = best["lr"], best["optimizer"], best["arch"]

with open(os.path.join(args.output_dir, "config.json"), "w") as f:
    json.dump(vars(args), f, indent=2)

# ---------------------------------------------------------------------------- #
#                       FINAL MODEL TRAINING & CHECKPOINT                       #
# ---------------------------------------------------------------------------- #
model = build_model(args.arch, num_classes=len(le.classes_)).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr) if args.optimizer=="adam" \
            else optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()

best_acc = 0.0
for epoch in range(1, args.epochs + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
    val_acc = eval_one_epoch(model, val_loader, criterion)
    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))

print(f"Training complete. Best validation accuracy: {best_acc:.4f}")
torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pth"))
