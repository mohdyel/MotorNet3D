#!/usr/bin/env python3
import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.nn import Parameter
from collections import OrderedDict

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions to inflate Conv2d→Conv3d, Linear→time-aware Linear, etc.
# ──────────────────────────────────────────────────────────────────────────────
def inflate_conv(conv2d,
                 time_dim=3,
                 time_padding=0,
                 time_stride=1,
                 time_dilation=1,
                 center=False):
    """
    Inflate a Conv2d into a Conv3d by replicating (or centering) the 2D kernels
    across the temporal dimension.
    """
    kernel_dim = (time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1])
    padding    = (time_padding, conv2d.padding[0], conv2d.padding[1])
    stride     = (time_stride, conv2d.stride[0], conv2d.stride[1])
    dilation   = (time_dilation, conv2d.dilation[0], conv2d.dilation[1])

    conv3d = torch.nn.Conv3d(
        in_channels=conv2d.in_channels,
        out_channels=conv2d.out_channels,
        kernel_size=kernel_dim,
        padding=padding,
        stride=stride,
        dilation=dilation
    )

    weight_2d = conv2d.weight.data  # shape: (out_c, in_c, H, W)
    if center:
        # Create a 5D tensor of zeros, then place the 2D weights in the center time slice
        weight_3d = torch.zeros(
            (weight_2d.shape[0],
             weight_2d.shape[1],
             time_dim,
             weight_2d.shape[2],
             weight_2d.shape[3]),
            dtype=weight_2d.dtype,
            device=weight_2d.device
        )
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        # Replicate the 2D filter across time_dim, then normalize by time_dim
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / float(time_dim)

    conv3d.weight = Parameter(weight_3d)
    if conv2d.bias is not None:
        conv3d.bias = conv2d.bias
    return conv3d


def inflate_linear(linear2d, time_dim):
    """
    Inflate a Linear layer (2D) into one that accounts for time_dim:
      new Linear has in_features = original_in * time_dim, out_features = original_out.
    """
    linear3d = torch.nn.Linear(linear2d.in_features * time_dim,
                               linear2d.out_features)
    weight2d = linear2d.weight.data  # shape: (out, in)
    # Repeat each column time_dim times, then divide by time_dim
    weight3d = weight2d.repeat(1, time_dim) / float(time_dim)
    linear3d.weight = Parameter(weight3d)
    linear3d.bias   = linear2d.bias
    return linear3d


def inflate_batch_norm(batch2d):
    """
    Convert a 2D BatchNorm into a 3D BatchNorm by reusing num_features.
    (We only override the _check_input_dim to accept 3D inputs.)
    """
    batch3d = torch.nn.BatchNorm3d(batch2d.num_features)
    batch2d._check_input_dim = batch3d._check_input_dim
    return batch2d


def inflate_pool(pool2d,
                 time_dim=1,
                 time_padding=0,
                 time_stride=None,
                 time_dilation=1):
    """
    Inflate a pooling layer (MaxPool2d / AvgPool2d / AdaptiveAvgPool2d) into 3D.
    """
    if isinstance(pool2d, torch.nn.AdaptiveAvgPool2d):
        # Always project to (T=1, H=1, W=1)
        return torch.nn.AdaptiveAvgPool3d((1, 1, 1))

    kernel_dim = (time_dim, pool2d.kernel_size, pool2d.kernel_size)
    padding    = (time_padding, pool2d.padding, pool2d.padding)
    if time_stride is None:
        time_stride = time_dim
    stride = (time_stride, pool2d.stride, pool2d.stride)

    if isinstance(pool2d, torch.nn.MaxPool2d):
        dilation = (time_dilation, pool2d.dilation, pool2d.dilation)
        return torch.nn.MaxPool3d(
            kernel_size=kernel_dim,
            padding=padding,
            dilation=dilation,
            stride=stride,
            ceil_mode=pool2d.ceil_mode
        )
    elif isinstance(pool2d, torch.nn.AvgPool2d):
        return torch.nn.AvgPool3d(kernel_size=kernel_dim, stride=stride)
    else:
        raise ValueError(f"{type(pool2d)} is not a known 2D pooling layer.")


# ──────────────────────────────────────────────────────────────────────────────
# Define a 3D Vision Transformer (all in one file, no external imports).
# ──────────────────────────────────────────────────────────────────────────────
class PatchEmbed3D(nn.Module):
    """
    3D patch embedding using a Conv3d: 
    splits the input 5D volume into non-overlapping (temporal x H x W) patches,
    then projects each patch to 'embed_dim' channels.
    """
    def __init__(self, img_size=(3, 224, 224),
                       patch_size=(2, 16, 16),
                       in_chans=3,
                       embed_dim=768):
        super().__init__()
        self.img_size   = img_size
        self.patch_size = patch_size
        # Calculate number of patches per dimension
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2]
        )
        self.num_patches = (self.grid_size[0] *
                            self.grid_size[1] *
                            self.grid_size[2])
        # Conv3d to extract patch embeddings
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # Input x shape: (B, in_chans, T, H, W)
        x = self.proj(x)  # (B, embed_dim, T/pt, H/ph, W/pw)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class Attention3D(nn.Module):
    """
    Multi-head self-attention for 3D patches.
    We explicitly implement qkv as a single Linear, then split.
    This matches many 2D-ViT implementations that use a single "qkv" Linear.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Combine query, key, and value projections into one Linear
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x shape: (B, N, C)
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B, N, 3C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]   # each (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v)  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MLP3D(nn.Module):
    """
    Feed-forward network (MLP) inside each transformer block.
    """
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block3D(nn.Module):
    """
    Transformer encoder block: 
      - LayerNorm → Attention3D → residual
      - LayerNorm → MLP3D → residual
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0,
                 qkv_bias=True, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention3D(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        # We’ll skip stochastic depth (drop_path) for simplicity; use Identity
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP3D(
            in_features=dim,
            hidden_features=hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer3D(nn.Module):
    """
    A 3D Vision Transformer:
      - PatchEmbed3D to split (C, T, H, W) into patches
      - cls_token + positional embeddings
      - A stack of transformer blocks (Block3D)
      - LayerNorm + classification head
    """
    def __init__(self,
                 img_size=(3, 224, 224),
                 patch_size=(2, 16, 16),
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 num_classes=1000):
        super().__init__()
        # 1) Patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # 2) Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)

        # 3) Transformer blocks
        self.blocks = nn.ModuleList([
            Block3D(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=0.0,
                attn_drop=0.0
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 4) Classification head (optional, needed to match the state_dict keys)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize parameters
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x shape: (B, in_chans, T, H, W)
        B = x.shape[0]
        x = self.patch_embed(x)              # (B, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)          # (B, 1 + num_patches, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]             # (B, embed_dim)
        out = self.head(cls_out)      # (B, num_classes)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Main: Parse arguments, load 2D checkpoint, inflate into 3D, save new checkpoint
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inflate a 2D ViT checkpoint into a 3D ViT checkpoint"
    )
    parser.add_argument(
        "--pretrained_2d",
        type=str,
        required=True,
        help="Path to the 2D ViT checkpoint (a .pth containing state_dict) to inflate."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path where the 3D-inflated checkpoint will be saved (e.g. vit_base_3d.pth)."
    )
    parser.add_argument(
        "--time_dim",
        type=int,
        default=3,
        help="Temporal kernel size used when inflating Conv2d → Conv3d (default=3)."
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="If set, place 2D weights only in the center time slice (instead of uniform replication)."
    )
    args = parser.parse_args()

    # 1) Load the 2D checkpoint (state_dict)
    state2d = torch.load(args.pretrained_2d, map_location="cpu")
    if "state_dict" in state2d:
        state2d = state2d["state_dict"]

    # 2) Instantiate the 3D ViT with the same hyperparameters as the 2D version,
    #    except injecting time_dim into img_size and patch_size. Adjust as needed.
    vit3d = VisionTransformer3D(
        img_size=(args.time_dim, 224, 224),    # e.g., time_dim × 224 × 224
        patch_size=(args.time_dim, 16, 16),    # e.g., (temporal, H, W)
        in_chans=3,                            # match the 2D ViT’s in_channels
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        num_classes=1000                       # must match the original 2D ViT’s head size
    )

    # 3) Build a new state_dict by “inflating” each 2D layer into 3D
    sd3 = vit3d.state_dict()
    new_state3d = OrderedDict()

    for name, param3d in sd3.items():
        # 3.a) If there’s an exact‐match key in the 2D checkpoint AND shapes match, copy directly
        if (name in state2d) and (state2d[name].shape == param3d.shape):
            new_state3d[name] = state2d[name].clone()
            continue

        # 3.b) If this is a Conv3d weight (5D) and there’s a same‐named 2D Conv2d (4D) in state2d:
        if name.endswith(".weight") and param3d.ndim == 5 and (name in state2d):
            w2d = state2d[name]  # shape: (out_c, in_c, H, W)
            conv2d = torch.nn.Conv2d(
                in_channels=w2d.shape[1],
                out_channels=w2d.shape[0],
                kernel_size=w2d.shape[2:]
            )
            conv2d.weight = Parameter(w2d)
            conv2d.bias = None
            conv3d = inflate_conv(
                conv2d,
                time_dim=args.time_dim,
                center=args.center
            )
            new_state3d[name] = conv3d.weight.data.clone()
            continue

        # 3.c) If this is a Linear weight (2D) and the target param3d is also 2D,
        #       then inflate (in_features→ in_features * time_dim)
        if name.endswith(".weight") and param3d.ndim == 2 and (name in state2d):
            w2d = state2d[name]  # shape: (out, in)
            linear2d = torch.nn.Linear(w2d.size(1), w2d.size(0))
            linear2d.weight = Parameter(w2d)
            linear2d.bias = None
            lin3d = inflate_linear(linear2d, args.time_dim)
            new_state3d[name] = lin3d.weight.data.clone()
            continue

        # 3.d) Copy any bias if it exists in the 2D checkpoint
        if name.endswith(".bias") and (name in state2d) and (state2d[name].shape == param3d.shape):
            new_state3d[name] = state2d[name].clone()
            continue

        # 3.e) If this is a BatchNorm parameter (“bn”) and exists in 2D, copy it
        if ("bn" in name) and (name in state2d) and (state2d[name].shape == param3d.shape):
            new_state3d[name] = state2d[name].clone()
            continue

        # 3.f) Otherwise, leave the randomly initialized 3D weight as is
        new_state3d[name] = param3d

    # 4) Load these inflated weights into vit3d (strict=False allows mismatches)
    missing, unexpected = vit3d.load_state_dict(new_state3d, strict=False)
    if missing or unexpected:
        print("Warning: Some keys did not exactly match:")
        if missing:
            print("  Missing keys in the 3D model:", missing)
        if unexpected:
            print("  Unexpected keys in the 3D model:", unexpected)

    # 5) Save the new 3D-inflated checkpoint
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(vit3d.state_dict(), args.output)
    print(f"Success: 3D checkpoint saved to {args.output}")
