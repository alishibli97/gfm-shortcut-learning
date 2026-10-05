#!/usr/bin/env python3
"""
extract_xview2_embeddings.py

Extract frozen GFM embeddings for xView2/xBD pre/post image pairs.

Input:
  data/xview2_full19_manifest_with_splits.csv

Output:
  embeddings/xview2_full19/<model>/
    pre_embeddings.npy
    post_embeddings.npy
    manifest.csv

Example:

python extract_xview2_embeddings.py \
  --manifest data/xview2_full19_manifest_with_splits.csv \
  --model prithvi \
  --out-dir embeddings/xview2_full19 \
  --batch-size 32

DINOv3:

python extract_xview2_embeddings.py \
  --manifest data/xview2_full19_manifest_with_splits.csv \
  --model dinov3 \
  --dinov3-arch dinov3_vitl16 \
  --dinov3-repo /path/to/local/dinov3 \
  --dinov3-weights /path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --out-dir embeddings/xview2_full19 \
  --batch-size 32
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from gfm_preprocessing import RGBPreprocessor


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class XView2PairDataset(Dataset):
    def __init__(
        self,
        manifest_csv,
        model_name,
        image_size=224,
        labeled_only=True,
        max_samples=None,
    ):
        self.df = pd.read_csv(manifest_csv)

        if labeled_only and "has_labels" in self.df.columns:
            self.df = self.df[self.df["has_labels"] == 1].reset_index(drop=True)

        if max_samples is not None:
            self.df = self.df.iloc[:max_samples].reset_index(drop=True)

        self.image_size = image_size
        self.preprocess = RGBPreprocessor(model_name)

        required = ["pre_image", "post_image"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise RuntimeError(f"Manifest missing columns: {missing}")

    def __len__(self):
        return len(self.df)

    def load_rgb(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)

        return self.preprocess(np.asarray(img))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        pre = self.load_rgb(row["pre_image"])
        post = self.load_rgb(row["post_image"])

        return {
            "idx": idx,
            "pre": pre,
            "post": post,
        }


# ---------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------

def freeze(model):
    model.eval()
    model.requires_grad_(False)
    return model


def build_prithvi(device):
    from terratorch import BACKBONE_REGISTRY

    model = BACKBONE_REGISTRY.build(
        "prithvi_eo_v2_300",
        pretrained=True,
        bands=["RED", "GREEN", "BLUE"],
        num_frames=1,
    ).to(device)

    model = freeze(model)
    sel = [5, 11, 17, 23]

    @torch.inference_mode()
    def extract(x):
        outs = model(x)
        emb = torch.cat([outs[i][:, 1:, :].mean(1) for i in sel], dim=-1)
        return emb

    return model, extract, 4096


def build_terramind(device):
    from terratorch import BACKBONE_REGISTRY

    model = BACKBONE_REGISTRY.build(
        "terramind_v1_base",
        pretrained=True,
        modalities=["RGB"],
    ).to(device)

    model = freeze(model)
    sel = [2, 5, 8, 11]

    @torch.inference_mode()
    def extract(x):
        try:
            outs = model(x)
        except Exception:
            outs = model({"RGB": x})

        emb = torch.cat([outs[i].mean(1) for i in sel], dim=-1)
        return emb

    return model, extract, 3072


def build_dofa_base(device):
    from terratorch import BACKBONE_REGISTRY

    sel = [2, 5, 8, 11]

    model = BACKBONE_REGISTRY.build(
        "dofa_base_patch16_224",
        pretrained=True,
        model_bands=["RED", "GREEN", "BLUE"],
        out_indices=sel,
    ).to(device)

    model = freeze(model)

    @torch.inference_mode()
    def extract(x):
        outs = model(x)
        emb = torch.cat([t[:, 1:, :].mean(1) for t in outs], dim=-1)
        return emb

    return model, extract, 3072


def build_dofa_large(device):
    from terratorch import BACKBONE_REGISTRY

    sel = [5, 11, 17, 23]

    model = BACKBONE_REGISTRY.build(
        "dofa_large_patch16_224",
        pretrained=True,
        model_bands=["RED", "GREEN", "BLUE"],
        out_indices=sel,
    ).to(device)

    model = freeze(model)

    @torch.inference_mode()
    def extract(x):
        outs = model(x)
        emb = torch.cat([t[:, 1:, :].mean(1) for t in outs], dim=-1)
        return emb

    return model, extract, 4096


def build_dinov3(device, weights, arch, repo=None):
    if weights is None:
        raise RuntimeError("DINOv3 requires --dinov3-weights")

    if not os.path.exists(weights):
        raise FileNotFoundError(weights)

    if repo is not None:
        if not os.path.isdir(repo):
            raise FileNotFoundError(f"DINOv3 repository not found: {repo}")
        model = torch.hub.load(
            repo,
            arch,
            source="local",
            weights=weights,
        ).to(device)
    else:
        model = torch.hub.load(
            "facebookresearch/dinov3",
            arch,
            weights=weights,
        ).to(device)

    model = freeze(model)

    if "vitl" in arch.lower():
        sel = [5, 11, 17, 23]
    else:
        sel = [2, 5, 8, 11]

    # dim is inferred after first batch
    expected_dim = None

    @torch.inference_mode()
    def extract(x):
        feats = model.get_intermediate_layers(
            x,
            n=sel,
            return_class_token=False,
            norm=True,
        )
        emb = torch.cat([f.mean(1) for f in feats], dim=-1)
        return emb

    return model, extract, expected_dim


def build_model(args, device):
    if args.model == "prithvi":
        return build_prithvi(device)
    if args.model == "terramind":
        return build_terramind(device)
    if args.model == "dofa_base":
        return build_dofa_base(device)
    if args.model == "dofa_large":
        return build_dofa_large(device)
    if args.model == "dinov3":
        return build_dinov3(
            device,
            args.dinov3_weights,
            args.dinov3_arch,
            args.dinov3_repo,
        )

    raise ValueError(f"Unknown model: {args.model}")


# ---------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--manifest", required=True)
    ap.add_argument(
        "--model",
        required=True,
        choices=["prithvi", "terramind", "dofa_base", "dofa_large", "dinov3"],
    )
    ap.add_argument("--out-dir", default="embeddings/xview2_full19")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--labeled-only", action="store_true", default=True)

    ap.add_argument("--dinov3-weights", default=None)
    ap.add_argument("--dinov3-arch", default="dinov3_vitl16")
    ap.add_argument("--dinov3-repo", default=None)

    args = ap.parse_args()

    print("=" * 80)
    print("xView2 embedding extraction")
    print("=" * 80)
    print("manifest:", args.manifest)
    print("model:", args.model)
    print("device:", args.device)
    print("batch size:", args.batch_size)

    dataset = XView2PairDataset(
        args.manifest,
        model_name=args.model,
        image_size=args.image_size,
        labeled_only=args.labeled_only,
        max_samples=args.max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print("num samples:", len(dataset))

    model, extract, expected_dim = build_model(args, args.device)

    out_root = Path(args.out_dir) / args.model
    out_root.mkdir(parents=True, exist_ok=True)

    pre_chunks = []
    post_chunks = []
    indices = []
    first_batch_input_stats = {}

    for batch in tqdm(loader, desc=f"extract {args.model}"):
        idx = batch["idx"].numpy()
        pre = batch["pre"].to(args.device, non_blocking=True)
        post = batch["post"].to(args.device, non_blocking=True)

        if not first_batch_input_stats:
            for name, tensor in (("pre", pre), ("post", post)):
                first_batch_input_stats[name] = {
                    "min": float(tensor.min().item()),
                    "max": float(tensor.max().item()),
                    "mean": float(tensor.mean().item()),
                    "std": float(tensor.std().item()),
                }
            print("first standardized input batch:", first_batch_input_stats)

        pre_emb = extract(pre).detach().cpu().float().numpy()
        post_emb = extract(post).detach().cpu().float().numpy()

        pre_chunks.append(pre_emb)
        post_chunks.append(post_emb)
        indices.append(idx)

    pre_all = np.concatenate(pre_chunks, axis=0)
    post_all = np.concatenate(post_chunks, axis=0)
    idx_all = np.concatenate(indices, axis=0)

    print("pre embeddings:", pre_all.shape)
    print("post embeddings:", post_all.shape)

    if expected_dim is not None and pre_all.shape[1] != expected_dim:
        raise RuntimeError(
            f"Expected embedding dim {expected_dim}, got {pre_all.shape[1]}"
        )

    # Keep manifest in exact embedding order.
    manifest_out = dataset.df.iloc[idx_all].reset_index(drop=True)

    np.save(out_root / "pre_embeddings.npy", pre_all)
    np.save(out_root / "post_embeddings.npy", post_all)
    manifest_out.to_csv(out_root / "manifest.csv", index=False)

    metadata = {
        "dataset": "xview2",
        "model": args.model,
        "dinov3_arch": args.dinov3_arch if args.model == "dinov3" else None,
        "dinov3_weights": args.dinov3_weights if args.model == "dinov3" else None,
        "image_size": args.image_size,
        "resize": "Pillow bicubic square resize",
        "num_samples": len(manifest_out),
        "pre_shape": list(pre_all.shape),
        "post_shape": list(post_all.shape),
        "preprocessing": dataset.preprocess.metadata(),
        "first_batch_standardized_input_stats": first_batch_input_stats,
    }
    with open(out_root / "preprocessing.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Small metadata file
    with open(out_root / "README.txt", "w") as f:
        f.write(f"model: {args.model}\n")
        f.write(f"manifest: {args.manifest}\n")
        f.write(f"num_samples: {len(manifest_out)}\n")
        f.write(f"pre_shape: {pre_all.shape}\n")
        f.write(f"post_shape: {post_all.shape}\n")
        f.write(f"preprocessing: {dataset.preprocess.profile.name}\n")
        f.write("Embeddings are raw frozen features. Normalize/standardize during probing.\n")

    print("\nSaved:")
    print(out_root / "pre_embeddings.npy")
    print(out_root / "post_embeddings.npy")
    print(out_root / "manifest.csv")
    print(out_root / "preprocessing.json")
    print(out_root / "README.txt")


if __name__ == "__main__":
    main()
