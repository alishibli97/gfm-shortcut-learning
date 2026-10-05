#!/usr/bin/env python3
"""Validate completeness and numerical integrity of REO embedding outputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def validate(dataset, root, expected_rows, models):
    rows = []
    for model in models:
        model_root = root / model
        row = {"dataset": dataset, "model": model, "root": str(model_root), "ok": False}
        try:
            required = {
                "pre": model_root / "pre_embeddings.npy",
                "post": model_root / "post_embeddings.npy",
                "manifest": model_root / "manifest.csv",
                "preprocessing": model_root / "preprocessing.json",
            }
            missing = [name for name, path in required.items() if not path.is_file()]
            if missing:
                raise FileNotFoundError("missing " + ", ".join(missing))
            pre = np.load(required["pre"], mmap_mode="r")
            post = np.load(required["post"], mmap_mode="r")
            manifest = pd.read_csv(required["manifest"])
            with open(required["preprocessing"]) as handle:
                preprocessing = json.load(handle)
            if pre.ndim != 2 or post.ndim != 2 or pre.shape != post.shape:
                raise RuntimeError(f"shape mismatch {pre.shape} / {post.shape}")
            if len(manifest) != len(pre) or len(pre) != expected_rows:
                raise RuntimeError(
                    f"row mismatch expected={expected_rows} manifest={len(manifest)} embeddings={len(pre)}"
                )
            if not np.isfinite(pre).all() or not np.isfinite(post).all():
                raise RuntimeError("NaN or Inf in embeddings")
            if float(np.std(pre)) == 0.0 or float(np.std(post)) == 0.0:
                raise RuntimeError("constant embeddings")
            row.update({
                "ok": True,
                "n": int(len(pre)),
                "dim": int(pre.shape[1]),
                "pre_mean": float(np.mean(pre)),
                "pre_std": float(np.std(pre)),
                "post_mean": float(np.mean(post)),
                "post_std": float(np.std(post)),
                "preprocessing_profile": preprocessing["preprocessing"]["name"],
            })
        except Exception as error:
            row["error"] = str(error)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xview-root", type=Path, required=True)
    parser.add_argument("--bright-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["prithvi", "terramind", "dofa_base", "dinov3"])
    parser.add_argument("--out", type=Path, default=Path("runs/embedding_validation.csv"))
    args = parser.parse_args()

    rows = []
    rows.extend(validate("xview2", args.xview_root, 9168, args.models))
    rows.extend(validate("bright", args.bright_root, 3029, args.models))
    output = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(output.to_string(index=False))
    print(f"Saved {args.out}")
    if not output["ok"].all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
