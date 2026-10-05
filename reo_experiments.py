#!/usr/bin/env python3
"""Rigorous REO audit and mitigation experiments on frozen paired embeddings.

The script is intentionally dataset-light: both xView2 and BRIGHT embedding
directories contain pre_embeddings.npy, post_embeddings.npy, and manifest.csv.
It evaluates an IID raw-feature baseline, macro leave-one-event-out transfer,
and event-identity recoverability with a damage-balanced diagnostic.

Mitigation controls:
  raw          standardized frozen concat features
  pca          a 256-D unsupervised bottleneck
  ce_mlp       same-capacity supervised MLP trained with cross entropy
  supcon       ordinary supervised contrastive alignment
  cross_event  same-label, different-event supervised contrastive alignment

The held-out event is never used for scaling, representation training, or the
damage classifier. Learned methods are repeated for every requested seed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


LEARNED_METHODS = {"ce_mlp", "supcon", "cross_event"}
ALL_METHODS = ["raw", "pca", "ce_mlp", "supcon", "cross_event"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_embeddings(emb_root: Path, model: str):
    root = emb_root / model
    required = [
        root / "pre_embeddings.npy",
        root / "post_embeddings.npy",
        root / "manifest.csv",
        root / "preprocessing.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing embedding outputs:\n" + "\n".join(missing))

    pre = np.load(required[0], mmap_mode="r")
    post = np.load(required[1], mmap_mode="r")
    meta = pd.read_csv(required[2])

    if pre.ndim != 2 or post.ndim != 2 or pre.shape != post.shape:
        raise RuntimeError(f"Invalid pre/post shapes: {pre.shape} / {post.shape}")
    if len(meta) != len(pre):
        raise RuntimeError(f"Row mismatch: manifest={len(meta)}, embeddings={len(pre)}")
    for column in ("event", "damage_present"):
        if column not in meta.columns:
            raise RuntimeError(f"manifest.csv is missing {column!r}")

    pre = np.asarray(pre, dtype=np.float32)
    post = np.asarray(post, dtype=np.float32)
    if not np.isfinite(pre).all() or not np.isfinite(post).all():
        raise RuntimeError(f"NaN or Inf found under {root}")

    features = np.concatenate([pre, post, post - pre], axis=1).astype(np.float32)
    labels = meta["damage_present"].to_numpy(dtype=np.int64)
    events = meta["event"].astype(str).to_numpy()
    return features, labels, events, meta


def get_iid_indices(meta: pd.DataFrame, split_csv: str | None, seed: int):
    if split_csv:
        split_df = pd.read_csv(split_csv)
        id_column = "sample_id" if "sample_id" in meta.columns else "example_id"
        if id_column not in split_df.columns or "split" not in split_df.columns:
            raise RuntimeError(f"{split_csv} must contain {id_column!r} and 'split'")
        mapping = dict(zip(split_df[id_column].astype(str), split_df["split"]))
        split = meta[id_column].astype(str).map(mapping)
        if split.isna().any():
            raise RuntimeError(f"{split_csv} does not cover the embedding manifest")
        return np.where(split.to_numpy() == "train")[0], np.where(split.to_numpy() == "test")[0]

    if "random_split" in meta.columns:
        split = meta["random_split"].astype(str).to_numpy()
        return np.where(split == "train_random")[0], np.where(split == "test_random")[0]

    idx = np.arange(len(meta))
    return train_test_split(
        idx,
        test_size=0.2,
        random_state=seed,
        stratify=meta["damage_present"].to_numpy(),
    )


def binary_probe(X_train, y_train, X_test, y_test, seed: int, C: float):
    clf = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        C=C,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    score = clf.decision_function(X_test)

    out = {
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "accuracy": float(accuracy_score(y_test, pred)),
    }
    out["roc_auc"] = (
        float(roc_auc_score(y_test, score)) if len(np.unique(y_test)) == 2 else np.nan
    )
    out["average_precision"] = (
        float(average_precision_score(y_test, score))
        if len(np.unique(y_test)) == 2
        else np.nan
    )
    return out


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class CEModel(nn.Module):
    def __init__(self, head: ProjectionHead, out_dim: int):
        super().__init__()
        self.head = head
        self.classifier = nn.Linear(out_dim, 1)

    def forward(self, x):
        z = self.head(x)
        return z, self.classifier(z).squeeze(1)


def event_balanced_batch(y: np.ndarray, event_codes: np.ndarray, batch_size: int, rng):
    """Balance damage labels and sample events uniformly within each label."""
    chosen = []
    per_label = [batch_size // 2, batch_size - batch_size // 2]
    for label, count in enumerate(per_label):
        label_events = np.unique(event_codes[y == label])
        if len(label_events) == 0:
            continue
        sampled_events = rng.choice(label_events, size=count, replace=True)
        for event in sampled_events:
            cell = np.where((y == label) & (event_codes == event))[0]
            chosen.append(int(rng.choice(cell)))
    if not chosen:
        raise RuntimeError("Cannot construct a balanced batch")
    chosen = np.asarray(chosen, dtype=np.int64)
    rng.shuffle(chosen)
    return chosen


def contrastive_loss(z, y, events, mode: str, temperature: float):
    n = len(z)
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    same_label = y[:, None].eq(y[None, :])
    different_event = events[:, None].ne(events[None, :])

    if mode == "supcon":
        positive = same_label & ~self_mask
        denominator = ~self_mask
    elif mode == "cross_event":
        positive = same_label & different_event & ~self_mask
        # Same-label/same-event pairs are ignored, not treated as negatives.
        denominator = ((~same_label) | positive) & ~self_mask
    else:
        raise ValueError(mode)

    logits = z @ z.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * denominator.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    counts = positive.sum(dim=1)
    valid = counts > 0
    if not valid.any():
        raise RuntimeError("Batch has no valid positive pairs")
    per_anchor = (positive.float() * log_prob).sum(dim=1) / counts.clamp_min(1)
    return -per_anchor[valid].mean()


def train_head(X, y, events, method: str, args, seed: int, device):
    seed_everything(seed)
    event_encoder = LabelEncoder()
    event_codes = event_encoder.fit_transform(events).astype(np.int64)
    rng = np.random.default_rng(seed)

    X_tensor = torch.from_numpy(X).float().to(device)
    y_tensor = torch.from_numpy(y.astype(np.int64)).long().to(device)
    e_tensor = torch.from_numpy(event_codes).long().to(device)
    head = ProjectionHead(X.shape[1], args.hidden_dim, args.out_dim, args.dropout).to(device)

    if method == "ce_mlp":
        trainable = CEModel(head, args.out_dim).to(device)
    else:
        trainable = head
    optimizer = torch.optim.AdamW(
        trainable.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    trainable.train()
    for _epoch in range(args.epochs):
        for _step in range(args.steps_per_epoch):
            idx = event_balanced_batch(y, event_codes, args.batch_size, rng)
            idx_t = torch.from_numpy(idx).long().to(device)
            xb, yb, eb = X_tensor[idx_t], y_tensor[idx_t], e_tensor[idx_t]

            if method == "ce_mlp":
                _, logits = trainable(xb)
                loss = F.binary_cross_entropy_with_logits(logits, yb.float())
            else:
                z = trainable(xb)
                loss = contrastive_loss(z, yb, eb, method, args.temperature)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return head


@torch.no_grad()
def transform_head(head, X, device, batch_size: int):
    head.eval()
    outputs = []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start : start + batch_size]).float().to(device)
        outputs.append(head(batch).cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def transform_method(method, X_train, X_test, y_train, events_train, args, seed, device):
    if method == "raw":
        return X_train, X_test
    if method == "pca":
        n_components = min(args.out_dim, X_train.shape[0] - 1, X_train.shape[1])
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
        return (
            pca.fit_transform(X_train).astype(np.float32),
            pca.transform(X_test).astype(np.float32),
        )
    if method in LEARNED_METHODS:
        head = train_head(X_train, y_train, events_train, method, args, seed, device)
        return (
            transform_head(head, X_train, device, args.transform_batch_size),
            transform_head(head, X_test, device, args.transform_batch_size),
        )
    raise ValueError(method)


def fit_transform_fold(X, train_idx, test_idx):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
    X_test = scaler.transform(X[test_idx]).astype(np.float32)
    return X_train, X_test


def cell_balance_weights(events, damage):
    keys = np.asarray([f"{e}::{int(y)}" for e, y in zip(events, damage)])
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    return weights / weights.mean()


def event_probe(X_train, e_train, d_train, X_test, e_test, d_test, seed: int):
    encoder = LabelEncoder().fit(e_train)
    y_train = encoder.transform(e_train)
    y_test = encoder.transform(e_test)

    ordinary = LogisticRegression(
        solver="lbfgs", class_weight="balanced", max_iter=2000, random_state=seed
    )
    ordinary.fit(X_train, y_train)
    ordinary_pred = ordinary.predict(X_test)

    train_weights = cell_balance_weights(e_train, d_train)
    test_weights = cell_balance_weights(e_test, d_test)
    conditional = LogisticRegression(
        solver="lbfgs", max_iter=2000, random_state=seed
    )
    conditional.fit(X_train, y_train, sample_weight=train_weights)
    conditional_pred = conditional.predict(X_test)

    return {
        "event_accuracy": float(accuracy_score(y_test, ordinary_pred)),
        "event_balanced_accuracy": float(balanced_accuracy_score(y_test, ordinary_pred)),
        "event_macro_f1": float(f1_score(y_test, ordinary_pred, average="macro", zero_division=0)),
        "damage_balanced_event_accuracy": float(
            accuracy_score(y_test, conditional_pred, sample_weight=test_weights)
        ),
        "damage_balanced_event_macro_f1": float(
            f1_score(
                y_test,
                conditional_pred,
                average="macro",
                sample_weight=test_weights,
                zero_division=0,
            )
        ),
        "chance_accuracy": float(1.0 / len(encoder.classes_)),
    }


def run_iid(X, y, events, meta, args):
    train_idx, test_idx = get_iid_indices(meta, args.iid_split_csv, args.split_seed)
    X_train, X_test = fit_transform_fold(X, train_idx, test_idx)
    metrics = binary_probe(
        X_train, y[train_idx], X_test, y[test_idx], args.seeds[0], args.C
    )
    return [{
        "dataset": args.dataset,
        "model": args.model,
        "protocol": "iid",
        "heldout_event": "none",
        "method": "raw",
        "seed": args.seeds[0],
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "test_positive_rate": float(y[test_idx].mean()),
        **metrics,
    }]


def run_loeo(X, y, events, args, device):
    rows = []
    heldout_events = sorted(np.unique(events))
    if args.max_events is not None:
        heldout_events = heldout_events[: args.max_events]
        print(f"[SMOKE MODE] limiting LOEO evaluation to {heldout_events}")
    for heldout in heldout_events:
        train_idx = np.where(events != heldout)[0]
        test_idx = np.where(events == heldout)[0]
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            print(f"[WARN] {heldout}: one damage class in train or test; skipping")
            continue
        X_train, X_test = fit_transform_fold(X, train_idx, test_idx)

        for seed in args.seeds:
            for method in args.methods:
                if method in {"raw", "pca"} and seed != args.seeds[0]:
                    continue
                Z_train, Z_test = transform_method(
                    method,
                    X_train,
                    X_test,
                    y[train_idx],
                    events[train_idx],
                    args,
                    seed,
                    device,
                )
                metrics = binary_probe(
                    Z_train, y[train_idx], Z_test, y[test_idx], seed, args.C
                )
                row = {
                    "dataset": args.dataset,
                    "model": args.model,
                    "protocol": "loeo",
                    "heldout_event": heldout,
                    "method": method,
                    "seed": seed,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "test_positive_rate": float(y[test_idx].mean()),
                    **metrics,
                }
                rows.append(row)
                print(
                    f"{heldout:28s} {method:12s} seed={seed} "
                    f"BA={metrics['balanced_accuracy']:.3f} AUC={metrics['roc_auc']:.3f}",
                    flush=True,
                )
    return rows


def stratified_event_damage_split(events, damage, test_size: float, seed: int):
    strata = np.asarray([f"{e}::{int(y)}" for e, y in zip(events, damage)])
    counts = pd.Series(strata).value_counts()
    if counts.min() < 2:
        # Event-only stratification remains deterministic for rare event-label cells.
        strata = events
    idx = np.arange(len(events))
    return train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=strata
    )


def run_event_audit(X, y, events, args, device):
    train_idx, test_idx = stratified_event_damage_split(
        events, y, args.event_test_size, args.split_seed
    )
    X_train, X_test = fit_transform_fold(X, train_idx, test_idx)
    rows = []
    event_methods = [method for method in args.methods if method in {"raw", "pca", "supcon", "cross_event"}]

    for seed in args.seeds:
        for method in event_methods:
            if method in {"raw", "pca"} and seed != args.seeds[0]:
                continue
            Z_train, Z_test = transform_method(
                method,
                X_train,
                X_test,
                y[train_idx],
                events[train_idx],
                args,
                seed,
                device,
            )
            metrics = event_probe(
                Z_train,
                events[train_idx],
                y[train_idx],
                Z_test,
                events[test_idx],
                y[test_idx],
                seed,
            )
            rows.append({
                "dataset": args.dataset,
                "model": args.model,
                "method": method,
                "seed": seed,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                **metrics,
            })
            print(
                f"event audit {method:12s} seed={seed} "
                f"Acc={metrics['event_accuracy']:.3f} "
                f"damage-balanced={metrics['damage_balanced_event_accuracy']:.3f}",
                flush=True,
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["xview2", "bright"], required=True)
    parser.add_argument("--emb-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/mitigation"))
    parser.add_argument("--iid-split-csv", default=None)
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=ALL_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--event-test-size", type=float, default=0.2)
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Limit LOEO folds for runtime smoke tests only; omit for final runs.",
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--out-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--transform-batch-size", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    X, y, events, meta = load_embeddings(args.emb_root, args.model)
    print(
        f"dataset={args.dataset} model={args.model} X={X.shape} "
        f"events={len(np.unique(events))} positive_rate={y.mean():.3f} device={device}"
    )

    transfer_rows = run_iid(X, y, events, meta, args)
    transfer_rows.extend(run_loeo(X, y, events, args, device))
    event_rows = run_event_audit(X, y, events, args, device)

    stem = f"{args.dataset}_{args.model}"
    transfer_path = args.out_dir / f"transfer_{stem}.csv"
    event_path = args.out_dir / f"event_probe_{stem}.csv"
    config_path = args.out_dir / f"config_{stem}.json"
    pd.DataFrame(transfer_rows).to_csv(transfer_path, index=False)
    pd.DataFrame(event_rows).to_csv(event_path, index=False)
    with open(config_path, "w") as handle:
        json.dump({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, handle, indent=2)
    print(f"Saved {transfer_path}")
    print(f"Saved {event_path}")
    print(f"Saved {config_path}")


if __name__ == "__main__":
    main()
