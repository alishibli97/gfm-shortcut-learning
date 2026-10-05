#!/usr/bin/env python3
"""Source-only shortcut audits for frozen pre/post EO embeddings.

The script is deliberately independent of the training code used for the
representation interventions.  It consumes the existing embedding layout:

    EMB_ROOT/MODEL/pre_embeddings.npy
    EMB_ROOT/MODEL/post_embeddings.npy
    EMB_ROOT/MODEL/manifest.csv

For each backbone it evaluates:
  * temporal-view ablations: pre, post, signed change, absolute change, concat;
  * ordinary versus event x damage-cell-balanced LOEO probes;
  * same-event, same-label pair shuffles at test time;
  * event-prior, building-density, and non-event metadata-only diagnostics.

Every fitted scaler, encoder, prior, and classifier uses source samples only.
The target event is used only for final evaluation.
"""

from __future__ import annotations

import argparse
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore", category=ConvergenceWarning)

FEATURES = ("pre", "post", "delta", "abs_delta", "concat")
BUILDING_COUNT_COLUMNS = ("num_buildings", "n_buildings")
CONTEXT_CATEGORICAL_COLUMNS = (
    "hazard",
    "disaster_type",
    "sensor",
    "source_name",
)
CONTEXT_NUMERIC_COLUMNS = ("gsd",)


def load_embeddings(emb_root: Path, model: str):
    model_root = emb_root / model
    pre = np.load(model_root / "pre_embeddings.npy").astype(np.float32)
    post = np.load(model_root / "post_embeddings.npy").astype(np.float32)
    meta = pd.read_csv(model_root / "manifest.csv")
    if len(pre) != len(post) or len(pre) != len(meta):
        raise RuntimeError(
            f"Row mismatch: pre={len(pre)} post={len(post)} manifest={len(meta)}"
        )
    required = {"event", "damage_present"}
    missing = required - set(meta.columns)
    if missing:
        raise RuntimeError(f"Manifest is missing required columns: {sorted(missing)}")
    return pre, post, meta


def make_feature(pre: np.ndarray, post: np.ndarray, name: str) -> np.ndarray:
    if name == "pre":
        return pre
    if name == "post":
        return post
    if name == "delta":
        return post - pre
    if name == "abs_delta":
        return np.abs(post - pre)
    if name == "concat":
        return np.concatenate([pre, post, post - pre], axis=1)
    raise ValueError(name)


def safe_metrics(y_true, pred, score):
    result = {
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "positive_rate": float(np.mean(y_true)),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, score))
        result["average_precision"] = float(average_precision_score(y_true, score))
    else:
        result["roc_auc"] = np.nan
        result["average_precision"] = np.nan
    return result


def event_cell_weights(events: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Give each observed event x damage cell equal total source weight."""
    keys = pd.Series(events.astype(str)) + "::" + pd.Series(y.astype(str))
    counts = keys.value_counts()
    weights = keys.map(lambda key: 1.0 / counts[key]).to_numpy(dtype=np.float64)
    return weights * (len(weights) / weights.sum())


def fit_embedding_probe(
    X_train,
    y_train,
    X_test,
    seed: int,
    C: float,
    sample_weight=None,
):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train)
    test_scaled = scaler.transform(X_test)
    clf = LogisticRegression(
        solver="liblinear",
        max_iter=5000,
        C=C,
        class_weight=None if sample_weight is not None else "balanced",
        random_state=seed,
    )
    clf.fit(train_scaled, y_train, sample_weight=sample_weight)
    return clf, scaler, clf.predict(test_scaled), clf.decision_function(test_scaled)


def iid_indices(meta: pd.DataFrame, split_csv: Path | None, split_seed: int):
    if split_csv is not None:
        split_df = pd.read_csv(split_csv)
        if "split" not in split_df:
            raise RuntimeError(f"{split_csv} must contain a 'split' column")
        id_candidates = ["sample_id", "example_id"]
        id_column = next(
            (col for col in id_candidates if col in split_df and col in meta), None
        )
        if id_column is None:
            if len(split_df) != len(meta):
                raise RuntimeError(
                    "IID split has no shared ID column and does not match manifest length"
                )
            split = split_df["split"].astype(str).to_numpy()
        else:
            mapping = dict(
                zip(split_df[id_column].astype(str), split_df["split"].astype(str))
            )
            split = meta[id_column].astype(str).map(mapping).to_numpy()
            if pd.isna(split).any():
                raise RuntimeError(f"IID split is missing manifest IDs for {id_column}")
        train = np.flatnonzero(np.isin(split, ["train", "training"]))
        test = np.flatnonzero(np.isin(split, ["test", "val", "validation"]))
        if not len(train) or not len(test):
            raise RuntimeError("IID split did not produce non-empty train and test sets")
        return train, test

    if "random_split" in meta:
        split = meta["random_split"].astype(str).str.lower().to_numpy()
        train = np.flatnonzero(np.isin(split, ["train", "training"]))
        test = np.flatnonzero(np.isin(split, ["test", "val", "validation"]))
        if len(train) and len(test):
            return train, test

    y = meta["damage_present"].astype(int).to_numpy()
    train, test = train_test_split(
        np.arange(len(meta)),
        test_size=0.2,
        stratify=y,
        random_state=split_seed,
    )
    return np.asarray(train), np.asarray(test)


def result_row(args, protocol, feature, weighting, heldout, seed, train, test, metrics):
    return {
        "dataset": args.dataset,
        "model": args.model,
        "protocol": protocol,
        "feature": feature,
        "weighting": weighting,
        "heldout_event": heldout,
        "seed": seed,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        **metrics,
    }


def run_loeo_event(heldout, pre, post, y, events, args):
    """Run every embedding diagnostic for one held-out event."""
    rows = []
    train = np.flatnonzero(events != heldout)
    test = np.flatnonzero(events == heldout)
    if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
        print(f"[WARN] skipping one-class LOEO fold {heldout}", flush=True)
        return rows

    concat_clf = None
    concat_scaler = None
    for feature in args.features:
        X = make_feature(pre, post, feature)
        weightings = ["class_balanced"]
        if feature == "concat":
            weightings.append("event_damage_balanced")
        for weighting in weightings:
            weights = (
                None
                if weighting == "class_balanced"
                else event_cell_weights(events[train], y[train])
            )
            clf, scaler, pred, score = fit_embedding_probe(
                X[train], y[train], X[test], args.seeds[0], args.C, weights
            )
            if feature == "concat" and weighting == "class_balanced":
                concat_clf, concat_scaler = clf, scaler
            rows.append(
                result_row(
                    args,
                    "loeo",
                    feature,
                    weighting,
                    heldout,
                    args.seeds[0],
                    train,
                    test,
                    safe_metrics(y[test], pred, score),
                )
            )

    # Reuse the already fitted class-balanced concat probe. Shuffling is
    # restricted to the same event and damage label, preserving prevalence
    # while destroying the correct local pre/post correspondence.
    if concat_clf is None or concat_scaler is None:
        X_train = make_feature(pre[train], post[train], "concat")
        concat_clf, concat_scaler, _, _ = fit_embedding_probe(
            X_train, y[train], X_train[:1], args.seeds[0], args.C
        )
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        for shuffled_side in ("pre", "post"):
            pre_test = pre[test].copy()
            post_test = post[test].copy()
            for label in (0, 1):
                local = np.flatnonzero(y[test] == label)
                if len(local) > 1:
                    permuted = rng.permutation(local)
                    if shuffled_side == "pre":
                        pre_test[local] = pre[test][permuted]
                    else:
                        post_test[local] = post[test][permuted]
            X_test = make_feature(pre_test, post_test, "concat")
            X_test = concat_scaler.transform(X_test)
            pred = concat_clf.predict(X_test)
            score = concat_clf.decision_function(X_test)
            rows.append(
                result_row(
                    args,
                    "loeo_pair_stress",
                    f"concat_{shuffled_side}_shuffled_same_label",
                    "class_balanced",
                    heldout,
                    seed,
                    train,
                    test,
                    safe_metrics(y[test], pred, score),
                )
            )
    return rows


def write_temporal_checkpoint(rows, checkpoint_path):
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame.drop_duplicates(
            subset=["protocol", "feature", "weighting", "heldout_event", "seed"],
            keep="last",
        )
    frame.to_csv(checkpoint_path, index=False)


def run_temporal_views(pre, post, meta, args):
    y = meta["damage_present"].astype(int).to_numpy()
    events = meta["event"].astype(str).to_numpy()
    checkpoint_path = (
        args.out_dir / f"partial_temporal_{args.dataset}_{args.model}.csv"
    )
    if checkpoint_path.exists():
        rows = pd.read_csv(checkpoint_path).to_dict("records")
        print(f"Resuming {len(rows)} checkpoint rows from {checkpoint_path}", flush=True)
    else:
        rows = []

    iid_existing = {
        row["feature"] for row in rows if row.get("protocol") == "iid"
    }
    if not set(args.features).issubset(iid_existing):
        rows = [row for row in rows if row.get("protocol") != "iid"]
        train, test = iid_indices(meta, args.iid_split_csv, args.split_seed)
        for feature in args.features:
            X = make_feature(pre, post, feature)
            _, _, pred, score = fit_embedding_probe(
                X[train], y[train], X[test], args.seeds[0], args.C
            )
            rows.append(
                result_row(
                    args,
                    "iid",
                    feature,
                    "class_balanced",
                    "none",
                    args.seeds[0],
                    train,
                    test,
                    safe_metrics(y[test], pred, score),
                )
            )
        write_temporal_checkpoint(rows, checkpoint_path)
        print("Checkpointed IID temporal views", flush=True)

    expected_event_rows = len(args.features) + 1 + 2 * len(args.seeds)
    counts = pd.Series(
        [
            str(row["heldout_event"])
            for row in rows
            if row.get("protocol") in {"loeo", "loeo_pair_stress"}
        ]
    ).value_counts()
    heldout_events = sorted(np.unique(events))
    pending = [
        event for event in heldout_events if counts.get(event, 0) < expected_event_rows
    ]
    # Discard an incomplete event before recomputing it.
    rows = [
        row
        for row in rows
        if not (
            str(row.get("heldout_event")) in pending
            and row.get("protocol") in {"loeo", "loeo_pair_stress"}
        )
    ]
    print(
        f"LOEO events complete={len(heldout_events) - len(pending)} "
        f"pending={len(pending)} workers={args.n_jobs}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.n_jobs) as executor:
        futures = {
            executor.submit(run_loeo_event, event, pre, post, y, events, args): event
            for event in pending
        }
        for future in as_completed(futures):
            event = futures[future]
            event_rows = future.result()
            rows.extend(event_rows)
            write_temporal_checkpoint(rows, checkpoint_path)
            print(
                f"Checkpointed heldout={event} rows={len(event_rows)} "
                f"total_rows={len(rows)}",
                flush=True,
            )
    return rows


def group_prior_scores(groups_train, y_train, groups_test):
    frame = pd.DataFrame({"group": groups_train.astype(str), "y": y_train})
    stats = frame.groupby("group")["y"].agg(["sum", "count"])
    # Jeffreys smoothing avoids exact 0/1 probabilities in small groups.
    priors = ((stats["sum"] + 0.5) / (stats["count"] + 1.0)).to_dict()
    fallback = float((np.sum(y_train) + 0.5) / (len(y_train) + 1.0))
    return np.asarray([priors.get(str(group), fallback) for group in groups_test])


def prior_row(args, protocol, name, heldout, train, test, y, scores):
    pred = (scores >= 0.5).astype(int)
    return result_row(
        args,
        protocol,
        name,
        "metadata_only",
        heldout,
        args.seeds[0],
        train,
        test,
        safe_metrics(y[test], pred, scores),
    )


def fit_tabular_probe(meta_train, y_train, meta_test, categorical, numeric, seed):
    transformers = []
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if not transformers:
        return None
    model = Pipeline(
        [
            ("prepare", ColumnTransformer(transformers)),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(meta_train, y_train)
    return model.predict(meta_test), model.predict_proba(meta_test)[:, 1]


def run_metadata_diagnostics(meta, args):
    y = meta["damage_present"].astype(int).to_numpy()
    events = meta["event"].astype(str).to_numpy()
    rows = []
    count_col = next((col for col in BUILDING_COUNT_COLUMNS if col in meta), None)
    categorical = [col for col in CONTEXT_CATEGORICAL_COLUMNS if col in meta]
    numeric = [col for col in CONTEXT_NUMERIC_COLUMNS if col in meta]

    # Prevent post-label fields from entering the context-only diagnostic.
    context_cols = categorical + numeric
    train_iid, test_iid = iid_indices(meta, args.iid_split_csv, args.split_seed)
    event_scores = group_prior_scores(events[train_iid], y[train_iid], events[test_iid])
    rows.append(
        prior_row(
            args,
            "iid",
            "event_prior",
            "none",
            train_iid,
            test_iid,
            y,
            event_scores,
        )
    )

    folds = [("iid", "none", train_iid, test_iid)]
    folds.extend(
        (
            "loeo",
            heldout,
            np.flatnonzero(events != heldout),
            np.flatnonzero(events == heldout),
        )
        for heldout in sorted(np.unique(events))
    )
    for protocol, heldout, train, test in folds:
        if len(np.unique(y[test])) < 2:
            continue
        if count_col is not None:
            values = np.log1p(
                pd.to_numeric(meta[count_col], errors="coerce").fillna(0).to_numpy()
            ).reshape(-1, 1)
            _, _, pred, score = fit_embedding_probe(
                values[train], y[train], values[test], args.seeds[0], args.C
            )
            rows.append(
                result_row(
                    args,
                    protocol,
                    "building_count_only",
                    "metadata_only",
                    heldout,
                    args.seeds[0],
                    train,
                    test,
                    safe_metrics(y[test], pred, score),
                )
            )
        if context_cols:
            fitted = fit_tabular_probe(
                meta.iloc[train],
                y[train],
                meta.iloc[test],
                categorical,
                numeric,
                args.seeds[0],
            )
            if fitted is not None:
                pred, score = fitted
                rows.append(
                    result_row(
                        args,
                        protocol,
                        "context_metadata_only",
                        "metadata_only",
                        heldout,
                        args.seeds[0],
                        train,
                        test,
                        safe_metrics(y[test], pred, score),
                    )
                )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["xview2", "bright"], required=True)
    parser.add_argument("--emb-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("runs/shortcut_audit")
    )
    parser.add_argument("--iid-split-csv", type=Path)
    parser.add_argument(
        "--features",
        nargs="+",
        choices=FEATURES,
        default=["pre", "post", "delta", "concat"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pre, post, meta = load_embeddings(args.emb_root, args.model)
    print(
        f"dataset={args.dataset} model={args.model} n={len(meta)} dim={pre.shape[1]} "
        f"events={meta['event'].nunique()} positive_rate={meta['damage_present'].mean():.3f}",
        flush=True,
    )

    rows = run_temporal_views(pre, post, meta, args)
    rows.extend(run_metadata_diagnostics(meta, args))
    output = args.out_dir / f"shortcut_{args.dataset}_{args.model}.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    config = args.out_dir / f"config_{args.dataset}_{args.model}.json"
    with open(config, "w", encoding="utf-8") as handle:
        json.dump(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            handle,
            indent=2,
        )
    print(f"Saved {output}")
    print(f"Saved {config}")
    print(f"COMPLETED: shortcut_{args.dataset}_{args.model}")


if __name__ == "__main__":
    main()
