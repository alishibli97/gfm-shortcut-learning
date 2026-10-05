#!/usr/bin/env python3
"""Aggregate shortcut-audit outputs into compact paper-facing tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ("balanced_accuracy", "roc_auc", "average_precision", "f1")


def bootstrap_delta(frame, reference, comparison, metric, repeats, seed):
    pivot = frame.pivot_table(
        index=["dataset", "model", "heldout_event"],
        columns="feature",
        values=metric,
        aggfunc="mean",
    ).dropna(subset=[reference, comparison])
    # Average backbones first, making held-out events the sampling units.
    event_values = pivot.reset_index().groupby(["dataset", "heldout_event"])[
        [reference, comparison]
    ].mean()
    deltas = event_values[comparison].to_numpy() - event_values[reference].to_numpy()
    rng = np.random.default_rng(seed)
    boot = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        boot[index] = rng.choice(deltas, size=len(deltas), replace=True).mean()
    return float(deltas.mean()), *np.quantile(boot, [0.025, 0.975]).tolist(), len(deltas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=Path, default=Path("runs/shortcut_audit")
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    paths = sorted(
        path for path in args.results_dir.glob("shortcut_*.csv")
        if path.name.startswith(("shortcut_xview2_", "shortcut_bright_"))
    )
    if not paths:
        raise FileNotFoundError(f"No shortcut_*.csv files in {args.results_dir}")
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)

    # Stochastic pair shuffles are seed-averaged before model/event aggregation.
    seed_mean = (
        raw.groupby(
            [
                "dataset",
                "model",
                "protocol",
                "feature",
                "weighting",
                "heldout_event",
            ],
            as_index=False,
        )[list(METRICS)]
        .mean()
    )
    macro = (
        seed_mean.groupby(
            ["dataset", "protocol", "feature", "weighting"], as_index=False
        )[list(METRICS)]
        .mean()
        .sort_values(["dataset", "protocol", "feature", "weighting"])
    )
    per_model = (
        seed_mean.groupby(
            ["dataset", "model", "protocol", "feature", "weighting"],
            as_index=False,
        )[list(METRICS)]
        .mean()
    )

    comparisons = []
    loeo = seed_mean[
        (seed_mean["protocol"] == "loeo")
        & (seed_mean["weighting"] == "class_balanced")
    ]
    for dataset in sorted(loeo["dataset"].unique()):
        subset = loeo[loeo["dataset"] == dataset]
        available_features = set(subset["feature"].unique())
        for feature in ("pre", "post", "delta", "abs_delta"):
            if feature not in available_features or "concat" not in available_features:
                continue
            for metric in ("balanced_accuracy", "roc_auc"):
                delta, low, high, n_events = bootstrap_delta(
                    subset, "concat", feature, metric, args.bootstrap, args.seed
                )
                comparisons.append(
                    {
                        "dataset": dataset,
                        "comparison": f"{feature}-minus-concat",
                        "metric": metric,
                        "delta": delta,
                        "ci_low": low,
                        "ci_high": high,
                        "n_events": n_events,
                    }
                )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    seed_mean.to_csv(args.results_dir / "shortcut_per_event_seed_mean.csv", index=False)
    per_model.to_csv(args.results_dir / "shortcut_macro_per_model.csv", index=False)
    macro.to_csv(args.results_dir / "shortcut_macro_mean.csv", index=False)
    pd.DataFrame(comparisons).to_csv(
        args.results_dir / "shortcut_paired_bootstrap.csv", index=False
    )

    print("\n=== Mean over backbones; LOEO macro over events ===")
    print(
        macro[
            macro["protocol"].isin(["iid", "loeo", "loeo_pair_stress"])
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\n=== Paired event-bootstrap temporal-view differences ===")
    print(
        pd.DataFrame(comparisons).to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )
    print("\nSaved summaries to", args.results_dir)


if __name__ == "__main__":
    main()
