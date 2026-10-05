#!/usr/bin/env python3
"""Aggregate per-model REO results and bootstrap paired event-level changes."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["f1", "balanced_accuracy", "roc_auc", "average_precision"]


def bootstrap_event_delta(df, method, metric, n_boot, seed):
    subset = df[df["method"].isin(["raw", method])].copy()
    per_seed = subset.groupby(
        ["dataset", "model", "heldout_event", "method"], as_index=False
    )[metric].mean()
    wide = per_seed.pivot_table(
        index=["dataset", "model", "heldout_event"], columns="method", values=metric
    ).dropna()
    wide["delta"] = wide[method] - wide["raw"]

    # Models share the same held-out events; average models first, then resample events.
    per_event = wide.reset_index().groupby(["dataset", "heldout_event"])["delta"].mean()
    rng = np.random.default_rng(seed)
    rows = []
    for dataset in per_event.index.get_level_values(0).unique():
        values = per_event.xs(dataset).to_numpy()
        draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
        rows.append({
            "dataset": dataset,
            "method": method,
            "metric": metric,
            "delta": float(values.mean()),
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)),
            "n_events": int(len(values)),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    out_dir = args.out_dir or args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    transfer_files = sorted(args.results_dir.glob("transfer_*.csv"))
    event_files = sorted(
        path for path in args.results_dir.glob("event_probe_*.csv")
        if path.name.startswith(("event_probe_xview2_", "event_probe_bright_"))
    )
    if not transfer_files or not event_files:
        raise RuntimeError("Expected transfer_*.csv and event_probe_*.csv files")
    transfer = pd.concat([pd.read_csv(path) for path in transfer_files], ignore_index=True)
    event = pd.concat([pd.read_csv(path) for path in event_files], ignore_index=True)

    iid = transfer[transfer["protocol"] == "iid"].copy()
    loeo = transfer[transfer["protocol"] == "loeo"].copy()
    loeo_seed_mean = loeo.groupby(
        ["dataset", "model", "method", "heldout_event"], as_index=False
    )[METRICS].mean()
    macro_model = loeo_seed_mean.groupby(
        ["dataset", "model", "method"], as_index=False
    )[METRICS].mean()
    macro_mean = macro_model.groupby(["dataset", "method"], as_index=False)[METRICS].mean()

    event_cols = [
        "event_accuracy",
        "event_balanced_accuracy",
        "event_macro_f1",
        "damage_balanced_event_accuracy",
        "damage_balanced_event_macro_f1",
        "chance_accuracy",
    ]
    event_summary = event.groupby(["dataset", "model", "method"], as_index=False)[event_cols].mean()

    bootstrap_rows = []
    for method in [m for m in loeo["method"].unique() if m != "raw"]:
        for metric in ["balanced_accuracy", "roc_auc"]:
            bootstrap_rows.extend(
                bootstrap_event_delta(loeo, method, metric, args.bootstrap, args.seed)
            )
    bootstrap = pd.DataFrame(bootstrap_rows)

    iid.to_csv(out_dir / "iid_raw.csv", index=False)
    loeo_seed_mean.to_csv(out_dir / "loeo_per_event_seed_mean.csv", index=False)
    macro_model.to_csv(out_dir / "loeo_macro_per_model.csv", index=False)
    macro_mean.to_csv(out_dir / "loeo_macro_mean.csv", index=False)
    event_summary.to_csv(out_dir / "event_probe_summary.csv", index=False)
    bootstrap.to_csv(out_dir / "paired_event_bootstrap.csv", index=False)

    print("\nLOEO macro mean across models")
    print(macro_mean.round(4).to_string(index=False))
    print("\nEvent probe summary")
    print(event_summary.round(4).to_string(index=False))
    print("\nPaired event bootstrap")
    print(bootstrap.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
