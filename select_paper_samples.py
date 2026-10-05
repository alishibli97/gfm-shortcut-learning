#!/usr/bin/env python3
"""Select the same image pairs and row order used in the paper."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["xview2", "bright"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    if args.dataset == "xview2":
        ids = pd.read_csv(repo / "data/samples/xview2.csv")
        id_column = "example_id"
    else:
        ids = pd.read_csv(repo / "data/bright_splits/random_damage_stratified.csv")
        id_column = "sample_id"
    manifest = pd.read_csv(args.manifest)
    if manifest[id_column].duplicated().any():
        raise ValueError(f"Duplicate {id_column} values in {args.manifest}")
    indexed = manifest.set_index(id_column)
    missing = ids.loc[~ids[id_column].isin(indexed.index), id_column]
    if len(missing):
        raise ValueError(f"Missing {len(missing)} paper samples; first five: {missing.head().tolist()}")
    selected = indexed.loc[ids[id_column]].reset_index()
    if args.dataset == "xview2":
        selected["random_split"] = ids["random_split"].to_numpy()
        # Preserve the context audit's source groups independently of folder layout.
        selected["source_name"] = ids["source_name"].to_numpy()
        if not selected["has_labels"].eq(1).all():
            raise ValueError("Some xView2 paper samples have no labels")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.out, index=False)
    print(f"Saved {len(selected)} samples from {selected.event.nunique()} events to {args.out}")


if __name__ == "__main__":
    main()
