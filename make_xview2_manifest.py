#!/usr/bin/env python3
"""
make_xview2_manifest.py

Build a single xView2/xBD manifest from multiple roots, e.g.

  raw challenge training set
  tier3 additional training data

It recursively scans for:
  *_pre_disaster.png
  *_post_disaster.png
  *_pre_disaster.json
  *_post_disaster.json

and creates one image-pair-level manifest.

Example:

python make_xview2_manifest.py \
  --roots \
    /path/to/xView2/raw \
    /path/to/xView2/tier3 \
  --out data/xview2_full19_manifest.csv
"""

import argparse
import csv
import json
import re
from pathlib import Path
from collections import Counter, defaultdict


FILENAME_RE = re.compile(
    r"^(?P<event>.+)_(?P<tile_id>\d+)_(?P<time>pre|post)_disaster\.(?P<ext>png|json)$"
)

DAMAGE_CLASSES = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed",
    "un-classified",
]


def parse_name(path: Path):
    m = FILENAME_RE.match(path.name)
    if m is None:
        return None

    event = m.group("event")
    tile_id = m.group("tile_id")
    time = m.group("time")
    ext = m.group("ext")
    example_id = f"{event}_{tile_id}"

    return {
        "event": event,
        "tile_id": tile_id,
        "time": time,
        "ext": ext,
        "example_id": example_id,
    }


def read_json_safe(path: Path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not read JSON {path}: {e}")
        return None


def count_damage_from_label(label_path: Path):
    counts = Counter({k: 0 for k in DAMAGE_CLASSES})
    metadata = {}

    if label_path is None or not label_path.exists():
        return counts, metadata

    js = read_json_safe(label_path)
    if js is None:
        return counts, metadata

    metadata = js.get("metadata", {}) or {}

    features = js.get("features", {}) or {}
    xy = features.get("xy", []) or []

    for feat in xy:
        props = feat.get("properties", {}) or {}
        subtype = props.get("subtype", "un-classified")

        if subtype not in DAMAGE_CLASSES:
            subtype = "un-classified"

        counts[subtype] += 1

    return counts, metadata


def infer_source(root: Path, path: Path):
    """
    Stores a readable source name.
    """
    try:
        rel = path.relative_to(root)
        first = rel.parts[0] if len(rel.parts) > 0 else ""
        return first
    except Exception:
        return ""


def build_manifest(roots):
    images = defaultdict(dict)
    labels = defaultdict(dict)

    # Track duplicate basenames just in case
    seen_image_keys = set()
    seen_label_keys = set()

    for root in roots:
        root = Path(root)

        if not root.exists():
            raise FileNotFoundError(f"Root does not exist: {root}")

        print(f"Scanning root: {root}")

        for p in root.rglob("*"):
            if not p.is_file():
                continue

            if p.suffix.lower() not in [".png", ".json"]:
                continue

            parsed = parse_name(p)
            if parsed is None:
                continue

            example_id = parsed["example_id"]
            time = parsed["time"]
            ext = parsed["ext"]

            key = (example_id, time, ext)

            source_name = infer_source(root, p)

            if ext == "png":
                if key in seen_image_keys:
                    print(f"[WARN] Duplicate image key {key}: {p}")
                seen_image_keys.add(key)

                images[example_id][time] = {
                    "path": p,
                    "event": parsed["event"],
                    "tile_id": parsed["tile_id"],
                    "source_root": str(root),
                    "source_name": source_name,
                }

            elif ext == "json":
                if key in seen_label_keys:
                    print(f"[WARN] Duplicate label key {key}: {p}")
                seen_label_keys.add(key)

                labels[example_id][time] = p

    rows = []

    all_ids = sorted(set(images.keys()))

    missing_pre = 0
    missing_post = 0

    for example_id in all_ids:
        item = images[example_id]

        if "pre" not in item:
            missing_pre += 1
            continue

        if "post" not in item:
            missing_post += 1
            continue

        pre_info = item["pre"]
        post_info = item["post"]

        event = pre_info["event"]
        tile_id = pre_info["tile_id"]

        pre_image = pre_info["path"]
        post_image = post_info["path"]

        pre_label = labels.get(example_id, {}).get("pre", None)
        post_label = labels.get(example_id, {}).get("post", None)

        counts, metadata = count_damage_from_label(post_label)

        no_damage_count = counts["no-damage"]
        minor_damage_count = counts["minor-damage"]
        major_damage_count = counts["major-damage"]
        destroyed_count = counts["destroyed"]
        unclassified_count = counts["un-classified"]

        damaged_count = minor_damage_count + major_damage_count + destroyed_count

        num_buildings = (
            no_damage_count
            + minor_damage_count
            + major_damage_count
            + destroyed_count
            + unclassified_count
        )

        damage_present = int(damaged_count > 0)

        if num_buildings > 0:
            damage_fraction = damaged_count / num_buildings
        else:
            damage_fraction = 0.0

        class_counts = {
            "no-damage": no_damage_count,
            "minor-damage": minor_damage_count,
            "major-damage": major_damage_count,
            "destroyed": destroyed_count,
            "un-classified": unclassified_count,
        }

        if num_buildings > 0:
            dominant_damage_class = max(class_counts, key=class_counts.get)
        else:
            dominant_damage_class = "none"

        row = {
            "dataset": "xView2",
            "event": event,
            "tile_id": tile_id,
            "example_id": example_id,

            "pre_image": str(pre_image),
            "post_image": str(post_image),
            "pre_label": str(pre_label) if pre_label is not None else "",
            "post_label": str(post_label) if post_label is not None else "",
            "has_labels": int(post_label is not None),

            "source_root": pre_info["source_root"],
            "source_name": pre_info["source_name"],

            "num_buildings": num_buildings,
            "damage_present": damage_present,
            "damage_fraction": damage_fraction,
            "damaged_count": damaged_count,

            "no_damage_count": no_damage_count,
            "minor_damage_count": minor_damage_count,
            "major_damage_count": major_damage_count,
            "destroyed_count": destroyed_count,
            "unclassified_count": unclassified_count,
            "dominant_damage_class": dominant_damage_class,

            "disaster_type": metadata.get("disaster_type", ""),
            "capture_date": metadata.get("capture_date", ""),
            "sensor": metadata.get("sensor", ""),
            "gsd": metadata.get("gsd", ""),
        }

        rows.append(row)

    print()
    print(f"Complete pre/post pairs: {len(rows)}")
    print(f"Missing pre:             {missing_pre}")
    print(f"Missing post:            {missing_post}")

    return rows


def write_csv(rows, out_path: Path):
    if len(rows) == 0:
        raise RuntimeError("No complete pre/post image pairs found.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote manifest: {out_path}")
    print(f"Rows: {len(rows)}")


def print_summary(rows):
    events = Counter(r["event"] for r in rows)
    labeled = Counter(r["has_labels"] for r in rows)
    damage = Counter(r["damage_present"] for r in rows)
    source_names = Counter(r["source_name"] for r in rows)

    print("\nSummary")
    print("=" * 80)
    print(f"Pairs:        {len(rows)}")
    print(f"Events:       {len(events)}")
    print(f"Has labels:   {dict(labeled)}")
    print(f"Damage label: {dict(damage)}")

    print("\nSource groups:")
    for k, v in source_names.most_common():
        print(f"  {k if k else 'UNKNOWN':30s} {v}")

    print("\nEvents:")
    for event, n in events.most_common():
        print(f"  {event:40s} {n}")

    print("\nDamage prevalence by event:")
    for event in sorted(events):
        ev_rows = [r for r in rows if r["event"] == event]
        rate = sum(int(r["damage_present"]) for r in ev_rows) / max(len(ev_rows), 1)
        mean_frac = sum(float(r["damage_fraction"]) for r in ev_rows) / max(len(ev_rows), 1)
        print(f"  {event:40s} present={rate:.3f}  fraction={mean_frac:.4f}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="One or more xView2 roots to scan recursively.",
    )

    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output manifest CSV.",
    )

    args = ap.parse_args()

    rows = build_manifest([Path(r) for r in args.roots])
    write_csv(rows, Path(args.out))
    print_summary(rows)


if __name__ == "__main__":
    main()
