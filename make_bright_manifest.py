"""Create a manifest from BRIGHT images and labels."""

import argparse
from pathlib import Path
import json
from collections import Counter
import pandas as pd

parser = argparse.ArgumentParser(description="Build a labeled BRIGHT manifest")
parser.add_argument("--root", type=Path, required=True, help="Extracted BRIGHT dataset root")
parser.add_argument("--out", type=Path, default=Path("data/bright_manifest.csv"))
args = parser.parse_args()

ROOT = args.root.expanduser().resolve()
OUT = args.out

PRE_DIR = ROOT / "pre-event"
POST_DIR = ROOT / "post-event"
LABEL_DIR = ROOT / "labels"

def norm_id(p):
    s = p.stem
    for suf in [
        "_pre_disaster",
        "_post_disaster",
        "_instance_damage",
        "_pre_event",
        "_post_event",
        "_pre",
        "_post",
    ]:
        if s.endswith(suf):
            s = s[:-len(suf)]
    return s

def event_from_id(sid):
    return sid.rsplit("_", 1)[0]

def hazard_from_event(event):
    e = event.lower()
    if "earthquake" in e:
        return "earthquake"
    if "volcano" in e:
        return "volcano"
    if "wildfire" in e or "fire" in e:
        return "wildfire"
    if "flood" in e:
        return "flood"
    if "explosion" in e:
        return "explosion"
    if "hurricane" in e:
        return "hurricane"
    if "conflict" in e:
        return "conflict"
    return "other"

pre = {norm_id(p): p for p in PRE_DIR.rglob("*.tif")}
post = {norm_id(p): p for p in POST_DIR.rglob("*.tif")}
labels = {norm_id(p): p for p in LABEL_DIR.rglob("*.json")}

common = sorted(set(pre) & set(post) & set(labels))

rows = []
for sid in common:
    with open(labels[sid], "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    id_to_name = {
        c.get("id"): str(c.get("name", "")).lower()
        for c in data.get("categories", [])
    }

    counts = Counter()
    for ann in data.get("annotations", []):
        cid = ann.get("category_id")
        cname = id_to_name.get(cid, "")

        if cid == 1 or "intact" in cname:
            counts["intact"] += 1
        elif cid == 2 or "damaged" in cname:
            counts["damaged"] += 1
        elif cid == 3 or "destroyed" in cname:
            counts["destroyed"] += 1
        else:
            counts["other"] += 1

    event = event_from_id(sid)
    hazard = hazard_from_event(event)

    n_intact = counts["intact"]
    n_damaged = counts["damaged"]
    n_destroyed = counts["destroyed"]
    n_other = counts["other"]
    n_buildings = n_intact + n_damaged + n_destroyed + n_other

    rows.append({
        "sample_id": sid,
        "event": event,
        "hazard": hazard,
        "pre_path": str(pre[sid]),
        "post_path": str(post[sid]),
        "label_path": str(labels[sid]),
        "n_buildings": n_buildings,
        "n_intact": n_intact,
        "n_damaged": n_damaged,
        "n_destroyed": n_destroyed,
        "n_other": n_other,
        "damage_present": int((n_damaged + n_destroyed) > 0),
    })

df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError(f"No labeled pre/post pairs found under {ROOT}")
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print("Saved:", OUT)
print("Rows:", len(df))
print("\nDamage distribution:")
print(df["damage_present"].value_counts().sort_index())

print("\nEvents:")
print(df.groupby("event")["damage_present"].agg(["count", "sum", "mean"]).sort_values("count", ascending=False))

print("\nHazards:")
print(df.groupby("hazard")["damage_present"].agg(["count", "sum", "mean"]).sort_values("count", ascending=False))
