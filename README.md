# Shortcut Learning in Geospatial Foundation Model Embeddings for Cross-Disaster Damage Assessment

Code for the paper accepted at NeurIPS 2026, 2nd Workshop on Advances in Representation Learning for Earth Observation. [Citation](CITATION.cff).

Follow these steps to run the code.

## 1. Install requirements

Use Python 3.10 or newer.

```bash
git clone https://github.com/alishibli97/gfm-shortcut-learning.git
cd gfm-shortcut-learning
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Make the dataset manifests

Download and extract [xView2/xBD](https://github.com/DIUx-xView/xView2_baseline) and [BRIGHT](https://github.com/ChenHongruixuan/BRIGHT). Replace the paths below with your dataset folders. The second command selects the same image pairs and splits used in the paper.

**xView2:**

```bash
python make_xview2_manifest.py \
  --roots /path/to/xView2/raw /path/to/xView2/tier3 \
  --out data/xview2_all.csv
python select_paper_samples.py \
  --dataset xview2 --manifest data/xview2_all.csv \
  --out data/xview2_manifest.csv
```

**BRIGHT:** The dataset folder should contain `pre-event/`, `post-event/`, and `labels/` with the instance-damage JSON files.

```bash
python make_bright_manifest.py \
  --root /path/to/BRIGHT --out data/bright_all.csv
python select_paper_samples.py \
  --dataset bright --manifest data/bright_all.csv \
  --out data/bright_manifest.csv
```

## 3. Get the models

The model names used in the commands are `prithvi`, `terramind`, `dofa_base`, and `dinov3`.

Prithvi, TerraMind, and DOFA load their pretrained weights automatically when you extract embeddings.

For DINOv3, download the **ViT-L/16 LVD-1689M** checkpoint from the [official repository](https://github.com/facebookresearch/dinov3) and get the model code:

```bash
git clone https://github.com/facebookresearch/dinov3.git ../dinov3
```

## 4. Extract embeddings

These examples use Prithvi. Replace `prithvi` with `terramind` or `dofa_base` to run those models.

```bash
python extract_xview2_embeddings.py \
  --manifest data/xview2_manifest.csv --model prithvi \
  --out-dir embeddings/xview2
python extract_bright_embeddings.py \
  --manifest data/bright_manifest.csv --model prithvi \
  --out-dir embeddings/bright
```

For DINOv3, also provide the model code and checkpoint paths:

```bash
python extract_xview2_embeddings.py \
  --manifest data/xview2_manifest.csv --model dinov3 \
  --dinov3-repo ../dinov3 \
  --dinov3-weights /path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --out-dir embeddings/xview2
```

For BRIGHT, use `extract_bright_embeddings.py`, `data/bright_manifest.csv`, and `embeddings/bright` in the same command.

## 5. Run the shortcut audits

Each command runs the pre/post, metadata, building-count, pair-shuffling, and event–class weighting audits. Replace `prithvi` with any of the four model names to test that model.

```bash
python reo_shortcut_audit.py \
  --dataset xview2 --emb-root embeddings/xview2 --model prithvi \
  --out-dir runs/shortcut_audit
python reo_shortcut_audit.py \
  --dataset bright --emb-root embeddings/bright --model prithvi \
  --iid-split-csv data/bright_splits/random_damage_stratified.csv \
  --out-dir runs/shortcut_audit
```

## 6. Run alignment and comparison methods

These commands run raw features, PCA, MLP, SupCon, and cross-event SupCon. Repeat them for each model.

```bash
python reo_experiments.py \
  --dataset xview2 --emb-root embeddings/xview2 --model prithvi \
  --methods raw pca ce_mlp supcon cross_event \
  --out-dir runs/mitigation
python reo_experiments.py \
  --dataset bright --emb-root embeddings/bright --model prithvi \
  --iid-split-csv data/bright_splits/random_damage_stratified.csv \
  --methods raw pca ce_mlp supcon cross_event \
  --out-dir runs/mitigation
```

Results are saved as CSV files under `runs/`. After running the models, combine their results with:

```bash
python summarize_shortcut_audit.py --results-dir runs/shortcut_audit
python summarize_reo_results.py --results-dir runs/mitigation
```
