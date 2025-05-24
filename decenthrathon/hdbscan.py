"""
HDBSCAN Segmentation Pipeline · Auto‑tune v2 (noise‑only cut, 2‑D grid)
======================================================================
* Grid‑search по **двум осям**:
    – min_cluster_size = 35‑65 (шаг 5)
    – min_samples ∈ {6, 8, 10, 12} (или фиксируется флагом --min_samples)
* Функция качества:  
  score = silhouette − ALPHA_NOISE·max(0, noise−MAX_NOISE) − BETA_K·|k−TARGET_K|
  где ALPHA = 1.5, BETA = 0.05, MAX_NOISE = 0.15, TARGET_K = 6.
* После обучения все точки `-1` удаляются, кластеры перенумеровываются 0‥k-1.

Запуск
------
```bash
# стандартная сетка
python src/hdbscan_segmentation_pipeline.py

# быстрая проверка (mcs = 45/55, ms = 8/10)
python src/hdbscan_segmentation_pipeline.py --quick

# собственная сетка и фикс. min_samples
python src/hdbscan_segmentation_pipeline.py --grid 30 70 10 --min_samples 9
```"""
from __future__ import annotations

import argparse, json, random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import build_features

try:
    import hdbscan  # type: ignore
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install hdbscan: pip install hdbscan") from e

# ── scoring constants ──────────────────────────────────────────
ALPHA_NOISE = 1.5
BETA_K      = 0.05
MAX_NOISE   = 0.15
TARGET_K    = 6
SAMPLE_ROWS = 10_000

# ── CLI ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--grid", nargs=3, type=int, metavar=("START", "END", "STEP"))
parser.add_argument("--quick", action="store_true")
parser.add_argument("--min_samples", type=int,
                    help="фиксировать одно значение min_samples; иначе используется {6,8,10,12}")
args = parser.parse_args()

# axis‑1: min_cluster_size
if args.grid:
    s, e, st = args.grid
    MCS_GRID = list(range(s, e + 1, st))
elif args.quick:
    MCS_GRID = [45, 55]
else:
    MCS_GRID = list(range(35, 66, 5))  # 35 40 45 50 55 60 65

# axis‑2: min_samples
if args.min_samples is not None:
    MS_GRID = [args.min_samples]
elif args.quick:
    MS_GRID = [8, 10]
else:
    MS_GRID = [6, 8, 10, 12]

# ── paths ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "raw"
INT  = ROOT / "data" / "interim"
PRO  = ROOT / "data" / "processed"
for p in (RAW, INT, PRO):
    p.mkdir(parents=True, exist_ok=True)

INPUT   = RAW / "DECENTRATHON_3.0.parquet"
FEAT    = INT / "features_for_hdbscan.parquet"
CLUST   = PRO / "clusters_hdbscan.parquet"
REPORT  = PRO / "param_search_report.json"

# ── 1. Load raw ───────────────────────────────────────────────
print("[1/6] Load raw …")
if not INPUT.exists():
    raise FileNotFoundError(INPUT)

df = pd.read_parquet(INPUT)
df["transaction_timestamp"] = pd.to_datetime(df["transaction_timestamp"], errors="coerce")
df["hour"] = df["transaction_timestamp"].dt.hour
for c in ["wallet_type", "acquirer_country_iso", "transaction_type", "mcc_category"]:
    df[c] = df[c].fillna("Unknown")

# ── 2. Feature engineering ────────────────────────────────────
print("[2/6] Build features …")
agg = build_features(df)
agg.to_parquet(FEAT, index=False)

num_cols = [c for c in agg.columns if c not in ("card_id",) and agg[c].dtype != "object"]
cat_cols = [c for c in agg.columns if agg[c].dtype == "object"]
X = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
]).fit_transform(agg[num_cols + cat_cols])

# ── 3. 2‑D grid‑search ───────────────────────────────────────
print("[3/6] Grid search …")
results: list[dict] = []
for ms in MS_GRID:
    for mcs in MCS_GRID:
        labels = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms).fit_predict(X)
        k_core = len(set(labels)) - (-1 in labels)
        noise  = (labels == -1).mean()
        if k_core >= 2:
            idx = random.sample(range(X.shape[0]), min(SAMPLE_ROWS, X.shape[0]))
            sil = silhouette_score(X[idx], labels[idx])
        else:
            sil = -1
        score = sil - ALPHA_NOISE * max(0, noise - MAX_NOISE) - BETA_K * abs(k_core - TARGET_K)
        results.append({
            "min_cluster_size": mcs,
            "min_samples": ms,
            "n_clusters": k_core,
            "noise": round(noise, 3),
            "silhouette": round(sil, 3),
            "score": round(score, 3)
        })

best = max(results, key=lambda r: r["score"])
print("        best →", best)

# ── 4. Fit best model & noise‑cut ─────────────────────────────
print("[4/6] Fit best HDBSCAN …")
labels = hdbscan.HDBSCAN(min_cluster_size=best["min_cluster_size"],
                         min_samples=best["min_samples"]).fit_predict(X)

mask_core = labels != -1
core_labels = labels[mask_core]
remap = {old: new for new, old in enumerate(sorted(set(core_labels)))}
core_labels = np.array([remap[c] for c in core_labels], dtype=int)

agg_core = agg.loc[mask_core].copy()
agg_core["cluster"] = core_labels
agg_core.to_parquet(CLUST, index=False)
print(f"        clusters saved → {CLUST.relative_to(ROOT)}  (k = {len(remap)}, noise removed = {(~mask_core).mean():.2%})")

# ── 5. Save report ────────────────────────────────────────────
print("[5/6] Save search report …")
REPORT.write_text(json.dumps({"grid": results, "best": best}, indent=2), encoding="utf-8")
print(f"        report → {REPORT.relative_to(ROOT)}")

print("[6/6] Done.")