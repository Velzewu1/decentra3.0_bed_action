"""
HDBSCAN Segmentation Pipeline · Auto‑tune v2 (noise‑only cut)
============================================================
* Grid‑search по `min_cluster_size` = 35‑65 (шаг 5), фиксированный
  `min_samples` (по умолчанию 10, меняется флагом `--min_samples`).
* Функция качества:  
  `score = silhouette − ALPHA_NOISE·max(0, noise−MAX_NOISE) − BETA_K·|k−TARGET_K|`  
  где ALPHA = 1.5, BETA = 0.05, MAX_NOISE = 0.15, TARGET_K = 6.
* **Post‑процесс**: отбрасываем все точки, помеченные HDBSCAN как noise (`-1`),
  и перенумеровываем кластеры в диапазон 0‥k‑1.  
  (Нет фильтра persistence и size.)

Запуск
------
```bash
# стандартная сетка 35‑65 шаг 5, min_samples = 10
python src/hdbscan_segmentation_pipeline.py

# быстрая проверка только на mcs = 45 и 55
python src/hdbscan_segmentation_pipeline.py --quick

# своя сетка и min_samples
python src/hdbscan_segmentation_pipeline.py --grid 30 70 10 --min_samples 8
```
"""
import argparse, json, random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import hdbscan  # type: ignore
except ImportError:  # pragma: no cover
    raise SystemExit("Install hdbscan: pip install hdbscan")

# ------------------------- scoring params -------------------------
ALPHA_NOISE = 1.5   # penalty for excess noise
BETA_K      = 0.05  # penalty for |k − target|
MAX_NOISE   = 0.15  # acceptable noise share
TARGET_K    = 6     # desired number of clusters
SAMPLE_ROWS = 10_000

# ------------------------- CLI -----------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--grid", nargs=3, type=int, metavar=("START", "END", "STEP"))
parser.add_argument("--quick", action="store_true")
parser.add_argument("--min_samples", type=int, default=10,
                    help="fixed min_samples for every grid point")
args = parser.parse_args()

if args.grid:
    s, e, st = args.grid
    GRID = list(range(s, e + 1, st))
elif args.quick:
    GRID = [45, 55]
else:
    GRID = list(range(35, 66, 5))  # 35 40 45 50 55 60 65

MIN_SAMPLES = args.min_samples

# ------------------------- paths ---------------------------------
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

# ------------------------- load data -----------------------------
print("[1/5] Load raw …")
if not INPUT.exists():
    raise FileNotFoundError(INPUT)

df = pd.read_parquet(INPUT)
df["transaction_timestamp"] = pd.to_datetime(df["transaction_timestamp"], errors="coerce")
df["hour"] = df["transaction_timestamp"].dt.hour
for c in ["wallet_type", "acquirer_country_iso", "transaction_type", "mcc_category"]:
    df[c] = df[c].fillna("Unknown")

# ------------------------- feature engineering -------------------
print("[2/5] Build features …")
agg = df.groupby("card_id").agg(
    total_tx_count=("transaction_id", "count"),
    avg_amount=("transaction_amount_kzt", "mean"),
    std_amount=("transaction_amount_kzt", "std"),
    pct_POS=("transaction_type", lambda x: (x == "POS").mean()),
    pct_ATM=("transaction_type", lambda x: (x == "ATM_WITHDRAWAL").mean()),
    pct_P2P=("transaction_type", lambda x: x.str.startswith("P2P").mean()),
    pct_BILL=("transaction_type", lambda x: (x == "BILL_PAYMENT").mean()),
    pct_ECOM=("transaction_type", lambda x: (x == "ECOM").mean()),
    unique_mcc_count=("merchant_mcc", pd.Series.nunique),
    top_mcc_category=("mcc_category", lambda x: x.mode().iat[0] if not x.mode().empty else "Unknown"),
    pct_wallet_usage=("wallet_type", lambda x: (x != "Unknown").mean()),
    wallet_type_preference=("wallet_type", lambda x: x.mode().iat[0] if not x.mode().empty else "Unknown"),
    pct_foreign=("acquirer_country_iso", lambda x: (x != "KAZ").mean()),
    tx_days=("transaction_timestamp", lambda x: x.dt.date.nunique()),
    avg_hour=("hour", "mean"),
).reset_index()
agg.to_parquet(FEAT, index=False)

num = [
    "total_tx_count", "avg_amount", "std_amount", "pct_POS", "pct_ATM", "pct_P2P",
    "pct_BILL", "pct_ECOM", "unique_mcc_count", "pct_wallet_usage", "pct_foreign",
    "tx_days", "avg_hour",
]
cat = ["top_mcc_category", "wallet_type_preference"]
X = ColumnTransformer([
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
]).fit_transform(agg[num + cat])

# ------------------------- grid search ---------------------------
print("[3/5] Grid search …")
results = []
for mcs in GRID:
    cl = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=MIN_SAMPLES).fit(X)
    lbl = cl.labels_
    k = len(set(lbl)) - (-1 in lbl)
    noise = (lbl == -1).mean()
    idx = random.sample(range(X.shape[0]), min(SAMPLE_ROWS, X.shape[0])) if k >= 2 else []
    sil = silhouette_score(X[idx], lbl[idx]) if idx else -1
    score = sil - ALPHA_NOISE * max(0, noise - MAX_NOISE) - BETA_K * abs(k - TARGET_K)
    results.append({
        "min_cluster_size": mcs,
        "min_samples": MIN_SAMPLES,
        "n_clusters": k,
        "noise": round(noise, 3),
        "silhouette": round(sil, 3),
        "score": round(score, 3)
    })

best = max(results, key=lambda r: r["score"])
print("        best →", best)

# ------------------------- final fit + noise cut ----------------
print("[4/5] Fit best HDBSCAN …")
labels = hdbscan.HDBSCAN(
    min_cluster_size=best["min_cluster_size"],
    min_samples=best["min_samples"],
).fit_predict(X)

mask_core = labels != -1
core_labels = labels[mask_core]
remap = {old: new for new, old in enumerate(sorted(set(core_labels)))}
core_labels = np.array([remap[x] for x in core_labels], dtype=int)

agg_core = agg.loc[mask_core].copy()
agg_core["cluster"] = core_labels
agg_core.to_parquet(CLUST, index=False)
print(f"        clusters saved → {CLUST.relative_to(ROOT)}  (k = {len(remap)}, noise removed = {(~mask_core).mean():.2%})")

# ------------------------- save report --------------------------
print("[5/5] Save search report …")
REPORT.write_text(json.dumps({"grid": results, "best": best}, indent=2), encoding="utf-8")
print(f"        report → {REPORT.relative_to(ROOT)}\nDone.")
