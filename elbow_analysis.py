"""
Elbow Analysis Helper
---------------------
Run this script after `features_for_kmeans.parquet` is generated.
It loads the aggregated features, performs the elbow analysis
(inertia & silhouette) for k = 2..10 and saves `elbow_curve.png`.

Usage
-----
$ python src/elbow_analysis.py

Requires: pandas, numpy, scikit-learn, matplotlib, kneed(optional)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

try:
    from kneed import KneeLocator
except ImportError:
    KneeLocator = None

ROOT_DIR = Path(__file__).resolve().parents[1]
INTERIM_DIR = ROOT_DIR / "data" / "interim"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

FEATURE_FILE = INTERIM_DIR / "features_for_kmeans.parquet"
ELBOW_PNG = PROCESSED_DIR / "elbow_curve.png"
REPORT_JSON = PROCESSED_DIR / "elbow_report.json"

print(f"Loading features: {FEATURE_FILE.relative_to(ROOT_DIR)}")
agg = pd.read_parquet(FEATURE_FILE)

numeric_cols = [
    "total_tx_count", "avg_amount", "std_amount", "pct_POS", "pct_ATM",
    "pct_P2P", "pct_BILL", "pct_ECOM", "unique_mcc_count",
    "pct_wallet_usage", "pct_foreign", "tx_days", "avg_hour",
]
cat_cols = ["top_mcc_category", "wallet_type_preference"]

prep = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

X = prep.fit_transform(agg[numeric_cols + cat_cols])

ks, inertias, silhouettes = [], [], []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)
    ks.append(k)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels))

plt.figure(figsize=(6, 3))
plt.plot(ks, inertias, marker="o")
plt.title("Elbow Curve (Inertia vs k)")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.tight_layout()
ELBOW_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(ELBOW_PNG, dpi=150)
print(f"Elbow curve saved to {ELBOW_PNG.relative_to(ROOT_DIR)}")

opt_k = None
if KneeLocator is not None:
    knee = KneeLocator(ks, inertias, curve="convex", direction="decreasing")
    opt_k = knee.knee
if not opt_k:
    opt_k = ks[int(np.argmax(silhouettes))]
print(f"Optimal k suggested: {opt_k}")

# save report json
report = {
    "ks": ks,
    "inertias": inertias,
    "silhouettes": silhouettes,
    "optimal_k": int(opt_k),
}
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
with open(REPORT_JSON, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print(f"Elbow report → {REPORT_JSON.relative_to(ROOT_DIR)}")
