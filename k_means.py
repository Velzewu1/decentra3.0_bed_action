"""
K‑Means + GPT Segmentation Pipeline
===================================
This Python script prepares behavioural features from the DECENTRATHON_3.0.parquet
transaction dataset, runs K‑Means clustering (≥3 segments), and exports cluster
summaries ready to be sent to ChatGPT (or any LLM) for human‑readable naming and
insight generation.

Requirements
------------
- pandas
- numpy
- scikit‑learn
- pyarrow (for parquet I/O)

Run
----
$ python kmeans_segmentation_pipeline.py

Outputs
-------
1. features_for_kmeans.parquet   – aggregated feature table per card_id
2. clusters.parquet              – same table with `cluster` label
3. gpt_prompts.json              – JSON list of cluster profiles for LLM
"""

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DATA_DIR = Path("")            # change if needed
INPUT_FILE = "DECENTRATHON_3.0.parquet"
N_CLUSTERS = 5                           # ≥3; tweak after elbow analysis
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. LOAD RAW TRANSACTIONS
# ---------------------------------------------------------------------------
print("Loading parquet …")
df_raw = pd.read_parquet(INPUT_FILE)
print(f"Rows: {len(df_raw):,}")

# Coerce timestamp & add hour
print("Parsing timestamps …")
df_raw["transaction_timestamp"] = pd.to_datetime(
    df_raw["transaction_timestamp"], errors="coerce"
)
df_raw["hour"] = df_raw["transaction_timestamp"].dt.hour

# Normalise NaNs
for col in [
    "wallet_type",
    "acquirer_country_iso",
    "transaction_type",
    "mcc_category",
]:
    df_raw[col] = df_raw[col].fillna("Unknown")

# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
print("Aggregating behavioural metrics …")
agg = df_raw.groupby("card_id").agg(
    total_tx_count=("transaction_id", "count"),
    avg_amount=("transaction_amount_kzt", "mean"),
    std_amount=("transaction_amount_kzt", "std"),
    pct_POS=("transaction_type", lambda x: (x == "POS").mean()),
    pct_ATM=("transaction_type", lambda x: (x == "ATM_WITHDRAWAL").mean()),
    pct_P2P=("transaction_type", lambda x: x.str.startswith("P2P").mean()),
    pct_BILL=("transaction_type", lambda x: (x == "BILL_PAYMENT").mean()),
    pct_ECOM=("transaction_type", lambda x: (x == "ECOM").mean()),
    unique_mcc_count=("merchant_mcc", pd.Series.nunique),
    top_mcc_category=("mcc_category", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
    pct_wallet_usage=("wallet_type", lambda x: (x != "Unknown").mean()),
    wallet_type_preference=("wallet_type", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
    pct_foreign=("acquirer_country_iso", lambda x: (x != "KAZ").mean()),
    tx_days=("transaction_timestamp", lambda x: x.dt.date.nunique()),
    avg_hour=("hour", "mean"),
).reset_index()

# Store feature table
FEATURE_FILE = DATA_DIR / "features_for_kmeans.parquet"
agg.to_parquet(FEATURE_FILE, index=False)
print(f"Saved features → {FEATURE_FILE}")

# ---------------------------------------------------------------------------
# 3. PREPROCESSING & K‑MEANS
# ---------------------------------------------------------------------------
print("Building preprocessing pipeline …")

numeric_cols = [
    "total_tx_count",
    "avg_amount",
    "std_amount",
    "pct_POS",
    "pct_ATM",
    "pct_P2P",
    "pct_BILL",
    "pct_ECOM",
    "unique_mcc_count",
    "pct_wallet_usage",
    "pct_foreign",
    "tx_days",
    "avg_hour",
]

categorical_cols = ["top_mcc_category", "wallet_type_preference"]

preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

pipeline = Pipeline(
    steps=[
        ("prep", preprocessor),
        (
            "kmeans",
            KMeans(n_clusters=N_CLUSTERS, n_init="auto", random_state=RANDOM_STATE),
        ),
    ]
)

print("Fitting K‑Means …")
labels = pipeline.fit_predict(agg[numeric_cols + categorical_cols])
agg["cluster"] = labels

CLUSTER_FILE = DATA_DIR / "clusters.parquet"
agg.to_parquet(CLUSTER_FILE, index=False)
print(f"Clustered data → {CLUSTER_FILE}")

# ---------------------------------------------------------------------------
# 4. BUILD PROFILES FOR GPT
# ---------------------------------------------------------------------------

print("Generating cluster profiles …")
cluster_profiles: List[dict] = []

for cluster_id in sorted(agg["cluster"].unique()):
    subset = agg[agg["cluster"] == cluster_id]
    size_abs = len(subset)
    size_rel = size_abs / len(agg)

    # get centroid (mean of features)
    centroid = subset[numeric_cols].mean().to_dict()

    # get dominant categorical values
    dom_mcc = subset["top_mcc_category"].mode().iloc[0]
    dom_wallet = subset["wallet_type_preference"].mode().iloc[0]

    profile_text = (
        f"Сегмент {cluster_id}:\n"
        f"• Кол-во клиентов: {size_abs} ({size_rel:.1%} от базы)\n"
        f"• Средний чек: {centroid['avg_amount']:.0f} ₸\n"
        f"• Транзакции в месяц: {centroid['total_tx_count']:.1f}\n"
        f"• Категория трат №1: {dom_mcc}\n"
        f"• Предпочтительный кошелёк: {dom_wallet}\n"
        f"• Доля POS: {centroid['pct_POS']:.0%}, ATM: {centroid['pct_ATM']:.0%}, P2P: {centroid['pct_P2P']:.0%}\n"
        f"• Зарубежные операции: {centroid['pct_foreign']:.0%}\n"
    )

    cluster_profiles.append({
        "cluster_id": int(cluster_id),
        "profile": profile_text,
    })

OUTPUT_JSON = DATA_DIR / "gpt_prompts.json"
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(cluster_profiles, f, ensure_ascii=False, indent=2)

print(f"Cluster profiles → {OUTPUT_JSON}\n\nDone.")
