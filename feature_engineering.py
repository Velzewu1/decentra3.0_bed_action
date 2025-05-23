"""
Feature Engineering Script (Step 2)
----------------------------------
Reads raw DECENTRATHON_3.0.parquet from data/raw/, aggregates behavioural
metrics per card_id and saves data/interim/features_for_kmeans.parquet.
Run **before** elbow_analysis.py and kmeans_fit.py.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

INPUT = RAW / "DECENTRATHON_3.0.parquet"
OUTPUT = INTERIM / "features_for_kmeans.parquet"

if not INPUT.exists():
    raise FileNotFoundError(f"Raw parquet not found: {INPUT}")

print(f"Loading {INPUT.relative_to(ROOT)} …")
df = pd.read_parquet(INPUT)
print(f"Rows: {len(df):,}")

# Basic preprocessing
cat_fill = ["wallet_type", "acquirer_country_iso", "transaction_type", "mcc_category"]
for c in cat_fill:
    df[c] = df[c].fillna("Unknown")

df["transaction_timestamp"] = pd.to_datetime(df["transaction_timestamp"], errors="coerce")
df["hour"] = df["transaction_timestamp"].dt.hour

print("Aggregating features …")
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
    top_mcc_category=("mcc_category", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
    pct_wallet_usage=("wallet_type", lambda x: (x != "Unknown").mean()),
    wallet_type_preference=("wallet_type", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
    pct_foreign=("acquirer_country_iso", lambda x: (x != "KAZ").mean()),
    tx_days=("transaction_timestamp", lambda x: x.dt.date.nunique()),
    avg_hour=("hour", "mean"),
).reset_index()

agg.to_parquet(OUTPUT, index=False)
print(f"Saved features → {OUTPUT.relative_to(ROOT)}")
