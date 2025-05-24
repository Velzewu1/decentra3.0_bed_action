"""Feature engineering for card‑holder behavioural segmentation.

build_features(df: pd.DataFrame) -> pd.DataFrame
------------------------------------------------
Returns an aggregated feature table (one row per `card_id`).
Every feature is mapped to an explicit business meaning for ease of
presentation to non‑technical stakeholders.

FREQUENCY  – customer activity ( engagement )
MONETARY   – customer value / profitability
BEHAVIOUR  – channel & time preferences
GEOGRAPHY  – mobility / lifestyle
RECENCY    – days active & transaction tempo
DERIVED    – ratios that enhance interpretability
"""
from __future__ import annotations
import pandas as pd

# ---------------------------------------------------------------------------
# Business‑driven feature map: alias → (source_column, aggregation description)
# ---------------------------------------------------------------------------
FEATURE_MAP: dict[str, tuple[str, str]] = {
    # ── Frequency ───────────────────────────────────────────────────────────
    "tx_count"      : ("transaction_id", "count"),
    # ── Monetary ────────────────────────────────────────────────────────────
    "avg_amount"    : ("transaction_amount_kzt", "mean"),
    "std_amount"    : ("transaction_amount_kzt", "std"),
    "total_amount"  : ("transaction_amount_kzt", "sum"),
    # ── Behaviour / Channel ────────────────
    "digital_wallet_ratio": ("wallet_type", "digital"),  # custom lambda
    "contactless_ratio"  : ("pos_entry_mode", "contactless"),
    "international_ratio": ("transaction_currency", "intl"),
    "tx_type_variety"    : ("transaction_type", "nunique"),
    # ── Geography ─────────────────────────
    "city_diversity"     : ("merchant_city", "nunique"),
    "country_diversity"  : ("acquirer_country_iso", "nunique"),
    # ── Recency ────────────────────────────
    "days_active"   : ("transaction_timestamp", "days_active"),
    "tx_frequency"  : ("transaction_timestamp", "freq"),
}

# helper lambdas -------------------------------------------------------------
_DW   = lambda s: (s.notna()).mean()
_CNT  = lambda s: (s == "Contactless").mean()
_INTL = lambda s: (s != "KZT").mean()
_DAYS = lambda s: (s.max() - s.min()).days + 1
_FREQ = lambda s: len(s) / ((s.max() - s.min()).days + 1)

AGG_FUNCS: dict[str, dict[str,str|callable]] = {
    "transaction_id"           : {"tx_count": "count"},
    "transaction_amount_kzt"   : {
        "avg_amount": "mean",
        "std_amount": "std",
        "total_amount": "sum",
    },
    "wallet_type"              : {"digital_wallet_ratio": _DW},
    "pos_entry_mode"           : {"contactless_ratio": _CNT},
    "transaction_currency"     : {"international_ratio": _INTL},
    "transaction_type"         : {"tx_type_variety": "nunique"},
    "merchant_city"            : {"city_diversity": "nunique"},
    "acquirer_country_iso"     : {"country_diversity": "nunique"},
    "transaction_timestamp"    : {
        "days_active": _DAYS,
        "tx_frequency": _FREQ,
    },
}

# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return aggregated feature DataFrame (index = card_id)."""
    grp = df.groupby("card_id").agg(**{
        new: (col, func) for col, mapping in AGG_FUNCS.items() for new, func in mapping.items()
    })
    grp = grp.reset_index()

    # Derived ratios ---------------------------------------------------------
    grp["amount_volatility"] = grp["std_amount"] / grp["avg_amount"].replace(0, 1)
    grp["spending_consistency"] = 1 / (1 + grp["amount_volatility"].fillna(0))
    grp = grp.fillna(0)
    return grp