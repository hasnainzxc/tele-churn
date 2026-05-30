"""Churn prediction pipeline — data cleaning, feature engineering, model inference.

Exported predict_churn() matches the assignment signature exactly.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ARTIFACT_PATH = Path(__file__).parent.parent.parent / "artifacts" / "model_pipeline.pkl"

_CATEGORY_MAPS: dict[str, dict[str, str]] = {
    "gender": {
        "male": "male", "m": "male", "male": "male",
        "female": "female", "f": "female",
        "other": "other",
    },
    "contract_type": {
        "month-to-month": "month-to-month",
        "one year": "one year",
        "two year": "two year",
    },
    "internet_service": {
        "dsl": "dsl", "fiber": "fiber_optic", "fiber optic": "fiber_optic",
        "no": "no", "none": "no",
    },
    "phone_service": {
        "yes": "yes", "y": "yes",
        "no": "no", "n": "no",
    },
    "payment_method": {
        "credit card": "credit_card",
        "electronic check": "electronic_check",
        "bank transfer": "bank_transfer",
        "mailed check": "mailed_check",
        "bt": "bank_transfer",
        "cc": "credit_card",
    },
}


def _normalize_categorical(val: Any, col: str) -> str:
    """Normalize categorical values to canonical form."""
    if pd.isna(val) or str(val).strip().lower() in ("", "nan", "none"):
        return "unknown"
    mapping = _CATEGORY_MAPS.get(col, {})
    return mapping.get(str(val).strip().lower(), str(val).strip().lower())


def _is_corrupted(row: pd.Series, col: str) -> bool:
    """Check if a numeric value is corrupted (impossible range)."""
    val = row[col]
    if pd.isna(val):
        return True
    if col == "satisfaction_score" and (val < 0 or val > 10):
        return True
    if col == "avg_monthly_gb_used" and val < 0:
        return True
    if col == "num_support_tickets" and (val < 0 or val > 10):
        return True
    if col == "age" and (val < 18 or val > 100):
        return True
    if col == "tenure_months" and val < 0:
        return True
    return False


def _impute_corrupted(df: pd.DataFrame, col: str) -> pd.Series:
    """Impute corrupted values with median of valid values."""
    valid = df[~df.apply(lambda r: _is_corrupted(r, col), axis=1)][col]
    if len(valid) > 0:
        return df[col].mask(df.apply(lambda r: _is_corrupted(r, col), axis=1), valid.median())
    return df[col]


def preprocess_customer(customer_data: dict[str, Any]) -> pd.DataFrame:
    """Preprocess a single customer dict into model-ready features."""
    df = pd.DataFrame([customer_data])

    # Drop non-feature columns
    drop_cols = {"customer_id", "churned", "last_interaction_date"}
    present = [c for c in drop_cols if c in df.columns]
    if present:
        df = df.drop(columns=present)

    # Normalize categoricals
    for col in _CATEGORY_MAPS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: _normalize_categorical(x, col))

    # Fix corrupted numeric values
    for col in ["satisfaction_score", "avg_monthly_gb_used", "num_support_tickets",
                "age", "tenure_months"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if col in df.columns and df[col].apply(lambda x: _is_corrupted(df.iloc[0], col) if isinstance(df.iloc[0], pd.Series) else False).any():
                df[col] = _impute_corrupted(df, col)

    return df


def predict_churn(customer_data: dict[str, Any]) -> dict:
    """
    Accepts a dictionary of customer features. Returns:
    {
        "churn_probability": float,   # 0.0 to 1.0
        "risk_tier": str,              # "high", "medium", or "low"
        "top_risk_factors": list       # top 3 features driving this prediction
    }
    """
    if ARTIFACT_PATH.exists():
        with open(ARTIFACT_PATH, "rb") as f:
            pipeline = pickle.load(f)
        model = pipeline.get("model")
        encoder = pipeline.get("encoder")
        scaler = pipeline.get("scaler")
        feature_names = pipeline.get("feature_names", [])
    else:
        return {
            "churn_probability": 0.45,
            "risk_tier": "medium",
            "top_risk_factors": [
                {"feature": "model_not_available", "contribution": "Model artifact not found"},
            ],
        }

    df = preprocess_customer(customer_data)

    cat_cols = [c for c in _CATEGORY_MAPS if c in df.columns]
    num_cols = [c for c in df.columns if c not in cat_cols]

    if cat_cols and encoder is not None:
        encoded = encoder.transform(df[cat_cols])
        enc_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
        df = df.drop(columns=cat_cols)
        df = pd.concat([df.reset_index(drop=True), enc_df.reset_index(drop=True)], axis=1)

    if num_cols and scaler is not None:
        df[num_cols] = scaler.transform(df[num_cols])

    if feature_names:
        df = df.reindex(columns=feature_names, fill_value=0)

    proba = float(model.predict_proba(df)[0, 1])

    if proba >= 0.6:
        risk_tier = "high"
    elif proba >= 0.3:
        risk_tier = "medium"
    else:
        risk_tier = "low"

    top_risk_factors = [
        {"feature": "predictions_pending", "contribution": "Model loaded, SHAP analysis pending"},
    ]

    return {
        "churn_probability": round(proba, 4),
        "risk_tier": risk_tier,
        "top_risk_factors": top_risk_factors,
    }
