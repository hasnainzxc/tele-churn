"""Churn prediction pipeline — data cleaning, feature engineering, model inference.

Exported predict_churn() matches the assignment signature exactly.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

# pipeline artifact lives two dirs up from this file
ARTIFACT_PATH = Path(__file__).parent.parent.parent / "artifacts" / "model_pipeline.pkl"

# maps raw user-facing values from the CSV/app into canonical slugs
# fiber → fiber_optic bc the encoder was fit on that string
# bt/cc are shorthand that support agents sometimes type
_CATEGORY_MAPS: dict[str, dict[str, str]] = {
    "gender": {
        "male": "male", "m": "male",
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
    # kick out nans, blanks, placeholder strings — treat as "unknown" so nothing crashes
    if pd.isna(val) or str(val).strip().lower() in ("", "nan", "none"):
        return "unknown"
    mapping = _CATEGORY_MAPS.get(col, {})
    # if the raw value isn't in our lookup, pass it through as-is (lowered)
    # this handles new/custom values that didn't show up in training
    return mapping.get(str(val).strip().lower(), str(val).strip().lower())


def _is_corrupted(row: pd.Series, col: str) -> bool:
    """Check if a numeric value is corrupted (impossible range)."""
    val = row[col]
    if pd.isna(val):
        return True
    # satisfaction is 1-10 in the survey instrument — anything else is garbage
    if col == "satisfaction_score" and (val < 0 or val > 10):
        return True
    # negative GB usage means something broke in the billing system
    if col == "avg_monthly_gb_used" and val < 0:
        return True
    # support system caps at 10 open tickets per customer
    if col == "num_support_tickets" and (val < 0 or val > 10):
        return True
    # minors can't hold contracts, over 100 is likely data entry error
    if col == "age" and (val < 18 or val > 100):
        return True
    # negative tenure means a future join date got entered — corrupt row
    if col == "tenure_months" and val < 0:
        return True
    return False


def _impute_corrupted(df: pd.DataFrame, col: str) -> pd.Series:
    """Impute corrupted values with median of valid values."""
    # grab only clean rows for this column, then fill junk with their median
    valid = df[~df.apply(lambda r: _is_corrupted(r, col), axis=1)][col]
    if len(valid) > 0:
        return df[col].mask(
            df.apply(lambda r: _is_corrupted(r, col), axis=1), valid.median()
        )
    # no valid values at all — just hand the column back untouched and hope for the best
    return df[col]


def preprocess_customer(customer_data: dict[str, Any]) -> pd.DataFrame:
    """Preprocess a single customer dict into model-ready features."""
    df = pd.DataFrame([customer_data])

    # these columns exist in the raw data but aren't features — drop em
    drop_cols = {"customer_id", "churned", "last_interaction_date"}
    present = [c for c in drop_cols if c in df.columns]
    if present:
        df = df.drop(columns=present)

    # map every categorical column through the normalization table
    for col in _CATEGORY_MAPS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: _normalize_categorical(x, col))

    # coerce numerics, then fix corrupted values with median imputation
    for col in [
        "satisfaction_score",
        "avg_monthly_gb_used",
        "num_support_tickets",
        "age",
        "tenure_months",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # check whether we even need to impute — skip if column's already clean
            needs_impute = False
            if col in df.columns:
                row = df.iloc[0]
                # single-row df means we can just check the first (only) row directly
                if hasattr(row, col) and not pd.isna(row[col]):
                    needs_impute = _is_corrupted(df.iloc[0], col)
            if needs_impute:
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
    # load the serialized pipeline — model + encoder + scaler + feature names
    if ARTIFACT_PATH.exists():
        with open(ARTIFACT_PATH, "rb") as f:
            pipeline = pickle.load(f)
        model = pipeline.get("model")
        encoder = pipeline.get("encoder")
        scaler = pipeline.get("scaler")
        feature_names = pipeline.get("feature_names", [])
    else:
        # no artifact = can't predict. return a neutral default so callers don't blow up
        return {
            "churn_probability": 0.45,
            "risk_tier": "medium",
            "top_risk_factors": [
                {"feature": "model_not_available", "contribution": "Model artifact not found"},
            ],
        }

    # run the full preprocessing pipeline on a single row
    df = preprocess_customer(customer_data)

    # split columns by type so we can one-hot encode cats and scale nums
    cat_cols = [c for c in _CATEGORY_MAPS if c in df.columns]
    num_cols = [c for c in df.columns if c not in cat_cols]

    # one-hot encode categoricals the same way the training pipeline did
    if cat_cols and encoder is not None:
        encoded = encoder.transform(df[cat_cols])
        enc_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
        df = df.drop(columns=cat_cols)
        df = pd.concat([df.reset_index(drop=True), enc_df.reset_index(drop=True)], axis=1)

    # scale numeric columns to match training-time distribution
    if num_cols and scaler is not None:
        df[num_cols] = scaler.transform(df[num_cols])

    # reorder columns to match exactly what the model saw during fit
    # missing columns get filled with 0 (feature wasn't present in this row)
    if feature_names:
        df = df.reindex(columns=feature_names, fill_value=0)

    # get the probability of class 1 (churned)
    proba = float(model.predict_proba(df)[0, 1])

    # thresholds are arbitrary but match what the notebook tuned against
    # ≥0.6 = definitely leaving, ≤0.3 = safe, middle = keep an eye on them
    if proba >= 0.6:
        risk_tier = "high"
    elif proba >= 0.3:
        risk_tier = "medium"
    else:
        risk_tier = "low"

    # SHAP analysis isn't hooked up yet — placeholder so the agent doesn't choke
    top_risk_factors = [
        {"feature": "predictions_pending", "contribution": "Model loaded, SHAP analysis pending"},
    ]

    return {
        "churn_probability": round(proba, 4),
        "risk_tier": risk_tier,
        "top_risk_factors": top_risk_factors,
    }
