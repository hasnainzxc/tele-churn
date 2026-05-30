"""Churn prediction pipeline — data cleaning, feature engineering, model inference.

Exported predict_churn() matches the assignment signature exactly.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

ARTIFACT_PATH = Path(__file__).parent.parent.parent / "artifacts" / "model_pipeline.pkl"

# Notebok-style category maps — canonical forms are Title Case with spaces.
# predict.py used underscores before; the encoder was trained on Title Case.
_CATEGORY_MAPS: dict[str, dict[str, str]] = {
    "gender": {
        "male": "Male", "m": "Male",
        "female": "Female", "f": "Female",
        "other": "Other",
    },
    "internet_service": {
        "dsl": "DSL", "fiber optic": "Fiber optic", "fiber": "Fiber optic",
        "no": "No", "none": "No",
    },
    "phone_service": {
        "yes": "Yes", "y": "Yes",
        "no": "No", "n": "No",
    },
    "payment_method": {
        "credit card": "Credit card", "cc": "Credit card",
        "electronic check": "Electronic check",
        "bank transfer": "Bank transfer", "bt": "Bank transfer",
        "mailed check": "Mailed check",
    },
}

# Median values computed from the 5,050-row training csv.
# Stored here so single-row inference works without full-dataset context.
_MEDIANS: dict[str, float] = {
    "satisfaction_score": 5.8,
    "avg_monthly_gb_used": 31.0,
    "num_support_tickets": 2.0,
    "age": 42.2,
    "tenure_months": 35.8,
    "monthly_charges": 69.4,
    "total_charges": 2200.0,
    "avg_monthly_minutes": 469.0,
    "num_additional_services": 1.0,
}

# Column name mapping: cleaned categoricals use short _clean suffix.
# Eg gender_clean, internet_clean, phone_clean, payment_clean
# contract_type stays as-is (no _clean)
_COL_RENAME: dict[str, str] = {
    "gender": "gender_clean",
    "internet_service": "internet_clean",
    "phone_service": "phone_clean",
    "payment_method": "payment_clean",
}


def _normalize_categorical(val: Any, col: str) -> str:
    if pd.isna(val) or str(val).strip().lower() in ("", "nan", "none"):
        return "Unknown"
    mapping = _CATEGORY_MAPS.get(col, {})
    return mapping.get(str(val).strip().lower(), str(val).strip().title())


def _is_corrupted(val: float, col: str) -> bool:
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


def _fix_corrupted(val: float, col: str) -> float:
    if _is_corrupted(val, col):
        return _MEDIANS.get(col, 0.0)
    return val


def preprocess_customer(customer_data: dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([customer_data])

    # Drop non-feature columns
    drop_cols = {"customer_id", "churned", "last_interaction_date"}
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    # Normalize categoricals using notebook-compatible Title Case maps
    for col, rename_col in _COL_RENAME.items():
        if col in df.columns:
            df[rename_col] = df[col].apply(lambda x: _normalize_categorical(x, col))
    # contract_type: lowercase only, no _clean suffix
    if "contract_type" in df.columns:
        df["contract_type"] = df["contract_type"].astype(str).str.lower().str.strip()

    # Fix corrupted numerics + coerce to float
    corruptible = [
        "satisfaction_score", "avg_monthly_gb_used", "num_support_tickets",
        "age", "tenure_months",
    ]
    for col in corruptible:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].apply(lambda x: _fix_corrupted(x, col))

    # Coerce remaining numerics
    for col in ["monthly_charges", "total_charges", "avg_monthly_minutes",
                 "num_additional_services"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if pd.isna(df[col].iloc[0]):
                df[col] = _MEDIANS.get(col, 0.0)

    # Engineered feature: billing_ratio
    monthly = max(df["monthly_charges"].iloc[0], 1.0)
    tenure = max(df["tenure_months"].iloc[0], 1.0)
    df["billing_ratio"] = df["total_charges"].iloc[0] / (monthly * tenure)
    df["billing_ratio"] = df["billing_ratio"].clip(0, 5.0)  # 99th percentile ~5
    if pd.isna(df["billing_ratio"].iloc[0]):
        df["billing_ratio"] = 1.0

    # Engineered feature: service_density
    internet_yes = df["internet_clean"].iloc[0] not in ("No", "Unknown")
    phone_yes = df["phone_clean"].iloc[0] not in ("No", "Unknown")
    df["service_density"] = int(internet_yes) + int(phone_yes)

    return df


def predict_churn(customer_data: dict[str, Any]) -> dict:
    if not ARTIFACT_PATH.exists():
        return {
            "churn_probability": 0.45,
            "risk_tier": "medium",
            "top_risk_factors": [
                {"feature": "model_not_available",
                 "contribution": "Model artifact not found"},
            ],
        }

    with open(ARTIFACT_PATH, "rb") as f:
        pipeline = pickle.load(f)

    df = preprocess_customer(customer_data)

    # Pipeline expects these columns (order matters):
    num_cols = [
        "age", "tenure_months", "monthly_charges", "total_charges",
        "avg_monthly_gb_used", "num_support_tickets", "avg_monthly_minutes",
        "satisfaction_score", "num_additional_services",
        "billing_ratio", "service_density",
    ]
    cat_cols = [
        "contract_type", "gender_clean", "internet_clean",
        "phone_clean", "payment_clean",
    ]
    feature_order = num_cols + cat_cols

    # Only select columns the pipeline was trained on
    available = [c for c in feature_order if c in df.columns]
    x = df[available]

    # Fill missing expected columns with defaults
    for c in feature_order:
        if c not in x.columns:
            if c in num_cols:
                x[c] = 0.0
            else:
                x[c] = "Unknown"
    x = x[feature_order]

    proba = float(pipeline.predict_proba(x)[0, 1])

    if proba >= 0.6:
        risk_tier = "high"
    elif proba >= 0.3:
        risk_tier = "medium"
    else:
        risk_tier = "low"

    # Extract top risk factors from LogisticRegression coefficients.
    # Coefficients map to the transformed (scaled + one-hot) feature space.
    # We reverse-map them to the original feature names via the preprocessor.
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    coeffs = classifier.coef_[0]
    feature_names_out = preprocessor.get_feature_names_out()

    # Pair feature names with absolute coefficient magnitude
    contributions = sorted(
        [(feature_names_out[i], abs(coeffs[i])) for i in range(len(coeffs))],
        key=lambda x: x[1], reverse=True,
    )

    # Show top 3 features that contributed most to this prediction
    top_risk_factors = [
        {"feature": fn, "contribution": round(contrib, 4)}
        for fn, contrib in contributions[:3]
    ]

    return {
        "churn_probability": round(proba, 4),
        "risk_tier": risk_tier,
        "top_risk_factors": top_risk_factors,
    }
