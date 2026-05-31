#!/usr/bin/env python3
"""Rebuild model_pipeline.pkl from test_datafile.csv.

Matches notebook preprocessing exactly:
- TitleCase category cleaning (Fiber optic, Credit card)
- Corrupted numeric detection + median imputation
- Engineered features: billing_ratio, service_density
- Pipeline: StandardScaler + OneHotEncoder -> LogisticRegression

Run:  uv run python scripts/rebuild_pipeline.py
"""
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── 1. Load + clean categoricals (notebook-style TitleCase maps) ──

df = pd.read_csv("test_datafile.csv")

gender_map = {
    "Male": "Male", "MALE": "Male", "M": "Male", "male": "Male",
    "Female": "Female", "FEMALE": "Female", "f": "Female", "F": "Female",
    "Other": "Other", "other": "Other",
}
internet_map = {
    "DSL": "DSL", "dsl": "DSL",
    "Fiber optic": "Fiber optic", "fiber": "Fiber optic",
    "No": "No", "no": "No", "None": "No",
    "nan": "Unknown",
}
phone_map = {
    "Yes": "Yes", "yes": "Yes", "Y": "Yes",
    "No": "No", "no": "No", "N": "No",
}
payment_map = {
    "Credit card": "Credit card", "credit card": "Credit card",
    "CC": "Credit card", "cc": "Credit card",
    "Bank transfer": "Bank transfer", "bank transfer": "Bank transfer",
    "BT": "Bank transfer", "bt": "Bank transfer",
    "Electronic check": "Electronic check", "electronic check": "Electronic check",
    "Mailed check": "Mailed check", "mailed check": "Mailed check",
}

df["gender_clean"] = df["gender"].map(gender_map).fillna("Unknown")
df["internet_clean"] = df["internet_service"].map(internet_map).fillna("Unknown")
df["phone_clean"] = df["phone_service"].map(phone_map).fillna("Unknown")
df["payment_clean"] = df["payment_method"].map(payment_map).fillna("Unknown")
df["contract_type"] = df["contract_type"].str.lower().str.strip()

# ── 2. Fix corrupted numerics ──

corruption = {
    "satisfaction_score": lambda v: (v < 0) | (v > 10),
    "avg_monthly_gb_used": lambda v: v < 0,
    "num_support_tickets": lambda v: (v < 0) | (v > 10),
    "age": lambda v: (v < 18) | (v > 100),
    "tenure_months": lambda v: v < 0,
}
for col, mask_fn in corruption.items():
    df.loc[mask_fn(df[col]) | df[col].isna(), col] = np.nan

numeric_cols = [
    "satisfaction_score", "avg_monthly_gb_used", "num_support_tickets",
    "age", "tenure_months", "monthly_charges", "total_charges",
    "avg_monthly_minutes", "num_additional_services",
]
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# ── 3. Engineered features ──

tenure_s = df["tenure_months"].replace(0, 1)
charges_s = df["monthly_charges"].replace(0, 1)
df["billing_ratio"] = df["total_charges"] / (charges_s * tenure_s)
df["billing_ratio"] = df["billing_ratio"].clip(
    lower=df["billing_ratio"].quantile(0.01),
    upper=df["billing_ratio"].quantile(0.99),
)
df["billing_ratio"] = df["billing_ratio"].fillna(1.0)

df["service_density"] = (
    (df["internet_clean"].isin(["DSL", "Fiber optic"])).astype(int)
    + (df["phone_clean"] == "Yes").astype(int)
)

# ── 4. Feature matrix ──

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

X = df[num_cols + cat_cols]
y = df["churned"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42,
)

# ── 5. Train pipeline ──

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(
        drop="first", handle_unknown="ignore", sparse_output=False
    ), cat_cols),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=42,
    )),
])

pipeline.fit(X_train, y_train)

# ── 6. Evaluate ──

y_pred = pipeline.predict(X_test)
rec = recall_score(y_test, y_pred)
print(f"Test recall: {rec:.4f}")
print(classification_report(y_test, y_pred, digits=4))

# ── 7. Save ──

import os

os.makedirs("artifacts", exist_ok=True)

with open("artifacts/model_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

medians = {col: float(df[col].median()) for col in numeric_cols}
with open("artifacts/feature_names.pkl", "wb") as f:
    pickle.dump({
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "feature_order": num_cols + cat_cols,
        "medians": medians,
    }, f)

print("\nSaved artifacts/model_pipeline.pkl + feature_names.pkl")
print("Agent is ready to use real predictions.")
