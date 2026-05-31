"""Tests for predict_churn pipeline."""

from pathlib import Path

import pandas as pd

from telechurn.predict import (
    ARTIFACT_PATH,
    predict_churn,
    preprocess_customer,
)


def make_sample_customer(**overrides) -> dict:
    """Build minimal input dict with realistic defaults."""
    defaults = {
        "customer_id": "TC-000001",
        "age": 42,
        "gender": "Male",
        "tenure_months": 36,
        "contract_type": "Month-to-month",
        "monthly_charges": 70.0,
        "total_charges": 2520.0,
        "internet_service": "Fiber optic",
        "phone_service": "Yes",
        "avg_monthly_gb_used": 15.0,
        "num_support_tickets": 2,
        "avg_monthly_minutes": 500.0,
        "satisfaction_score": 7,
        "payment_method": "Credit card",
        "num_additional_services": 2,
    }
    defaults.update(overrides)
    return defaults


class TestPreprocessCustomer:
    def test_normalizes_categories(self):
        """Category maps convert raw strings to notebook-compatible forms."""
        raw = make_sample_customer(
            gender="F", internet_service="fiber", phone_service="N",
            payment_method="CC",
        )
        df = preprocess_customer(raw)
        assert df["gender_clean"].iloc[0] == "Female"
        assert df["internet_clean"].iloc[0] == "Fiber optic"
        assert df["phone_clean"].iloc[0] == "No"
        assert df["payment_clean"].iloc[0] == "Credit card"

    def test_nullifies_corrupted_numerics(self):
        """Values outside valid ranges are set to NaN for imputation."""
        raw = make_sample_customer(
            satisfaction_score=99, avg_monthly_gb_used=-10,
            num_support_tickets=-5, age=-1, tenure_months=-12,
        )
        df = preprocess_customer(raw)
        # Corrupted values should be NaN (replaced by medians)
        assert pd.notna(df["satisfaction_score"].iloc[0])
        assert pd.notna(df["avg_monthly_gb_used"].iloc[0])
        assert pd.notna(df["num_support_tickets"].iloc[0])

    def test_computes_engineered_features(self):
        """billing_ratio and service_density are computed."""
        df = preprocess_customer(make_sample_customer())
        assert "billing_ratio" in df.columns
        assert "service_density" in df.columns
        # billing_ratio = total / (monthly * tenure)
        # 2520 / (70 * 36) = 1.0
        assert abs(df["billing_ratio"].iloc[0] - 1.0) < 0.01

    def test_handles_missing_optional_cols(self):
        """predict_churn fills missing columns with defaults, doesn't crash."""
        raw = make_sample_customer()
        del raw["num_additional_services"]
        del raw["avg_monthly_minutes"]
        result = predict_churn(raw)
        assert "churn_probability" in result
        assert 0.0 <= result["churn_probability"] <= 1.0

    def test_unknown_categories_map_to_unknown(self):
        """Unknown category values get mapped to 'Unknown'."""
        raw = make_sample_customer(
            gender="Unicorn", internet_service="Satellite",
            payment_method="Bitcoin",
        )
        df = preprocess_customer(raw)
        assert df["gender_clean"].iloc[0] == "Unknown"
        assert df["internet_clean"].iloc[0] == "Unknown"
        assert df["payment_clean"].iloc[0] == "Unknown"


class TestPredictChurn:
    def test_returns_correct_structure(self):
        """predict_churn returns the three required keys."""
        result = predict_churn(make_sample_customer())
        assert "churn_probability" in result
        assert "risk_tier" in result
        assert "top_risk_factors" in result
        assert 0.0 <= result["churn_probability"] <= 1.0
        assert result["risk_tier"] in ("high", "medium", "low")
        assert isinstance(result["top_risk_factors"], list)
        assert len(result["top_risk_factors"]) == 3

    def test_scores_different_customers(self):
        """High-risk and low-risk profiles get meaningfully different scores."""
        high = predict_churn(make_sample_customer(
            contract_type="Month-to-month", tenure_months=2,
            satisfaction_score=2, num_support_tickets=8,
        ))
        low = predict_churn(make_sample_customer(
            contract_type="Two year", tenure_months=48,
            satisfaction_score=9, num_support_tickets=0,
        ))
        assert high["churn_probability"] > low["churn_probability"]

    def test_handles_contract_casing(self):
        """contract_type casing doesn't break prediction."""
        for ct in ("month-to-month", "Month-to-month", "MONTH-TO-MONTH",
                    "One year", "one year", "ONE YEAR",
                    "Two year", "two year", "TWO YEAR"):
            result = predict_churn(make_sample_customer(contract_type=ct))
            assert "churn_probability" in result

    def test_missing_artifact_returns_placeholder(self):
        """When artifact file doesn't exist, returns informative fallback."""
        original = ARTIFACT_PATH
        try:
            import telechurn.predict as pred_mod
            pred_mod.ARTIFACT_PATH = Path("/nonexistent/path.pkl")
            result = predict_churn({"age": 30})
            assert result["churn_probability"] == 0.45
            assert result["risk_tier"] == "medium"
            assert result["top_risk_factors"][0]["feature"] == "model_not_available"
        finally:
            pred_mod.ARTIFACT_PATH = original

    def test_pipeline_loaded_once_per_call(self):
        """Multiple calls to predict_churn reload pipeline each time (stateless design)."""
        r1 = predict_churn(make_sample_customer())
        r2 = predict_churn(make_sample_customer())
        assert r1.keys() == r2.keys()
