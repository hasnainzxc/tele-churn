"""Tests for retention agent tools."""

from telechurn.agent.tools import (
    ChurnPrediction,
    CustomerProfile,
    GetRetentionOffersInput,
    RetentionOffer,
)


class TestCustomerProfile:
    def test_valid_profile(self):
        p = CustomerProfile(customer_id="TC-000001", age=35.0, gender="Male")
        assert p.customer_id == "TC-000001"
        assert p.age == 35.0

    def test_nullable_fields(self):
        p = CustomerProfile(customer_id="TC-000001")
        assert p.age is None
        assert p.gender is None


class TestChurnPrediction:
    def test_valid_prediction(self):
        p = ChurnPrediction(
            churn_probability=0.75,
            risk_tier="high",
            top_risk_factors=[{"feature": "tenure", "contribution": "low"}],
        )
        assert p.risk_tier == "high"
        assert 0 <= p.churn_probability <= 1


class TestGetRetentionOffersInput:
    def test_valid_input(self):
        inp = GetRetentionOffersInput(risk_tier="medium", contract_type="Month-to-month")
        assert inp.risk_tier == "medium"


class TestRetentionOffer:
    def test_valid_offer(self):
        o = RetentionOffer(
            id="TEST_01",
            name="Test Offer",
            description="A test",
            offer_type="discount",
            cost_to_company="$0",
            expected_retention_lift="+10%",
        )
        assert o.id == "TEST_01"
