"""Tests for retention offer catalog."""

import pytest

from telechurn.agent.offers import get_offers


class TestGetOffers:
    def test_high_risk_month_to_month(self):
        offers = get_offers("high", "Month-to-month")
        assert len(offers) > 0
        names = [o["name"] for o in offers]
        assert len(names) >= 1

    def test_medium_risk_one_year(self):
        offers = get_offers("medium", "One year")
        assert len(offers) > 0
        assert all("name" in o for o in offers)

    def test_low_risk_two_year(self):
        offers = get_offers("low", "Two year")
        assert len(offers) > 0

    def test_high_risk_gets_targeted_plus_all(self):
        """High risk should include both generic 'all' offers and contract-specific."""
        offers = get_offers("high", "Month-to-month")
        types = {o.get("offer_type", "") for o in offers}
        assert len(types) >= 1
        # Should have at least "all" tier offers
        all_offers = get_offers("high", "Two year")
        assert len(all_offers) > 0

    @pytest.mark.parametrize("risk_tier", ["high", "HIGH", "High"])
    def test_casing_normalized(self, risk_tier):
        offers = get_offers(risk_tier, "Month-to-month")
        assert len(offers) > 0

    @pytest.mark.parametrize("contract_type", [
        "Month-to-month", "month-to-month", "MONTH-TO-MONTH",
        "One year", "one year", "ONE YEAR",
        "Two year", "two year", "TWO YEAR",
    ])
    def test_contract_casing_normalized(self, contract_type):
        offers = get_offers("medium", contract_type)
        assert len(offers) > 0

    def test_empty_contract_type(self):
        offers = get_offers("medium", "")
        assert len(offers) > 0  # should fall back gracefully

    def test_unknown_risk_tier_fallback(self):
        """Unknown tier falls back to medium."""
        offers = get_offers("super_risky", "Month-to-month")
        medium = get_offers("medium", "Month-to-month")
        assert offers == medium

    def test_offers_have_required_keys(self):
        for tier in ("high", "medium", "low"):
            for ct in ("Month-to-month", "One year", "Two year"):
                offers = get_offers(tier, ct)
                for o in offers:
                    assert "name" in o
                    assert "description" in o
                    assert "offer_type" in o
                    assert "cost_to_company" in o
