"""Retention offer catalog loaded from external JSON."""

import json
from pathlib import Path

_OFFERS_PATH = Path(__file__).resolve().parents[3] / "artifacts" / "offers.json"


def _load_offers() -> dict:
    if _OFFERS_PATH.exists():
        return json.loads(_OFFERS_PATH.read_text())
    return _FALLBACK_OFFERS


def get_offers(risk_tier: str, contract_type: str) -> list[dict]:
    """Return matching offers for given risk tier and contract type."""
    offers = _load_offers()
    risk_tier = risk_tier.lower()
    tier_offers = offers.get(risk_tier, offers.get("medium", {}))

    contract_type_clean = contract_type.strip().replace("-", " ").title().replace(" ", "-")

    all_offers = tier_offers.get("all", [])
    specific = tier_offers.get(contract_type_clean, [])
    return all_offers + specific


# Minimal fallback — full catalog is in artifacts/offers.json.
# This exists so the module never fails to import.
_FALLBACK_OFFERS = {
    "medium": {
        "all": [
            {
                "id": "DISCOUNT_15",
                "name": "15% Monthly Discount (6 months)",
                "description": "15% off monthly bill for 6 months.",
                "cost_to_company": "15% MRR reduction for 6mo",
                "expected_retention_lift": "+20%",
                "offer_type": "discount",
            },
        ],
    },
}
