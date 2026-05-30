"""Retention offer catalog — realistic offers by risk tier and contract type."""

OFFERS = {
    "low": {
        "all": [
            {
                "id": "LOYALTY_REWARD_5",
                "name": "5% Loyalty Discount",
                "description": "Automatic 5% monthly discount applied for being a valued long-term customer.",
                "cost_to_company": "5% MRR reduction",
                "expected_retention_lift": "+8%",
                "offer_type": "discount",
            },
            {
                "id": "REFERRAL_BONUS",
                "name": "$50 Referral Credit",
                "description": "Earn $50 credit per friend who signs up with your referral code.",
                "cost_to_company": "$50 one-time",
                "expected_retention_lift": "+3%",
                "offer_type": "credit",
            },
        ],
        "Month-to-month": [
            {
                "id": "ANNUAL_DISCOUNT_10",
                "name": "Switch to Annual — 10% Off",
                "description": "Move from month-to-month to annual billing and save 10% on monthly rate.",
                "cost_to_company": "10% MRR reduction offset by reduced churn",
                "expected_retention_lift": "+15%",
                "offer_type": "contract_upgrade",
            },
        ],
        "One year": [
            {
                "id": "PREMIUM_SUPPORT_FREE",
                "name": "Free Premium Support (3 months)",
                "description": "Complimentary access to 24/7 premium technical support for 3 months.",
                "cost_to_company": "$15/mo support cost",
                "expected_retention_lift": "+10%",
                "offer_type": "service_upgrade",
            },
        ],
        "Two year": [
            {
                "id": "SPEED_BOOST_6MO",
                "name": "6-Month Speed Boost",
                "description": "Free speed tier upgrade for 6 months. Enjoy faster internet at no extra cost.",
                "cost_to_company": "$20/mo bandwidth",
                "expected_retention_lift": "+12%",
                "offer_type": "service_upgrade",
            },
        ],
    },
    "medium": {
        "all": [
            {
                "id": "DISCOUNT_15",
                "name": "15% Monthly Discount (6 months)",
                "description": "15% off your monthly bill for the next 6 months if you stay with us.",
                "cost_to_company": "15% MRR reduction for 6mo",
                "expected_retention_lift": "+20%",
                "offer_type": "discount",
            },
            {
                "id": "FREE_PREMIUM_CHANNELS",
                "name": "3 Months Free Premium Channels",
                "description": "Access to premium channel package at no cost for 3 months.",
                "cost_to_company": "$12/mo content licensing",
                "expected_retention_lift": "+10%",
                "offer_type": "service_upgrade",
            },
        ],
        "Month-to-month": [
            {
                "id": "ANNUAL_SWITCH_20",
                "name": "Switch to Annual — 20% Off + Free Install",
                "description": "Lock in annual billing at 20% discount with free professional installation of any new equipment.",
                "cost_to_company": "20% MRR reduction, waived install fee",
                "expected_retention_lift": "+25%",
                "offer_type": "contract_upgrade",
            },
        ],
        "One year": [
            {
                "id": "EQUIPMENT_UPGRADE",
                "name": "Free Equipment Upgrade",
                "description": "Complimentary upgrade to latest modem/router hardware (no contract extension).",
                "cost_to_company": "$150 hardware + install",
                "expected_retention_lift": "+18%",
                "offer_type": "hardware",
            },
        ],
        "Two year": [
            {
                "id": "MONTH_FREE",
                "name": "One Month Free + Priority Queue",
                "description": "One full month free and priority customer support queue for 12 months.",
                "cost_to_company": "1 month ARPU + priority support overhead",
                "expected_retention_lift": "+15%",
                "offer_type": "bundle",
            },
        ],
    },
    "high": {
        "all": [
            {
                "id": "RETENTION_EXECUTIVE_CALL",
                "name": "Personal Executive Retention Call",
                "description": "A senior retention specialist will call within 1 business day with a personalised retention package including up to 30% discount, free equipment, and priority support.",
                "cost_to_company": "Variable — up to 30% MRR reduction + hardware cost",
                "expected_retention_lift": "+35%",
                "offer_type": "high_touch",
            },
            {
                "id": "CUSTOM_PLAN",
                "name": "Custom Service Plan",
                "description": "Design a custom plan matching exactly what you need — speed, channels, price. Our retention team will build it for you.",
                "cost_to_company": "Variable — custom negotiation",
                "expected_retention_lift": "+30%",
                "offer_type": "high_touch",
            },
        ],
        "Month-to-month": [
            {
                "id": "SAVE30_ANNUAL",
                "name": "Save 30% — Switch to Annual",
                "description": "30% discount on annual plan plus first month free and waived all fees.",
                "cost_to_company": "30% MRR reduction + 1 month free + waived fees",
                "expected_retention_lift": "+40%",
                "offer_type": "contract_upgrade",
            },
        ],
        "One year": [
            {
                "id": "LOCK_RATE_2YR",
                "name": "Lock Current Rate for 2 Years + $200 Credit",
                "description": "2-year rate lock at current price plus $200 account credit applied immediately.",
                "cost_to_company": "$200 one-time, inflation risk on rate lock",
                "expected_retention_lift": "+35%",
                "offer_type": "bundle",
            },
        ],
        "Two year": [
            {
                "id": "VIP_UPGRADE",
                "name": "VIP Service Upgrade + $300 Credit",
                "description": "Premium support tier, top-speed internet, all premium channels, plus $300 credit. Best retention package available.",
                "cost_to_company": "Full upgrade cost + $300 credit",
                "expected_retention_lift": "+45%",
                "offer_type": "bundle",
            },
        ],
    },
}


def get_offers(risk_tier: str, contract_type: str) -> list[dict]:
    """Return matching offers for given risk tier and contract type."""
    risk_tier = risk_tier.lower()
    tier_offers = OFFERS.get(risk_tier, OFFERS["medium"])
    contract_type_clean = contract_type.strip().title()
    all_offers = tier_offers.get("all", [])
    specific = tier_offers.get(contract_type_clean, [])
    return all_offers + specific
