"""Tool definitions for the retention agent — Pydantic schemas for LangGraph tool calling."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# These models are the contract between the agent graph and the LLM.
# The LLM sees them as JSON schemas via LangChain's bind_tools(),
# so field descriptions become part of the tool-calling prompt. Keep 'em tight.

class CustomerProfile(BaseModel):
    # Flattened view of what the lookup returns. All fields Optional because
    # the lookup can fail or the dataset is messy and columns go missing.
    customer_id: str
    age: float | None = None
    gender: str | None = None
    tenure_months: float | None = None
    contract_type: str | None = None
    monthly_charges: float | None = None
    total_charges: float | None = None
    internet_service: str | None = None
    phone_service: str | None = None
    avg_monthly_gb_used: float | None = None
    num_support_tickets: float | None = None
    avg_monthly_minutes: float | None = None
    satisfaction_score: float | None = None
    payment_method: str | None = None
    num_additional_services: int | None = None
    last_interaction_date: str | None = None


class ChurnPrediction(BaseModel):
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    # The regex constraint here is belt-and-suspenders — the model already
    # bins probabilities into tiers, but this catches any output corruption.
    risk_tier: str = Field(..., pattern="^(high|medium|low)$")
    top_risk_factors: list[dict[str, Any]]


class RetentionOffer(BaseModel):
    id: str
    name: str
    description: str
    offer_type: str
    cost_to_company: str
    expected_retention_lift: str


class LookupCustomerInput(BaseModel):
    customer_id: str = Field(..., description="Customer ID in format TC-XXXXXX")


class LookupCustomerOutput(BaseModel):
    customer: CustomerProfile | None
    error: str | None = None  # non-None when lookup fails (bad ID, missing row, etc.)


class PredictChurnInput(BaseModel):
    customer_data: dict[str, Any] = Field(
        ..., description="Raw customer feature dictionary from lookup"
    )


class PredictChurnOutput(BaseModel):
    prediction: ChurnPrediction | None
    error: str | None = None  # set if predict_fn blows up (model not loaded, bad features, etc.)


class GetRetentionOffersInput(BaseModel):
    risk_tier: str = Field(..., description="One of: high, medium, low")
    contract_type: str = Field(..., description="One of: Month-to-month, One year, Two year")


class GetRetentionOffersOutput(BaseModel):
    offers: list[RetentionOffer]
    risk_tier: str
    contract_type: str


class LogInteractionInput(BaseModel):
    customer_id: str
    outcome: str = Field(..., description="Resolution summary")
    offers_presented: list[str] = Field(default_factory=list)
    escalation_reason: str | None = None
    notes: str = ""


class LogInteractionOutput(BaseModel):
    log_id: str
    timestamp: str
    status: str = "logged"


class EscalateToSupervisorInput(BaseModel):
    customer_id: str
    reason: str = Field(
        ..., description="Why escalation is needed: legal_threat, complex_dispute, out_of_scope, etc."
    )
    context_summary: str = Field(..., description="Brief summary of conversation and relevant facts")
    # HACK: defaulting priority to "normal" so the LLM doesn't have to provide it
    priority: str = Field(default="normal", pattern="^(low|normal|high|critical)$")


class EscalateToSupervisorOutput(BaseModel):
    escalation_id: str
    timestamp: str
    status: str = "escalated"
    message: str
