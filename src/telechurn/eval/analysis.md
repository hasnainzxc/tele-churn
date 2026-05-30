# Part 2.4: Results & Analysis

## Aggregate Scorecard

| Metric | Score | Notes |
|--------|-------|-------|
| **Tool Selection Accuracy** | 0.800 | 12/15 cases called expected tools |
| **Parameter Precision** | 0.822 | 82% of expected params passed correctly |
| **Response Completeness** | 0.900 | Strong response quality across all cases |
| **Judge Overall Mean** | 3.94/5 | LLM-as-judge average across 4 dimensions |

### Judge Scores by Dimension

| Dimension | Mean | Min | Max |
|-----------|------|-----|-----|
| Factual Correctness | 3.93 | 1 | 5 |
| Tool Use Appropriateness | 4.07 | 1 | 5 |
| Actionability | 3.67 | 1 | 5 |
| Hallucination | 4.07 | 1 | 5 |

### Tool Selection Accuracy by Category

| Category | Score | Cases |
|----------|-------|-------|
| ambiguous_input | 1.000 | TC-005, TC-006 |
| model_disagreement | 1.000 | TC-009 |
| multi_step_chaining | 1.000 | TC-003, TC-004, TC-014, TC-015 |
| out_of_scope | 1.000 | TC-010 |
| adversarial_edge_case | 0.667 | TC-011, TC-012, TC-013 |
| escalation_trigger | 0.500 | TC-007, TC-008 |
| single_tool_happy_path | 0.500 | TC-001, TC-002 |

## Success Cases

### TC-003: Multi-step Full Retention Flow
**Input**: "Customer TC-004711 is on the phone. They're month-to-month. Pull their profile, check their churn risk, and tell me what offers I should present."

**What went right**: The agent correctly chained all three tools in sequence — lookup_customer → predict_churn → get_retention_offers — then synthesized a coherent, actionable recommendation. It identified TC-004711 as medium risk (45% churn probability), cited specific risk factors (month-to-month contract, low tenure, high charges), and presented concrete offers (15% monthly discount, free premium channels). All 4 judge dimensions scored 5/5.

**Why it worked**: The user input contained all necessary context (customer ID, contract type, clear intent). The ReAct loop's router correctly decomposed the multi-step request, and the response node synthesized tool outputs into a rep-ready recommendation rather than a data dump.

### TC-005: Ambiguous Input — No Customer ID
**Input**: "I have a high-risk customer on the phone. What offers should I give?"

**What went right**: The agent recognized the missing customer ID and asked clarifying questions instead of blindly calling tools. It responded with structured guidance: confirm the customer ID first, then proceed with lookup → churn prediction → offers. Perfect scores across all metrics and all 4 judge dimensions at 5/5.

**Why it worked**: The system prompt explicitly instructs "If no customer_id is provided, ASK for it. Do not guess." The LLM followed this rule exactly, demonstrating that well-crafted system prompts can reliably prevent tool hallucination on incomplete inputs.

## Failure Cases

### TC-002: Single-tool Offers Without Customer ID
**Input**: "What offers are available for a low-risk customer on month-to-month contract?"

**What happened**: Tool selection accuracy 0.0, param precision 0.0. The agent attempted to look up a customer first (because the system prompt says "always look up a customer before getting offers"), but there was no customer ID to work with. It returned an error requesting the customer ID and never called get_retention_offers.

**Root cause**: The system prompt rule "Always look up a customer before predicting churn or getting offers for them" is too rigid. This query is a hypothetical/general inquiry — the rep wants to know what offers exist for a risk tier + contract combination, not for a specific customer. The agent should handle general offer inquiries without requiring a customer ID.

**Fix**: Add explicit guidance to the system prompt: "If the user is asking about what retention offers exist in general (without specifying a customer ID), you can call get_retention_offers directly. Only require customer lookup when the query is about a specific customer." This is a 2-line change to SYSTEM_PROMPT in agent.py.

### TC-008: Complex Dispute Escalation
**Input**: "Customer TC-001425 has billing issues going back 6 months and multiple unresolved tickets. Handle this."

**What happened**: Tool selection accuracy 0.0 (expected: lookup + escalate; actual: lookup only). The agent looked up the customer but then failed to escalate. It reported "unable to retrieve customer profile" (because the mock returns fake data) and asked for more information instead of recognizing the escalation trigger.

**Root cause**: Two issues compounding:
1. The mock customer data (`df=None` fallback) always returns a clean profile — there's no real billing history to flag. The agent sees a healthy customer and doesn't detect a dispute pattern.
2. The escalation trigger detection relies entirely on the LLM's interpretation of natural language. "Billing issues going back 6 months and multiple unresolved tickets" should be enough — but when the agent looks up the customer and sees clean data, it overrides the LLM's initial concern.

**Fix**: Inject real data in eval or strengthen the system prompt with explicit escalation keywords: "If the rep mentions billing disputes, multiple unresolved tickets, or issues spanning multiple months, escalate even if the customer profile looks normal." A more robust approach would be a pre-processing step that scans user input for escalation keywords before entering the ReAct loop.

## Production Roadmap

To run this evaluation pipeline in CI/CD at scale: containerize the eval runner (Docker image with pinned dependencies), execute it as a GitHub Actions job on every PR targeting main, store results as timestamped JSON artifacts in S3/GCS, and publish a summary comment back to the PR. Add latency tracking (p50/p95 per case), track score trends over time to detect regressions, and calibrate the LLM judge against 20+ human-labeled examples with inter-rater reliability metrics (Cohen's kappa). For production use, replace the heuristic completeness scoring with a second LLM call that evaluates response quality against the quality_criteria field already defined in each test case.
