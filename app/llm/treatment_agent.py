"""
Agent 3 - Treatment Agent.

Takes the facts from Agent 1 and the policy retrieved by Agent 2, and produces:
  - four ASC 606 characterization judgments (LLM),
  - the resulting revenue treatment: performance obligations, transaction price,
    recognition method (deterministic rules),
  - a straight-line monthly revenue schedule and opening deferred balance
    (deterministic arithmetic).

The LLM is used only for the four judgments. Everything downstream is plain Python
so the numbers are reproducible and easy to check against ground truth.

Each judgment comes back from the LLM as an object:
    {"reasoning": "...", "kb_basis": ["KB1_3", "KB2_2"], "conclusion": "TRUE"}
The deterministic steps read the "conclusion" field.
"""
import calendar
import json
from datetime import date, timedelta

from app.llm.client import get_llm_response
from app.prompts import treatment


# Number of months billed at contract start, per billing pattern. Drives the
# opening deferred balance: money billed up front before the service is delivered
# is deferred revenue. "full_term" means the whole term is billed on day one.
BILLING_UPFRONT_MONTHS = {
    "annual_upfront": 12,
    "annual_upfront_each_year": 12,
    "quarterly_upfront": 3,
    "monthly": 1,
    "full_term_upfront": "full_term",
    "monthly_arrears": 0,
}


# Helpers for reading Agent 1's (nested, sometimes incomplete) facts.

def _fee_amount(node):
    """Agent 1 stores money as {"amount": x, "currency": ...}; pull the number."""
    if isinstance(node, dict):
        return node.get("amount")
    if isinstance(node, (int, float)):
        return node
    return None


def _looks_like_onboarding(fee):
    name = (fee.get("name") or "").lower()
    return any(k in name for k in
               ("onboard", "implementation", "setup", "training", "migration"))


def _conclusion(judgments, field):
    """Pull the plain conclusion value out of a nested judgment object."""
    node = judgments.get(field) or {}
    if isinstance(node, dict):
        return node.get("conclusion")
    return node   # tolerate a bare string if the model ever returns one



# Step 1 - validate the input coming from Agent 2.

def validate_agent2_output(agent2_output):
    """Make sure Agent 2 handed us usable retrieval before we reason on it."""
    if not isinstance(agent2_output, dict):
        raise ValueError("Agent 2 output must be a dict.")

    chunks = agent2_output.get("retrieved_policy_chunks")
    if not chunks:
        raise ValueError("Agent 2 output has no retrieved_policy_chunks.")

    # We need at least the ASC 606 guidance (KB1) to make any judgment.
    prefixes = {str(c.get("kb_prefix", "")).upper() for c in chunks}
    if "KB1" not in prefixes:
        raise ValueError("Agent 2 retrieved no KB1 (ASC 606) guidance.")

    return True


# Step 2 - characterize the contract with the LLM (the four judgments).

def _policy_context(agent2_output):
    """Flatten Agent 2's retrieved chunks into one block for the prompt.

    Each line keeps its [doc_id] tag (e.g. [KB1_3], [KB2_2]) so the model can
    cite the ids it relied on in kb_basis / reasoning.
    """
    lines = []
    for chunk in agent2_output["retrieved_policy_chunks"]:
        lines.append(f"[{chunk['doc_id']}] {chunk['text']}")
    return "\n\n".join(lines)


def characterize(facts, agent2_output):
    """Ask the LLM for the four characterization judgments, grounded in the policy.

    Returns a dict of judgment objects, each shaped
    {"reasoning": ..., "kb_basis": [...], "conclusion": ...}.
    """
    messages = [
        {"role": "system", "content": treatment.SYSTEM_PROMPT},
        {"role": "user", "content": treatment.USER_PROMPT.format(
            facts_json=json.dumps(facts, indent=2),
            policy_context=_policy_context(agent2_output),
        )},
    ]
    raw = get_llm_response(messages, response_format="json_object")

    # Fallback shape if the model returns invalid or incomplete JSON.
    defaults = {
        "onboarding_distinct": {"reasoning": "", "kb_basis": [], "conclusion": "UNDETERMINED"},
        "usage_model":         {"reasoning": "", "kb_basis": [], "conclusion": "none"},
        "discount_type":       {"reasoning": "", "kb_basis": [], "conclusion": "none"},
        "material_right":      {"reasoning": "", "kb_basis": [], "conclusion": "UNDETERMINED"},
    }
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return defaults

    # Keep only the four keys we asked for; fall back per-field if one is missing.
    return {field: parsed.get(field, default) for field, default in defaults.items()}


# Step 3 - derive the revenue treatment from facts + judgments (rules only).

def _fixed_consideration(facts):
    """Sum the fixed (non-usage) consideration = subscription + onboarding.

    Usage fees are variable consideration and are excluded from the transaction
    price here. Onboarding is included whether or not it is a distinct PO - being
    distinct changes *when* it is recognized, not the total price.
    """
    total = 0.0
    has_onboarding_fee = False
    for fee in facts.get("fees") or []:
        amount = _fee_amount(fee.get("amount"))
        if amount:
            total += amount
            if _looks_like_onboarding(fee):
                has_onboarding_fee = True

    # Agent 1 sometimes reports onboarding outside of fees[]; add it if so.
    if not has_onboarding_fee:
        onboarding = _fee_amount((facts.get("onboarding_terms") or {}).get("fee"))
        if onboarding:
            total += onboarding

    return total


def derive_treatment(facts, judgments):
    """Turn the judgment conclusions into POs, transaction price, and method.

    TODO(draft): two cases are not modeled yet and will be wrong on purpose until
    we add support (the eval will surface them):
      - invalid / undetermined contracts where ground truth is num_POs = 0
        (needs a contract-validity judgment, KB2_15 - not one of our four).
      - multi-service bundles with separately priced support / consulting that
        form their own POs (e.g. C35), which our two levers below cannot produce.
    """
    onboarding_distinct = _conclusion(judgments, "onboarding_distinct")
    material_right = _conclusion(judgments, "material_right")
    usage_model = _conclusion(judgments, "usage_model")

    # Performance obligations. Start from the subscription; add a PO for each
    # judgment that creates one.
    po_list = ["core_subscription"]
    if onboarding_distinct == "TRUE":
        po_list.append("onboarding(distinct)")
    if material_right == "TRUE":
        po_list.append("material_right(renewal)")
    # Usage is variable consideration inside the subscription, not its own PO.

    # Transaction price = fixed consideration only.
    transaction_price = _fixed_consideration(facts)

    # Recognition method. If the contract is genuinely ambiguous upstream, don't
    # force a concrete method - carry the uncertainty forward as UNDETERMINED so a
    # human reviews it, rather than guessing.
    ambiguous = (
        not facts.get("subscription_term_months")   # term not established
        or onboarding_distinct == "UNDETERMINED"     # PO structure unclear
        or material_right == "UNDETERMINED"          # renewal option unclear
    )
    if ambiguous:
        recognition_method = "UNDETERMINED"
    elif usage_model == "pure_usage":
        recognition_method = "over_time_as_usage_occurs"
    elif onboarding_distinct == "TRUE":
        recognition_method = "sub:over_time_ratable; onboarding:point_in_time"
    else:
        recognition_method = "over_time_ratable"

    return {
        "num_POs": len(po_list),
        "po_list": po_list,
        "transaction_price": transaction_price,
        "recognition_method": recognition_method,
        "recognition_period_months": facts.get("subscription_term_months"),
    }


# Step 4 - build the revenue schedule (pure arithmetic, no LLM).

def _parse_date(value):
    """Parse an ISO date string; return None if missing or malformed."""
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _add_months(d, n):
    """Add n calendar months, clamping the day to the end of the target month."""
    idx = d.month - 1 + n
    year = d.year + idx // 12
    month = idx % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def build_schedule(transaction_price, term_months, billing, start_date=None):
    """Straight-line monthly recognition, opening deferred balance, and a
    month-by-month table anchored to the contract start date.

    monthly_revenue  = transaction_price / term_months (straight-line).
    opening_deferred = the amount billed up front before any service is delivered
                       = monthly_revenue * (months billed up front).

    Each schedule row carries period_start / period_end from start_date. Revenue is
    still split evenly across months - this is not day-level proration (KB2_13); a
    mid-month start labels the periods but does not prorate the first month yet.
    """
    if not term_months:
        return {"monthly_revenue": 0.0, "opening_deferred": 0.0, "monthly_schedule": []}

    monthly_revenue = transaction_price / term_months

    upfront = BILLING_UPFRONT_MONTHS.get(billing, 12)
    upfront_months = term_months if upfront == "full_term" else min(upfront, term_months)
    opening_deferred = monthly_revenue * upfront_months

    # Month-by-month recognition table (the "expected revenue schedule").
    start = _parse_date(start_date)
    monthly_schedule = []
    for m in range(int(term_months)):
        entry = {"month": m + 1, "revenue_recognized": round(monthly_revenue, 2)}
        if start:
            entry["period_start"] = _add_months(start, m).isoformat()
            entry["period_end"] = (_add_months(start, m + 1) - timedelta(days=1)).isoformat()
        monthly_schedule.append(entry)

    return {
        "monthly_revenue": monthly_revenue,
        "opening_deferred": opening_deferred,
        "monthly_schedule": monthly_schedule,
    }


def _infer_billing(facts):
    """Rough billing pattern from Agent 1 fee metadata (frequency + timing).

    Agent 1 has no explicit billing field, so this reads fees[].billing_frequency
    and fees[].payment_timing. Multi-year 'each year' vs 'full term upfront' cannot
    be distinguished from this metadata - flagged as an upstream Agent 1 gap.
    """
    fees = facts.get("fees") or []
    timings = " ".join((f.get("payment_timing") or "") for f in fees).lower()
    freqs = " ".join((f.get("billing_frequency") or "") for f in fees).lower()

    if "arrear" in timings:
        return "monthly_arrears"
    if "quarter" in freqs:
        return "quarterly_upfront"
    if "annual" in freqs or "advance" in timings:
        return "annual_upfront"
    if "month" in freqs:
        return "monthly_arrears"   # monthly without "in advance" -> treat as arrears
    return "annual_upfront"


# Orchestration - run the full agent on one contract.

def run(agent1_output, agent2_output):
    """Run Agent 3 end to end for a single contract."""
    facts = (agent1_output.get("validated_output")
             or agent1_output.get("candidate_output")
             or agent1_output)

    # Step 1: input from Agent 2 is usable.
    validate_agent2_output(agent2_output)

    # Step 2: LLM characterization judgments.
    judgments = characterize(facts, agent2_output)

    # Step 3: deterministic treatment.
    treatment_values = derive_treatment(facts, judgments)

    # Step 4: deterministic schedule.
    billing = _infer_billing(facts)
    schedule = build_schedule(
        treatment_values["transaction_price"],
        treatment_values["recognition_period_months"],
        billing,
        facts.get("start_date"),
    )

    return {
        "contract_id": facts.get("contract_id"),
        "source_file": facts.get("source_file"),
        "billing_used": billing,
        "characterization": judgments,
        "treatment": treatment_values,
        "schedule": schedule,
    }


if __name__ == "__main__":
    # Smoke test on the first extracted contract, using Agent 2 live.
    from pathlib import Path
    from app.llm.policy_retrieval import PolicyRetrievalAgent

    extracted = json.loads(
        (Path(__file__).parent / "extracted_contracts.json").read_text(encoding="utf-8")
    )
    agent2 = PolicyRetrievalAgent()

    record = next(r for r in extracted if r["file_name"].startswith("C11"))
    facts = record.get("validated_output") or record.get("candidate_output") or {}
    result = run(record, agent2.run(facts))
    print(json.dumps(result, indent=2))