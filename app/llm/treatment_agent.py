"""
Agent 3 - Treatment Agent.

Takes the facts from Agent 1 and the policy retrieved by Agent 2, and produces:
  - ASC 606 characterization judgments and the transaction price (LLM),
  - a focused LLM pass that reads the fee table into structured components,
  - the resulting performance-obligation count and recognition method (deterministic rules),
  - a straight-line monthly revenue schedule and opening deferred balance
    (deterministic arithmetic).

The LLM makes the seven judgments (six characterizations plus the transaction price) and reads
the fee table; the PO count, recognition method, and schedule are plain Python so those numbers
are reproducible and easy to check. onboarding_distinct is then overridden by a deterministic
rule (see apply_onboarding_rule), not left to the LLM.

Each judgment comes back from the LLM as an object:
    {"reasoning": "...", "kb_basis": ["KB1_3", "KB2_2"], "conclusion": "TRUE"}
The deterministic steps read the "conclusion" field.
"""
import calendar
import json
import re
from datetime import date, timedelta

from app.llm.client import get_llm_response
from app.prompts import treatment


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
    if not any(k in name for k in
               ("onboard", "implementation", "setup", "training", "migration")):
        return False
    # A line that also names the subscription is a combined offering (e.g. "Subscription &
    # Implementation"), not a separately priced onboarding fee, so it does not create a
    # standalone onboarding obligation.
    return "subscription" not in name


def _has_onboarding_fee(facts):
    """True only when the contract actually carries an onboarding/implementation fee.

    Guards against the model returning onboarding_distinct=TRUE on a contract that has no
    onboarding at all - without an actual fee there is no separate obligation to recognize.
    """
    for fee in facts.get("fees") or []:
        if _looks_like_onboarding(fee) and _fee_amount(fee.get("amount")):
            return True
    return bool(_fee_amount((facts.get("onboarding_terms") or {}).get("fee")))


def _conclusion(judgments, field):
    """Pull the plain conclusion value out of a nested judgment object."""
    node = judgments.get(field) or {}
    if isinstance(node, dict):
        return node.get("conclusion")
    return node   # tolerate a bare string if the model ever returns one


def _parse_number(value):
    """Coerce an LLM conclusion ('$45,000', '45000', 45000) to a float, or None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[,$\s]", "", value)
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None



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


# Step 2 - the LLM step: make the characterization judgments.

def _policy_context(agent2_output):
    """Flatten Agent 2's retrieved chunks into one block for the prompt.

    Each line keeps its [doc_id] tag (e.g. [KB1_3], [KB2_2]) so the model can
    cite the ids it relied on in kb_basis / reasoning.
    """
    lines = []
    for chunk in agent2_output["retrieved_policy_chunks"]:
        lines.append(f"[{chunk['doc_id']}] {chunk['text']}")
    return "\n\n".join(lines)


def llm_judgments(facts, agent2_output, raw_text="", fee_components=None):
    """Ask the LLM for the seven judgments (six characterizations plus transaction_price),
    grounded in the policy.

    The structured facts are the primary evidence; the contract text is passed as
    supplementary context so the model can catch qualifying wording the extraction missed.

    Returns a dict of judgment objects, each shaped
    {"reasoning": ..., "kb_basis": [...], "conclusion": ...}.
    """
    messages = [
        {"role": "system", "content": treatment.SYSTEM_PROMPT},
        {"role": "user", "content": treatment.USER_PROMPT.format(
            facts_json=json.dumps(facts, indent=2),
            fee_components=(json.dumps(fee_components, indent=2) if fee_components
                            else "(none - use the fee list in the facts above)"),
            policy_context=_policy_context(agent2_output),
            contract_text=(raw_text or "")[:6000],
        )},
    ]
    raw = get_llm_response(messages, response_format="json_object")

    # Fallback shape if the model returns invalid or incomplete JSON. contract_valid
    # defaults to TRUE so a missing/garbled response never zeroes out a real contract.
    defaults = {
        "onboarding_distinct": {"reasoning": "", "kb_basis": [], "conclusion": "UNDETERMINED"},
        "usage_model":         {"reasoning": "", "kb_basis": [], "conclusion": "none"},
        "discount_type":       {"reasoning": "", "kb_basis": [], "conclusion": "none"},
        "material_right":      {"reasoning": "", "kb_basis": [], "conclusion": "UNDETERMINED"},
        "contract_valid":      {"reasoning": "", "kb_basis": [], "conclusion": "TRUE"},
        "additional_distinct_services": {"reasoning": "", "kb_basis": [], "conclusion": []},
        "transaction_price":   {"reasoning": "", "kb_basis": [], "conclusion": None},
    }
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return defaults

    # Keep only the keys we asked for; fall back per-field if one is missing.
    result = {field: parsed.get(field, default) for field, default in defaults.items()}
    # Normalize the boolean-style conclusions to the allowed vocabulary so the rules layer,
    # which compares by exact string, does not throw away a correct answer on formatting
    # (e.g. the model emitting "DISTINCT" instead of "TRUE").
    for f in ("onboarding_distinct", "material_right", "contract_valid"):
        node = result.get(f)
        if isinstance(node, dict) and "conclusion" in node:
            node["conclusion"] = _normalize_bool_conclusion(node["conclusion"])
    return result


def _normalize_bool_conclusion(value):
    """Map a free-form TRUE/FALSE/UNDETERMINED/n-a conclusion to the allowed vocabulary.
    Order matters: n/a and undetermined are checked first, and negations before positives,
    because 'not distinct' contains the substring 'distinct'."""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    s = str(value).strip().lower()
    if s in ("n/a", "na", "n.a.", "none", "nil") or "not applicable" in s:
        return "n/a"
    if any(k in s for k in ("undetermined", "unclear", "cannot")):
        return "UNDETERMINED"
    if "not distinct" in s or "non-distinct" in s or s in ("false", "no", "not distinct"):
        return "FALSE"
    if "distinct" in s or "true" in s or s == "yes":
        return "TRUE"
    return value


# Step 3 - derive the revenue treatment from facts + judgments (rules only).

def _is_noise_fee_name(name):
    """Agent 1 sometimes emits a 'fee' whose name is a run-on chunk of contract prose with a
    number scraped out of it. A real fee label is short; prose is not."""
    n = (name or "").strip()
    if not n:
        return True
    if len(n.split()) > 9:                 # real labels are short
        return True
    if n[0].islower() or n[0] == "$":      # prose fragment / bare amount
        return True
    if re.match(r"^\d+\.\d", n):           # section number like "3.1 ..."
        return True
    return False


def _fee_components(facts):
    """Return (recurring_per_year, one_time) from the cleaned fee list.

    Recurring = the annual subscription (counted once per contract year); one-time =
    onboarding / implementation (counted once). Extraction noise is filtered out, fees are
    deduped by NAME (not amount, which threw away real fees that happened to share a value),
    and the renewal price is excluded. If filtering removes everything, fall back to the raw
    fees (a few contracts have only prose-named fee lines)."""
    fees = facts.get("fees") or []
    clean = [f for f in fees if not _is_noise_fee_name(f.get("name"))]
    if not clean:
        clean = fees

    renewal_price = _fee_amount((facts.get("renewal_terms") or {}).get("renewal_price"))
    recurring = one_time = 0.0
    has_onboarding_fee = False
    seen_names = set()
    for fee in clean:
        amount = _fee_amount(fee.get("amount"))
        if not amount:
            continue
        if renewal_price and abs(amount - renewal_price) < 1:
            continue
        nkey = (fee.get("name") or "").strip().lower()
        if nkey in seen_names:
            continue
        seen_names.add(nkey)
        is_onboarding = _looks_like_onboarding(fee)
        if is_onboarding or (fee.get("billing_frequency") or "").lower() == "one_time":
            one_time += amount
            if is_onboarding:
                has_onboarding_fee = True
        else:
            recurring += amount

    # Agent 1 sometimes reports onboarding outside of fees[]; add it if not already counted.
    if not has_onboarding_fee:
        onboarding = _fee_amount((facts.get("onboarding_terms") or {}).get("fee"))
        if onboarding:
            one_time += onboarding

    return recurring, one_time


def _fixed_consideration(facts):
    """Total fixed (non-usage) consideration over the WHOLE contract term =
    recurring subscription x contract years + one-time fees."""
    term = facts.get("subscription_term_months") or 12
    num_years = max(1, round(term / 12))
    recurring, one_time = _fee_components(facts)
    return recurring * num_years + one_time


# Fee-table reader - one focused LLM pass that transcribes the fee table into structured
# components. Regex cannot recover bundled-total rows, "(see below)" lines, renewal prices,
# or bundle discounts; the LLM only reads and classifies, it does not compute a price.

def _fees_block(raw_text):
    """Slice the fee/payment section out of the contract text for the fee reader. Falls back
    to a leading chunk of the whole text when the heading cannot be located."""
    if not raw_text:
        return ""
    m = re.search(r"\n\s*5\.?\s*FEES", raw_text, re.I) or re.search(r"FEES\s*&?\s*PAYMENT", raw_text, re.I)
    if not m:
        return raw_text[:2500]
    start = m.start()
    nxt = re.search(r"\n\s*6\.\s+[A-Z]", raw_text[start + 10:])
    end = start + 10 + nxt.start() if nxt else start + 2500
    return raw_text[start:end]


def read_fee_table(raw_text, term_months=None):
    """Return the fee table as structured components (fee_lines/stated_total/
    combined_standalone_price/bundle_discount/excluded), or None if the call or parse fails
    (the caller then falls back to Agent 1's extracted fees)."""
    block = _fees_block(raw_text)
    if not block.strip():
        return None
    messages = [
        {"role": "system", "content": treatment.FEE_READER_SYSTEM},
        {"role": "user", "content": treatment.FEE_READER_USER.format(
            term_months=term_months if term_months is not None else "unknown",
            fees_block=block[:4000],
        )},
    ]
    try:
        parsed = json.loads(get_llm_response(messages, response_format="json_object"))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    parsed.setdefault("fee_lines", [])
    parsed.setdefault("stated_total", None)
    parsed.setdefault("combined_standalone_price", None)
    parsed.setdefault("bundle_discount", None)
    parsed.setdefault("excluded", [])
    return parsed


_RECURRENCE_TO_FREQ = {
    "recurring_annual": "annual",
    "recurring_quarterly": "quarterly",
    "one_time": "one_time",
}


def _fees_from_struct(fee_struct, term_months=None):
    """Convert the fee reader's fee_lines into Agent 1's fees[] shape so the deterministic
    fallback (transaction price, schedule, billing inference) can still consume them. Excluded
    lines (future renewals, optional add-ons) are intentionally dropped - they are not current
    consideration."""
    num_years = max(1, round((term_months or 12) / 12))
    out = []
    for line in fee_struct.get("fee_lines") or []:
        if not isinstance(line, dict):
            continue
        amount = _parse_number(line.get("amount"))
        if amount is None:
            continue
        rec = str(line.get("recurrence") or "").lower()
        # The fee reader emits a multi-year subscription as a single line carrying the
        # whole-term amount. Store it per year so _fee_components (which multiplies the
        # recurring amount by the number of years) reflects the year-one billing that the
        # opening deferred balance is based on.
        if rec == "recurring_annual" and num_years > 1:
            amount = amount / num_years
        out.append({
            "name": line.get("label") or "",
            "amount": amount,
            "billing_frequency": _RECURRENCE_TO_FREQ.get(rec, "annual"),
            "payment_timing": "",
        })
    return out


def apply_onboarding_rule(judgments, facts):
    """onboarding_distinct is decided by a RULE, not the LLM. Agent 1's top-level
    implementation_required_for_platform flag is a reliable predictor, and the LLM tends to
    read it correctly then discard it for the KB2_2 default. The LLM's original answer is kept
    under onboarding_distinct_llm for the audit trail.

    Known simplification: KB2_3's distinct exception is a conjunction (standardized / optional /
    third-party-available AND access-not-dependent-on-the-provider). This flag only covers the
    second condition; the two co-occur across this dataset, so the rule is dataset-specific, not
    a complete implementation of KB2_3.
    """
    llm = judgments.get("onboarding_distinct")
    if isinstance(llm, dict):
        judgments["onboarding_distinct_llm"] = llm

    node = _onboarding_rule_node(facts)
    if node is None:                            # flag unknown -> keep the LLM's own call
        return judgments
    judgments["onboarding_distinct"] = node
    return judgments


def _onboarding_rule_node(facts):
    """The deterministic onboarding_distinct judgment as a full node (conclusion + reasoning +
    kb_basis), or None when the flag is unknown so the LLM's own answer should stand. Each branch
    carries wording and a basis that match its own conclusion - no branch is empty or generic."""
    req = facts.get("implementation_required_for_platform")
    if not _has_onboarding_fee(facts):
        return {
            "conclusion": "n/a",
            "kb_basis": ["KB2_1"],
            "reasoning": ("Rule (not LLM): the contract carries no separately priced onboarding "
                          "or implementation fee, so there is no distinct onboarding obligation "
                          "to assess and the judgment does not apply."),
        }
    if req is True:
        return {
            "conclusion": "FALSE",
            "kb_basis": ["KB2_2"],
            "reasoning": ("Rule (not LLM): production access to the platform is conditioned on "
                          "completion of the implementation, so the service is combined into the "
                          "subscription performance obligation and is not distinct."),
        }
    if req is False:
        return {
            "conclusion": "TRUE",
            "kb_basis": ["KB2_3"],
            "reasoning": ("Rule (not LLM): platform access does not depend on RevLens performing "
                          "the implementation, so the distinct-service exception applies and the "
                          "onboarding is a separate performance obligation."),
        }
    return None


def derive_treatment(facts, judgments):
    """Turn the judgment conclusions into POs, transaction price, and method.

    Limitation: ASC 606 Step 4 (allocating the transaction price across the POs by relative
    standalone selling price) is not implemented - the price is carried at the contract level.
    """
    # ASC 606 Step 1: with no valid contract there is nothing to recognize yet. Only a
    # definite FALSE triggers this - UNDETERMINED/TRUE keep the normal treatment so an
    # unclear validity call never zeroes out an otherwise valid contract.
    if _conclusion(judgments, "contract_valid") == "FALSE":
        return {
            "num_POs": 1,
            "po_list": ["core_subscription"],
            "transaction_price": 0,
            "recognition_method": "UNDETERMINED",
            "recognition_period_months": facts.get("subscription_term_months") or 0,
        }

    onboarding_distinct = _conclusion(judgments, "onboarding_distinct")
    material_right = _conclusion(judgments, "material_right")
    usage_model = _conclusion(judgments, "usage_model")

    # Only treat onboarding as its own obligation when the contract ACTUALLY has an
    # onboarding fee. The model sometimes returns onboarding_distinct=TRUE on contracts
    # with no onboarding at all; without this guard that adds a phantom PO and flips the
    # recognition method.
    has_onboarding = _has_onboarding_fee(facts)
    onboarding_is_distinct = (onboarding_distinct == "TRUE") and has_onboarding

    # Performance obligations. Start from the subscription; add a PO for each
    # judgment that creates one.
    po_list = ["core_subscription"]
    if onboarding_is_distinct:
        po_list.append("onboarding(distinct)")
    if material_right == "TRUE":
        po_list.append("material_right(renewal)")
    # Extra separately-priced services the model judged distinct (KB2_1/KB1_3) each add
    # their own PO. Skip anything that echoes the subscription or onboarding so those
    # already-counted obligations are not double counted.
    extra = _conclusion(judgments, "additional_distinct_services") or []
    if isinstance(extra, str):
        extra = [extra]
    # A distinct service delivered as a discrete effort (training, consulting, advisory,
    # implementation, onboarding) is a point-in-time obligation. A service like premium
    # support is also its own PO but is delivered continuously, so it stays over time.
    point_in_time_service = ("training", "consulting", "advisory", "implementation", "onboarding")
    extra_onetime_added = False
    for service in extra:
        name = str(service).strip().lower()
        if name and not any(k in name for k in ("subscription", "onboard", "implementation", "setup", "core")):
            po_list.append(f"{name}(distinct)")
            if any(k in name for k in point_in_time_service):
                extra_onetime_added = True
    # Usage is variable consideration inside the subscription, not its own PO.

    # Transaction price = fixed consideration only.
    transaction_price = _fixed_consideration(facts)

    # Recognition method. If the contract is genuinely ambiguous upstream, don't
    # force a concrete method - carry the uncertainty forward as UNDETERMINED so a
    # human reviews it, rather than guessing.
    # The PO structure is undetermined only when a distinct-ness call itself is unresolved.
    # A missing term makes the recognition PERIOD unknown, not the PO count, so it is kept
    # out of this condition and handled through recognition_method below.
    po_undetermined = (
        (has_onboarding and onboarding_distinct == "UNDETERMINED")  # PO structure unclear
        or material_right == "UNDETERMINED"                 # renewal option unclear
    )
    ambiguous = po_undetermined or not facts.get("subscription_term_months")
    # A distinct one-time obligation (onboarding OR a discrete service such as
    # training/consulting) is recognized at a point in time while the subscription runs over time.
    # Continuous services (e.g. support) are separate POs but stay over time, so they do not count.
    has_distinct_onetime = onboarding_is_distinct or extra_onetime_added
    if ambiguous:
        recognition_method = "UNDETERMINED"
    elif usage_model == "pure_usage":
        recognition_method = "over_time_as_usage_occurs"
    elif usage_model == "undefined":
        # Usage is priced but its mechanics are unresolved: the base is recognized over time
        # while the usage component cannot yet be characterized.
        recognition_method = "base:over_time_ratable; usage:UNDETERMINED"
    elif has_distinct_onetime:
        recognition_method = "sub:over_time_ratable; onboarding:point_in_time"
        # A distinct onboarding PO that sits alongside a usage-priced subscription adds a
        # separately recognized usage component.
        if onboarding_is_distinct and usage_model not in ("none", "n/a"):
            recognition_method += "; usage:as_incurred"
    else:
        recognition_method = "over_time_ratable"

    return {
        # num_POs is 0 only when the PO structure itself is undetermined (a distinct-ness call is
        # unresolved). A merely unknown term does not zero the count. po_list stays for the audit trail.
        "num_POs": 0 if po_undetermined else len(po_list),
        "po_list": po_list,
        "transaction_price": transaction_price,
        "recognition_method": recognition_method,
        # 0 (not None) when the term is missing: the schedule length is undeterminable but the
        # evaluator scores None as wrong against a ground truth of 0.
        "recognition_period_months": facts.get("subscription_term_months") or 0,
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


def build_schedule(transaction_price, term_months, billing, start_date=None, facts=None):
    """Straight-line monthly recognition, opening deferred balance, and a
    month-by-month table anchored to the contract start date.

    monthly_revenue  = transaction_price / term_months (straight-line, a recognition fact).
    opening_deferred = the amount BILLED up front (a billing fact) - computed from the fee
                       schedule by billing pattern, NOT as monthly_revenue * months (those
                       only agree on a plain 12-month prepaid contract).
    """
    # Opening deferred by billing pattern (what is actually invoiced up front). This does not
    # depend on the term length, so it is computed even when the term is unknown.
    if billing == "monthly_arrears":
        opening_deferred = 0.0
    elif billing == "quarterly_upfront" and term_months:
        opening_deferred = transaction_price / term_months * 3
    elif billing == "annual_upfront_each_year" and facts is not None:
        # one year of recurring fees + all upfront one-time fees
        recurring, one_time = _fee_components(facts)
        opening_deferred = recurring + one_time
    else:
        # annual_upfront (term <= 12) and full_term_upfront: the whole price is prepaid
        opening_deferred = transaction_price

    # With no term the recognition length is undeterminable: revenue cannot be spread over
    # months, but the price and the amount billed up front (opening deferred) are still known.
    if not term_months:
        return {"monthly_revenue": 0.0, "opening_deferred": opening_deferred, "monthly_schedule": []}

    monthly_revenue = transaction_price / term_months

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


def _infer_billing(facts, raw_text=""):
    """Billing pattern → drives the opening deferred balance.

    Prefer the contract text (where the payment terms are actually stated); fall back to
    Agent 1 fee metadata. The billing pattern decides how much is billed up front:
    arrears/monthly-minimum = 0, quarterly = one quarter, each-year = one year,
    full-term = the whole contract.
    """
    t = (raw_text or "").lower()
    term = facts.get("subscription_term_months") or 0
    if t:
        # IMPORTANT ordering:
        # (1) prepaid ("in advance") patterns BEFORE any "in arrears" - many contracts prepay an
        #     annual base and bill only the *usage* in arrears; that must not flip the whole thing.
        # (2) full-term BEFORE per-year - a "$X/yr" contract that also says the whole total is
        #     "payable in advance, in full ... representing the full term" is a full-term prepay,
        #     not a per-year prepay, so full-term must be checked first.

        # whole term prepaid up front
        if re.search(r"payable in advance,\s*in full|for the (?:full|entire) term|representing the (?:full|entire)[^.]*term|total fixed fees[^.]*payable in advance", t):
            return "full_term_upfront"
        # one year prepaid, renewed each contract year - only meaningful for multi-year terms
        if term > 12 and re.search(r"\$[\d,]+\s*/\s*yr|billed annually in advance|each (?:contract )?year|on its anniversary|each subsequent (?:contract )?year|first annual", t):
            return "annual_upfront_each_year"
        # one quarter prepaid
        if re.search(r"quarterly in advance|billed quarterly", t):
            return "quarterly_upfront"
        # single year prepaid in advance
        if re.search(r"per year,?\s*payable in advance|annual[^.]*in advance", t):
            return "annual_upfront"
        # only now: the BASE itself is billed monthly / in arrears / as a monthly minimum
        if re.search(r"billed monthly in arrears|monthly installments|billed in (?:twelve|\d+)[^.]*monthly|monthly minimum|per month and is billed monthly", t):
            return "monthly_arrears"

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

def run(agent1_output, agent2_output, raw_text=None):
    """Run Agent 3 end to end for a single contract."""
    facts = (agent1_output.get("validated_output")
             or agent1_output.get("candidate_output")
             or agent1_output)

    # Contract text (supplementary evidence): explicit arg, else carried on the Agent 1 record.
    if raw_text is None:
        raw_text = agent1_output.get("full_text") or ""

    # Step 1: input from Agent 2 is usable.
    validate_agent2_output(agent2_output)

    # Step 2a: read the fee table with a focused LLM pass. Regex misses bundled-total rows,
    # renewal prices, and bundle discounts, which is where the money-field errors come from.
    # The structured result feeds both the judgment prompt (for the transaction price) and,
    # as a clean fee list, the deterministic schedule/fallback.
    fee_struct = read_fee_table(raw_text, facts.get("subscription_term_months"))
    if fee_struct:
        clean_fees = _fees_from_struct(fee_struct, facts.get("subscription_term_months"))
        if clean_fees:
            facts = {**facts, "fees": clean_fees}

    # Step 2b: LLM characterization judgments (conclusions normalized inside llm_judgments).
    # onboarding_distinct is then set by a deterministic rule - the LLM kept reading the
    # implementation_required_for_platform flag correctly then discarding it, so the rule
    # supersedes both the LLM and the old reflection loop for that field.
    judgments = llm_judgments(facts, agent2_output, raw_text, fee_struct)
    judgments = apply_onboarding_rule(judgments, facts)

    # Step 3: treatment. transaction_price now comes from the fee-informed LLM judgment - it
    # reasons about bundled totals and multi-year coverage that a flat deterministic sum cannot.
    # The deterministic sum is kept as a fallback (when the model returns nothing usable) and
    # under transaction_price_det for the audit trail. A FALSE validity call still forces 0.
    treatment_values = derive_treatment(facts, judgments)
    det_tp = treatment_values["transaction_price"]
    llm_tp = _parse_number(_conclusion(judgments, "transaction_price"))
    treatment_values["transaction_price_det"] = det_tp
    if _conclusion(judgments, "contract_valid") != "FALSE" and llm_tp is not None and llm_tp > 0:
        treatment_values["transaction_price"] = llm_tp

    # Step 4: schedule (pure arithmetic off the transaction price + billing pattern + fees).
    billing = _infer_billing(facts, raw_text)
    schedule = build_schedule(
        treatment_values["transaction_price"],
        treatment_values["recognition_period_months"],
        billing,
        facts.get("start_date"),
        facts,
    )

    return {
        "contract_id": facts.get("contract_id"),
        "source_file": facts.get("source_file"),
        "billing_used": billing,
        "characterization": judgments,
        "treatment": treatment_values,
        "schedule": schedule,
        "fee_components": fee_struct,
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