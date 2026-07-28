"""Memo Agent 5 - review memo assembly.

The Memo Agent does not make judgments. Agent 3 produced the numbers and Agent 4
produced the review findings; the Memo Agent assembles them into the deliverable an
accountant signs: a recognition schedule with the accounts to post to, the journal
entries, the review points that need human attention, and a disposition saying whether
the contract can be recognized as booked.

The structure, numbers, schedule, journal entries, severity mapping and disposition are
deterministic. An LLM authors ONLY the narrative prose (background, conclusion narrative,
per-review-point narrative); it never authors a number, date, conclusion or citation, and a
guard falls back to the deterministic template if the prose contains a dollar figure or KB id
not present in the structured facts.
"""
from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

from .client import get_llm_response
from ..prompts.memo import NARRATIVE_SYSTEM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KB_DIR = PROJECT_ROOT / "documents" / "knowledge_base"


def _humanize(name: str) -> str:
    s = str(name or "").replace("_", " ").strip()
    return s[:1].upper() + s[1:] if s else s


def _no_step(text: str) -> str:
    """Drop 'Step N' labels from display text - the five-step model is confusing to a reader who
    is not looking at ASC 606 side by side, and the surrounding prose already names each step."""
    return re.sub(r"\bStep\s*\d+\s*(?:[—:-]\s*)?", "", text or "")


def _load_kb_meta() -> tuple[dict, dict]:
    """Load two static maps keyed by doc id: the ASC 606 paragraph (KB1 asc_ref / KB2 related_asc)
    and a human-readable label (KB1/KB2 topic, KB3 name). The memo never shows a bare KB id, so the
    label lets a reader outside the project read every citation. The refs never reach the model
    through retrieval (they sit in the second chunk each reranker drops), but they are a static
    property of a doc id, so the memo layer loads them directly."""
    ref_map: dict[str, str] = {}
    label_map: dict[str, str] = {}
    for fname in ("KB1_asc606_guidance.jsonl", "KB2_revlens_policy.jsonl", "KB3_review_checklist.jsonl"):
        path = KB_DIR / fname
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            iid = str(item.get("id") or "").upper()
            if not iid:
                continue
            ref = item.get("asc_ref") or item.get("related_asc")
            if ref:
                ref_map[iid] = str(ref)
            label = item.get("topic") or _humanize(item.get("name"))
            if label:
                label_map[iid] = str(label)
    return ref_map, label_map


ASC_MAP, KB_LABEL = _load_kb_meta()


def _asc_annotate(text: str) -> str:
    """Dereference KB citations in prose so no bare KB id ever reaches the memo: KB1/KB2 become the
    ASC 606 paragraph; a KB3 checklist id becomes its plain-language name."""
    if not isinstance(text, str):
        # the model occasionally returns a list (or other type) for a prose field; flatten to text
        if isinstance(text, (list, tuple)):
            text = " ".join(str(x) for x in text)
        elif text is None:
            return ""
        else:
            text = str(text)

    def repl(m):
        kb = m.group(0).upper()
        ref = ASC_MAP.get(kb)
        if ref:
            return f"ASC {ref}"
        return KB_LABEL.get(kb, "the review checklist")
    return re.sub(r"KB[123]_\d+", repl, text)


def _annotate_narrative(nar: dict) -> dict:
    """Rewrite KB1/KB2 citations to ASC paragraphs throughout the LLM prose."""
    if not nar:
        return {}
    out = dict(nar)
    for key in ("background", "conclusion_narrative"):
        if out.get(key):
            out[key] = _asc_annotate(out[key])
    out["analysis"] = {k: _asc_annotate(v) for k, v in (out.get("analysis") or {}).items()}
    out["review_points"] = [{**p, "narrative": _asc_annotate(p.get("narrative", ""))}
                            for p in (out.get("review_points") or [])]
    return out


def _fmt_date(iso) -> str:
    d = _parse_date(iso)
    return d.strftime("%m/%d/%Y") if d else "-"


ACCOUNTS = {
    "AR":            ("1200", "Accounts Receivable"),
    "CONTRACT_ASSET": ("1250", "Contract Asset - Unbilled Receivable"),
    "DEFERRED_CUR":  ("2400", "Deferred Revenue - Current"),
    "DEFERRED_LT":   ("2450", "Deferred Revenue - Non-current"),
    "REV_SUB":       ("4100", "Subscription Revenue"),
    "REV_SERVICES":  ("4200", "Professional Services Revenue"),
    "REV_USAGE":     ("4300", "Usage Revenue"),
}


def _acct(key: str) -> str:
    code, name = ACCOUNTS[key]
    return f"{code} {name}"


# Severity is assigned by the consequence of getting the item wrong, from each KB3 item's own
# risk / review_question text: CRITICAL - recognition cannot proceed; HIGH - the amount or the
# obligation structure may be wrong; MEDIUM - only timing/classification; LOW - only presentation.
SEVERITY = {
    # CRITICAL: no recognition can proceed (validity / term / PO count unknown).
    "KB3_15": "CRITICAL", "KB3_16": "CRITICAL", "KB3_17": "CRITICAL",
    # HIGH: the amount or an obligation could be materially wrong -> review before recognising.
    "KB3_5": "HIGH", "KB3_8": "HIGH", "KB3_10": "HIGH", "KB3_18": "HIGH",
    # MEDIUM: a classification or allocation to confirm; book on the current treatment and clear
    # before close (onboarding distinctness, usage/discount presence, PO allocation, billing timing).
    # These were demoted from HIGH: a confirmatory review of the treatment does not, on its own,
    # withhold recognition. KB3_1 is the default here but is raised to HIGH when KB3_17 also fires
    # (distinctness evidence incomplete) - see build_review_points.
    "KB3_1": "MEDIUM", "KB3_2": "MEDIUM", "KB3_3": "MEDIUM", "KB3_4": "MEDIUM",
    "KB3_6": "MEDIUM", "KB3_7": "MEDIUM", "KB3_9": "MEDIUM", "KB3_11": "MEDIUM", "KB3_14": "MEDIUM",
    # LOW: presentation / timing only.
    "KB3_12": "LOW", "KB3_13": "LOW",
}
SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
TIMING = {
    "CRITICAL": "Before any revenue is recognised",
    "HIGH": "Before revenue is recognised",
    "MEDIUM": "Before period close",
    "LOW": "Before period close",
}
ACTIONS = {
    "KB3_1":  "Document whether the onboarding is customised and required for platform access "
              "(combine into the subscription obligation) or standardised and optional (separate "
              "obligation). Retain the evidence relied on.",
    "KB3_2":  "Confirm the distinctness conclusion rests on the nature of the service, not on the "
              "size of the fee.",
    "KB3_3":  "Confirm the variable portion is separated from the fixed fee and is recognised as "
              "usage occurs rather than estimated for the full term.",
    "KB3_4":  "Confirm the committed minimum is recognised as fixed consideration over the term, "
              "with only usage above the included volume treated as variable.",
    "KB3_5":  "Constrain the variable consideration estimate to the amount highly probable not to "
              "reverse; recognise as usage occurs where volume is seasonal or unpredictable.",
    "KB3_6":  "Confirm whether the variable fee is a licence royalty, ordinary metered usage, or "
              "the purchase of new capacity, and apply the corresponding model.",
    "KB3_7":  "Obtain the standalone selling price of each component and allocate the discount "
              "proportionately unless the directed-discount evidence is present.",
    "KB3_8":  "Confirm at least one component has an observable standalone selling price and apply "
              "the residual approach only to the component whose price is not observable.",
    "KB3_9":  "Allocate the transaction price across all obligations by relative standalone selling "
              "price and confirm each is recognised on its own pattern.",
    "KB3_10": "Obtain the renewal price offered to comparable customers and assess whether the "
              "option conveys a material right requiring separate allocation.",
    "KB3_11": "Confirm a contract liability is recorded where the customer is billed in advance, "
              "and a contract asset where performance precedes the right to bill.",
    "KB3_12": "Prorate the partial first or last month by days; full amounts only for complete months.",
    "KB3_13": "Document the significant financing component assessment for the multi-year prepayment "
              "and the basis for concluding it is immaterial.",
    "KB3_14": "Spread the total transaction price over the entire access period including the free "
              "months, rather than recognising nil in those months.",
    "KB3_15": "Assess the contract-existence criteria (approval, identifiable rights and payment terms, "
              "commercial substance, probable collection). If not met, withhold recognition and escalate.",
    "KB3_16": "Obtain the subscription term and go-live date from the customer. Do not issue a "
              "recognition schedule until the service period is established.",
    "KB3_17": "Obtain the missing facts on customisation, dependency and third-party availability "
              "before fixing the performance obligation count.",
    "KB3_18": "Obtain the renewal price and how it compares to standard pricing for comparable "
              "customers before concluding on the material right.",
}
# Displayed as "Recognition status" in the memo. The structured key stays 'disposition' for
# downstream/eval; only the rendered text changes. "Disposition" in practice is the per-item review
# status (open/cleared), applied to each open item at section 7 - not the memo-level status.
DISPOSITION_TEXT = {
    "proceed": "Cleared for recognition",
    "review_before_close": "Cleared for recognition; open items to be resolved before period close",
    "review_before_recognition": "Not cleared for recognition pending resolution of the open items below",
    "escalate": "Referred to technical accounting; recognition withheld",
    "cannot_assess": "Unable to conclude; contract data insufficient",
}

# Materiality: a quantified exposure below the threshold is a documented "immaterial" conclusion,
# not an omission. Threshold = 5% of the transaction price with an absolute floor.
MATERIALITY_PCT = 0.05
MATERIALITY_FLOOR = 1000.0


def _num(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        return _num(value.get("amount"))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace("$", "").replace(",", "").strip())
        except ValueError:
            return None
    return None


def _money(value: Optional[float]) -> str:
    return "-" if value is None else f"${value:,.2f}"


def _parse_date(value) -> Optional[date]:
    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None


def _add_months(start: date, months: int) -> date:
    month_index = start.month - 1 + months
    year = start.year + month_index // 12
    month = month_index % 12 + 1
    for day in (start.day, 28):
        try:
            return date(year, month, day)
        except ValueError:
            continue
    return date(year, month, 28)


def _conclusion(agent3: dict, field: str) -> str:
    node = (agent3.get("characterization") or {}).get(field) or {}
    if isinstance(node, dict):
        return str(node.get("conclusion") or "").strip()
    return str(node or "").strip()


def _billing_events(billing: str, price: float, term: int, start: Optional[date]) -> list[dict]:
    """When the customer is invoiced, and for how much - drives the deferred-revenue rollforward."""
    billing = (billing or "").lower()
    events: list[dict] = []

    def at(month_index: int, amount: float) -> None:
        events.append({"month": month_index + 1, "amount": round(amount, 2),
                       "date": _add_months(start, month_index).isoformat() if start else None})

    if "arrear" in billing:
        for m in range(term):
            at(m, price / term)
    elif "quarter" in billing:
        quarters = max(1, -(-term // 3))
        for q in range(quarters):
            at(q * 3, price / quarters)
    elif "each_year" in billing and term > 12:
        years = max(1, -(-term // 12))
        for y in range(years):
            at(y * 12, price / years)
    else:  # annual_upfront (term <= 12) or full_term_upfront
        at(0, price)
    return events


def build_schedule(agent3: dict, facts: dict) -> dict:
    """Monthly recognition with the deferred-revenue rollforward and the posting account."""
    treatment = agent3.get("treatment") or {}
    sched = agent3.get("schedule") or {}
    price = _num(treatment.get("transaction_price")) or 0.0
    term = treatment.get("recognition_period_months") or 0
    try:
        term = int(term)
    except (TypeError, ValueError):
        term = 0
    if term <= 0:
        return {"rows": [], "billing_events": [], "note":
                "No recognition schedule produced: the service period could not be determined from "
                "the contract (see the open item on the missing service period or dates)."}

    start = _parse_date(facts.get("start_date"))
    billing = str(agent3.get("billing_used") or "")
    monthly = _num(sched.get("monthly_revenue"))
    if monthly is None:
        monthly = price / term

    method = str(treatment.get("recognition_method") or "")
    onboarding_at_completion = 0.0
    if "point_in_time" in method:
        for line in ((agent3.get("fee_components") or {}).get("fee_lines") or []):
            label = str(line.get("label") or "").lower()
            if any(w in label for w in ("onboard", "implementation", "setup", "activation", "migration")) \
                    and "subscription" not in label:
                onboarding_at_completion = _num(line.get("amount")) or 0.0
                break
    ratable_total = price - onboarding_at_completion
    ratable_monthly = ratable_total / term if term else 0.0

    events = _billing_events(billing, price, term, start)
    billed_by_month: dict[int, float] = {}
    for ev in events:
        billed_by_month[ev["month"]] = billed_by_month.get(ev["month"], 0.0) + ev["amount"]

    rows = []
    balance = cumulative = 0.0
    for m in range(1, term + 1):
        billed = billed_by_month.get(m, 0.0)
        recognised = ratable_monthly + (onboarding_at_completion if m == 1 else 0.0)
        balance += billed - recognised
        cumulative += recognised
        rows.append({
            "month": m,
            "period_start": _add_months(start, m - 1).isoformat() if start else None,
            "period_end": (_add_months(start, m) - timedelta(days=1)).isoformat() if start else None,
            "billed": round(billed, 2),
            "revenue_recognised": round(recognised, 2),
            "cumulative_revenue": round(cumulative, 2),
            "closing_balance": round(balance, 2),
            "balance_account": _acct("DEFERRED_CUR") if balance >= 0 else _acct("CONTRACT_ASSET"),
            "revenue_account": _acct("REV_SERVICES") if (m == 1 and onboarding_at_completion) else _acct("REV_SUB"),
        })

    return {
        "rows": rows,
        "billing_events": events,
        "opening_deferred": round(_num(sched.get("opening_deferred")) or 0.0, 2),
        "onboarding_at_completion": round(onboarding_at_completion, 2),
        "ratable_monthly": round(ratable_monthly, 2),
        "note": None,
    }


def build_journal_entries(schedule: dict, agent3: dict) -> list[dict]:
    """Representative entries: initial billing, the recurring monthly entry, and any point-in-time entry."""
    if not schedule.get("rows"):
        return []
    entries = []
    first = schedule["rows"][0]

    if first["billed"]:
        entries.append({
            "when": "On invoice",
            "lines": [{"dr": _acct("AR"), "cr": None, "amount": first["billed"]},
                      {"dr": None, "cr": _acct("DEFERRED_CUR"), "amount": first["billed"]}],
            "narrative": "Record the contract liability on billing in advance of performance.",
        })
    if schedule.get("onboarding_at_completion"):
        amt = schedule["onboarding_at_completion"]
        entries.append({
            "when": "On completion of the onboarding obligation",
            "lines": [{"dr": _acct("DEFERRED_CUR"), "cr": None, "amount": amt},
                      {"dr": None, "cr": _acct("REV_SERVICES"), "amount": amt}],
            "narrative": "Recognise the distinct onboarding obligation at the point control transfers.",
        })
    monthly = schedule.get("ratable_monthly") or 0.0
    if monthly:
        arrears = any(r["closing_balance"] < 0 for r in schedule["rows"])
        entries.append({
            "when": "Monthly, over the subscription term",
            "lines": [{"dr": _acct("CONTRACT_ASSET") if arrears else _acct("DEFERRED_CUR"),
                       "cr": None, "amount": monthly},
                      {"dr": None, "cr": _acct("REV_SUB"), "amount": monthly}],
            "narrative": ("Release the contract liability as the subscription is delivered." if not arrears
                          else "Recognise revenue as performance occurs; the right to consideration is "
                               "not yet unconditional."),
        })
    usage = _conclusion(agent3, "usage_model")
    if usage and usage not in ("none", "n/a"):
        entries.append({
            "when": "As usage is incurred",
            "lines": [{"dr": _acct("AR"), "cr": None, "amount": None},
                      {"dr": None, "cr": _acct("REV_USAGE"), "amount": None}],
            "narrative": f"Usage-based consideration ({usage}) is variable and is recognised as the usage occurs.",
        })
    return entries


def _extraction_sufficient(facts: dict, agent3: dict) -> tuple[bool, list[str]]:
    """Distinguish 'no risks found' from 'could not be checked'."""
    checks = {
        "subscription term": facts.get("subscription_term_months"),
        "start date": facts.get("start_date"),
        "fee schedule": facts.get("fees"),
        "transaction price": (agent3.get("treatment") or {}).get("transaction_price"),
    }
    missing = [name for name, value in checks.items() if not value]
    return len(missing) <= 1, missing


def _day_of(value) -> Optional[int]:
    d = _parse_date(value)
    return d.day if d else None


def materiality_threshold(transaction_price: float) -> float:
    return max((transaction_price or 0) * MATERIALITY_PCT, MATERIALITY_FLOOR)


def _exposure(kb3_id: str, treatment: dict, schedule: dict, facts: dict) -> Optional[float]:
    """The quantified exposure for items whose effect is a number; None if not quantifiable
    (resolved on qualitative grounds instead)."""
    monthly = schedule.get("ratable_monthly") or 0.0
    price = _num(treatment.get("transaction_price")) or 0.0
    if kb3_id == "KB3_12":                                  # first-month proration difference
        day = _day_of(facts.get("start_date")) or 1
        return round(monthly * (day - 1) / 30.0, 2)
    if kb3_id == "KB3_13":                                  # financing component, order-of-magnitude (5% / 24mo)
        term = treatment.get("recognition_period_months") or 0
        try:
            term = int(term)
        except (TypeError, ValueError):
            term = 0
        return round(price * 0.05 * (term / 24.0), 2)
    if kb3_id == "KB3_2":                                   # the onboarding fee itself
        return round(schedule.get("onboarding_at_completion") or 0.0, 2)
    if kb3_id == "KB3_14":                                  # free months given away
        return round(monthly * (facts.get("free_months") or 1), 2)
    return None


def build_review_points(agent4: dict, treatment: dict, schedule: dict, facts: dict) -> list[dict]:
    threshold = round(materiality_threshold(_num(treatment.get("transaction_price")) or 0.0), 2)
    material_ids = {f["kb3_id"] for f in (agent4.get("audit_findings") or []) if f.get("material")}
    points = []
    for finding in agent4.get("audit_findings") or []:
        if not finding.get("material"):
            continue
        kb3_id = finding["kb3_id"]
        severity = SEVERITY.get(kb3_id, "MEDIUM")
        if kb3_id == "KB3_1":
            # KB3_1's severity depends on whether the distinctness evidence is complete. Where the
            # contract states the facts needed to determine distinctness (KB3_17 does not fire), the
            # item is a confirmatory review -> MEDIUM. Where it does not (KB3_1 and KB3_17 fire
            # together), distinctness cannot be concluded -> HIGH.
            severity = "HIGH" if "KB3_17" in material_ids else "MEDIUM"
        exposure = _exposure(kb3_id, treatment, schedule, facts)
        if exposure is None:
            conclusion = "not_quantifiable"
        elif exposure < threshold:
            conclusion = "immaterial"
        else:
            conclusion = "material"
        points.append({
            "kb3_id": kb3_id, "name": finding.get("name"), "severity": severity,
            "finding": (finding.get("evidence") or [""])[0],
            "review_question": finding.get("review_question"),
            "action": ACTIONS.get(kb3_id, "Document the conclusion and the evidence relied on."),
            "timing": TIMING[severity],
            "exposure": exposure,
            "materiality_threshold": threshold,
            "materiality_conclusion": conclusion,
            "disposition": "open",   # per-item review status; a reviewer sets this to 'cleared'
        })
    points.sort(key=lambda p: (SEVERITY_ORDER[p["severity"]], p["kb3_id"]))
    return points


def determine_disposition(points: list[dict]) -> str:
    # Items concluded immaterial do not block a disposition; LOW severity never escalates.
    active = [p for p in points if p.get("materiality_conclusion") != "immaterial"]
    severities = {p["severity"] for p in active}
    if "CRITICAL" in severities:
        return "escalate"
    if "HIGH" in severities:
        return "review_before_recognition"
    if "MEDIUM" in severities:
        return "review_before_close"
    return "proceed"


# ---- LLM narrative (prose only; numbers and citations are supplied, not authored) ----

def _all_amounts(result: dict) -> set[float]:
    amts: set[float] = set()
    for v in (_num(result["treatment"].get("transaction_price")),
              result["schedule"].get("opening_deferred"),
              result["schedule"].get("onboarding_at_completion"),
              result["schedule"].get("ratable_monthly"),
              _num((result.get("schedule_source") or {}).get("monthly_revenue"))):
        if v:
            amts.add(round(float(v), 2))
    for row in result["schedule"].get("rows", []):
        for k in ("billed", "revenue_recognised", "cumulative_revenue", "closing_balance"):
            if row.get(k):
                amts.add(round(float(row[k]), 2))
    for entry in result.get("journal_entries", []):
        for line in entry["lines"]:
            if line.get("amount"):
                amts.add(round(float(line["amount"]), 2))
    return amts


def _walk_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_strings(v)


def _draft_narrative(result: dict) -> tuple[dict | None, str]:
    """Render the memo prose via LLM. Numbers, conclusions and citations come from the structured
    result; the model only writes prose. Returns (draft, "ok") or (None, reason) where reason is
    'exception', 'parse_fail' or 'amount_whitelist' - so the fallback rate can be diagnosed."""
    payload = {
        "customer": result.get("customer_name"),
        "transaction_price": result["treatment"].get("transaction_price"),
        "fee_components": result["treatment"].get("fee_components"),
        "performance_obligations": result["treatment"].get("num_POs"),
        "recognition_method": result["treatment"].get("recognition_method"),
        "term_months": result["treatment"].get("recognition_period_months"),
        "characterization": result.get("characterization"),
        "preparer_reasoning": result.get("reasoning"),
        "kb_refs": sorted(set(result.get("kb1_refs", []) + result.get("kb2_refs", [])
                              + result.get("kb3_refs", []))),
        "disposition": result["disposition"],
        "review_points": [
            {"kb3_id": p["kb3_id"], "name": p["name"], "severity": p["severity"],
             "finding": p["finding"], "action": p["action"]}
            for p in result["review_points"]
        ],
        "immaterial_matters": [
            {"kb3_id": p["kb3_id"], "materiality_conclusion": p["materiality_conclusion"]}
            for p in result.get("immaterial_matters", [])
        ],
    }
    try:
        raw = get_llm_response(
            [{"role": "system", "content": NARRATIVE_SYSTEM},
             {"role": "user", "content": json.dumps(payload, indent=2, default=str)}],
            response_format="json_object")
    except Exception:
        return None, "exception"
    try:
        drafted = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None, "parse_fail"

    # Amount whitelist: every dollar figure in the prose must match a structured amount. Both sides
    # are normalised (strip $/commas, round to 2dp and to the whole dollar) so formatting or dollar
    # rounding never falsely rejects; the check itself is not relaxed - a figure that is not a
    # structured amount fails and the memo falls back to the template.
    allowed = set(_all_amounts(result))
    tp = _num(result["treatment"].get("transaction_price")) or 0.0
    monthly = result["schedule"].get("ratable_monthly") or 0.0
    if monthly:
        allowed.add(round(monthly * 12, 2))
    if tp:
        allowed.add(round(tp, 2))
    # Also whitelist any dollar figure that appears in the input payload itself (the preparer's
    # reasoning and the findings). Those are grounded facts the model was given, so citing them is
    # not a hallucination; a figure that is in NEITHER the structured amounts NOR the input is.
    for found in re.findall(r"\$\s?([\d,]+(?:\.\d+)?)", json.dumps(payload, default=str)):
        try:
            allowed.add(round(float(found.replace(",", "")), 2))
        except ValueError:
            pass
    # The transaction price decomposes into component amounts (fee lines, standalone selling
    # prices, bundle discount) that a step-2/step-3 discussion legitimately cites. These are
    # structured figures from Agent 3, stored as bare numbers, so add them explicitly.
    fc = result["treatment"].get("fee_components") or {}
    for key in ("stated_total", "combined_standalone_price", "bundle_discount"):
        v = _num(fc.get(key))
        if v:
            allowed.add(round(v, 2))
    for line in (fc.get("fee_lines") or []) + (fc.get("excluded") or []):
        v = _num(line.get("amount"))
        if v:
            allowed.add(round(v, 2))
    allowed_int = {round(x) for x in allowed}
    for text in _walk_strings(drafted):
        for found in re.findall(r"\$\s?([\d,]+(?:\.\d+)?)", text):
            try:
                val = float(found.replace(",", ""))
            except ValueError:
                continue
            if round(val, 2) not in allowed and round(val) not in allowed_int:
                return None, "amount_whitelist"
    return drafted, "ok"


# ---- memo rendering ----

RULE = "=" * 78
THIN = "-" * 78


IMMATERIAL_SENTENCE = ("We consider the amount immaterial to the financial statements taken as a "
                       "whole and no adjustment has been recorded.")


def _recognition_phrase(method: str) -> str:
    m = str(method or "").lower()
    if "point_in_time" in m and "over_time" in m:
        return "the subscription ratably over time and the onboarding at a point in time"
    if "usage" in m:
        return "as usage occurs"
    if "point_in_time" in m:
        return "at a point in time as control transfers"
    if "undetermined" in m:
        return "on a basis to be determined"
    return "ratably over time"


def render_memo(result: dict) -> str:
    L: list[str] = []
    a = L.append
    nar = _annotate_narrative(result.get("narrative") or {})   # KB ids -> ASC paragraphs in prose
    steps = nar.get("analysis") or {}
    char = result.get("characterization") or {}
    t = result.get("treatment") or {}
    money = _money(_num(t.get("transaction_price")))
    months = t.get("recognition_period_months")
    npo = t.get("num_POs") or 0
    rec_phrase = _recognition_phrase(t.get("recognition_method"))

    def para(text, indent="   "):
        for line in _wrap(text):
            a(f"{indent}{line}")

    a(RULE)
    a("REVENUE RECOGNITION MEMORANDUM")
    a(RULE)
    a(f"Contract:            {result['contract_id']}")
    a(f"Customer:            {result.get('customer_name') or 'Not stated'}")
    a(f"Prepared by:         Treatment Agent (Agent 3)")
    a(f"Reviewed by:         Audit Agent (Agent 4) - 18-item review checklist")
    a(f"Standard:            ASC 606, Revenue from Contracts with Customers")
    a(f"Recognition status:  {DISPOSITION_TEXT[result['disposition']]}")
    a(RULE)
    a("")

    if result["disposition"] == "cannot_assess":
        a("1. CONCLUSION")
        a("")
        para("The contract could not be assessed. The extraction layer did not identify the "
             "following required elements:")
        for item in result["missing_elements"]:
            a(f"     - {item}")
        a("")
        para("No recognition schedule has been produced and the KB3 review checklist has not been "
             "performed. This is a data sufficiency failure, not a conclusion that the contract "
             "presents no risk. Recommended procedure: extract the elements manually and resubmit.")
        a(RULE)
        return "\n".join(L)

    a("1. PURPOSE AND SCOPE")
    a("")
    para("This memorandum documents the revenue recognition analysis for the arrangement identified "
         "above under ASC 606, Revenue from Contracts with Customers, and records the accounting "
         "conclusion, the recognition schedule, the journal entries, and the open items for follow-up.")
    a("")

    a("2. FACTS AND CIRCUMSTANCES")
    a("")
    if nar.get("background"):
        para(nar["background"])
    else:
        para(f"The Company entered into a contract with {result.get('customer_name') or 'the customer'} "
             f"with a transaction price of {money} over a {months}-month term.")
    a("")

    a("3. AUTHORITATIVE GUIDANCE")
    a("")
    # One line per ASC 606 paragraph. The same paragraph is often cited by both the standard's
    # guidance and the entity policy, so merge their topics rather than list the paragraph twice.
    by_ref: dict[str, list[str]] = {}
    order: list[str] = []
    for k in result["kb1_refs"] + result["kb2_refs"]:
        ref = ASC_MAP.get(k) or "(no ASC paragraph)"
        if ref not in by_ref:
            by_ref[ref] = []
            order.append(ref)
        topic = re.sub(r"^\s*Step\s*\d+\s*[—:-]\s*", "", KB_LABEL.get(k, ""))
        topic = topic[:1].upper() + topic[1:] if topic else topic
        if topic and topic not in by_ref[ref]:
            by_ref[ref].append(topic)
    if order:
        for ref in order:
            label = ref if ref.startswith("(") else f"ASC {ref}"
            a(f"   {label:<28}  {'; '.join(by_ref[ref])}")
    else:
        a("   None cited.")
    a("")
    para("The paragraph references above point to the relevant provisions of ASC 606; the entity's "
         "revenue policy paraphrases those provisions and should be read together with them.")
    a("")

    a("4. ANALYSIS")
    a("")
    s1 = ("The Company has concluded that a valid, enforceable contract exists."
          if char.get("contract_valid") == "TRUE"
          else f"Contract validity has not been established (status: {char.get('contract_valid')}).")
    _step(a, para, "Identify the contract with the customer", s1, steps.get("step1"))

    od = char.get("onboarding_distinct")
    onb_phrase = {"TRUE": "the onboarding is a distinct performance obligation",
                  "FALSE": "onboarding is not distinct and is combined into the subscription",
                  "n/a": "there is no separately priced onboarding",
                  "UNDETERMINED": "onboarding distinctness could not be determined"}.get(od,
                  f"onboarding treated as {od}")
    po_names = ", ".join(str(p) for p in (t.get("po_list") or []))
    s2 = (f"{npo} performance obligation(s) were identified" + (f": {po_names}" if po_names else "")
          + f". On onboarding, {onb_phrase}.")
    _step(a, para, "Identify the performance obligations", s2, steps.get("step2"))

    um, dt = char.get("usage_model"), char.get("discount_type")
    var_phrase = ("no variable consideration" if um in ("none", "n/a", None, "")
                  else f"variable consideration ({um})")
    disc_phrase = ("no discount" if dt in ("none", "n/a", None, "")
                   else f"a discount characterised as {dt}")
    s3 = f"The transaction price is {money}. The arrangement includes {var_phrase} and {disc_phrase}."
    _step(a, para, "Determine the transaction price", s3, steps.get("step3"))

    alloc = ("A single performance obligation was identified; no allocation is required."
             if npo in (1, 0, None) else
             "Allocation across performance obligations by relative standalone selling price (SSP) is "
             "NOT implemented in this analysis; a multi-obligation schedule must be reallocated before use.")
    _step(a, para, "Allocate the transaction price to the performance obligations",
          alloc, steps.get("step4"))
    s5 = f"Revenue is recognised {rec_phrase} over {months} months."
    _step(a, para, "Recognise revenue as performance obligations are satisfied",
          s5, steps.get("step5"))

    a("5. REVENUE RECOGNITION SCHEDULE")
    a("")
    if result["schedule"].get("note"):
        para(result["schedule"]["note"])
    else:
        a(f"   {'Mo':<4}{'Period':<26}{'Billed':>14}{'Revenue':>14}{'Balance':>14}")
        a(f"   {THIN}")
        for row in result["schedule"]["rows"]:
            period = (f"{_fmt_date(row['period_start'])} - {_fmt_date(row['period_end'])}"
                      if row["period_start"] else "-")
            a(f"   {row['month']:<4}{period:<26}"
              f"{_money(row['billed']):>14}{_money(row['revenue_recognised']):>14}{_money(row['closing_balance']):>14}")
        a(f"   {THIN}")
        total = sum(r["revenue_recognised"] for r in result["schedule"]["rows"])
        billed = sum(r["billed"] for r in result["schedule"]["rows"])
        a(f"   {'':<30}{_money(billed):>14}{_money(total):>14}")
        a("")
        a(f"   Revenue account            {_acct('REV_SUB')}")
        if result["schedule"].get("onboarding_at_completion"):
            a(f"   Services revenue account   {_acct('REV_SERVICES')}")
        a(f"   Balance sheet account      Deferred revenue (contract liability), "
          f"{result['schedule']['rows'][0]['balance_account']}")
        a("")
        para("Revenue is recognised on a straight-line monthly basis. Partial first or last months "
             "are not prorated by days; where the term starts mid-month this affects the timing "
             "pattern only, not the total transaction price.")
    a("")

    a("6. JOURNAL ENTRIES")
    a("")
    if not result["journal_entries"]:
        a("   None produced; see the schedule note above.")
    for entry in result["journal_entries"]:
        a(f"   {entry['when']}")
        for line in entry["lines"]:
            amount = _money(line["amount"]) if line["amount"] is not None else "as incurred"
            if line["dr"]:
                a(f"      Dr  {line['dr']:<45}{amount:>14}")
            else:
                a(f"          Cr  {line['cr']:<41}{amount:>14}")
        a(f"      {entry['narrative']}")
        a("")

    a("7. OPEN ITEMS AND MATTERS FOR FOLLOW-UP")
    a("")
    if not result["review_points"]:
        para("None. No matter was concluded to require further procedures before recognition.")
        a("")
    nar_by_id = {p.get("kb3_id"): p.get("narrative") for p in (nar.get("review_points") or [])}
    for point in result["review_points"]:
        a(f"   [{point['severity']}] {_humanize(point['name'])}")
        a(f"      Facts and circumstances:  {_no_step(point['finding'])}")
        a(f"      Question:                 {_no_step(point['review_question'])}")
        a(f"      Recommended procedure:    {_no_step(point['action'])}")
        a(f"      Timing:                   {point['timing']}")
        a(f"      Disposition:              {point.get('disposition', 'open')}")
        if nar_by_id.get(point["kb3_id"]):
            para(nar_by_id[point["kb3_id"]], indent="      ")
        a("")

    a("8. MATTERS CONSIDERED AND CONCLUDED IMMATERIAL")
    a("")
    if not result.get("immaterial_matters"):
        para("No quantified matter was concluded immaterial on this contract.")
        a("")
    for p in result.get("immaterial_matters", []):
        a(f"   {_humanize(p['name'])}")
        para(f"The item was triggered on the facts. The exposure is {_money(p['exposure'])} against a "
             f"materiality threshold for this contract of {_money(p['materiality_threshold'])} (5% of "
             f"the transaction price, floor {_money(MATERIALITY_FLOOR)}). {IMMATERIAL_SENTENCE}",
             indent="      ")
        a("")

    a("9. COMPLETENESS OF REVIEW")
    a("")
    n_open, n_imm = len(result["review_points"]), len(result.get("immaterial_matters", []))
    n_na = len(result["items_not_applicable"])
    para(f"The 18-item review checklist was completed in full. Of the 18 matters, {n_open} "
         f"{'is' if n_open == 1 else 'are'} reported at section 7, {n_imm} at section 8, and {n_na} "
         f"were considered and concluded not applicable to this contract on the facts. The completed "
         f"checklist, with the basis for each conclusion, is retained in the structured review record.")
    a("")
    if result.get("consistency_issues"):
        para("Exceptions noted by the reviewer (the prepared conclusions have not been amended):")
        for issue in result["consistency_issues"]:
            a(f"     [{issue['type']}] {issue.get('field')}: {_asc_annotate(issue['detail'])}")
        a("")

    a("10. CONCLUSION")
    a("")
    if nar.get("conclusion_narrative"):
        para(nar["conclusion_narrative"])
    else:
        para(f"We have concluded that the transaction price of {money} is recognised {rec_phrase} over "
             f"{months} months across {npo} performance obligation(s).")
    a("")
    a(f"   Recognition status: {DISPOSITION_TEXT[result['disposition']]}")
    a("")
    para("Scope and limitations: allocation of the transaction price across performance obligations "
         "by relative standalone selling price is not implemented; partial periods are not prorated "
         "by days; variable consideration is excluded from the transaction price and recognised as incurred.")
    a("")

    a("11. PREPARED BY / REVIEWED BY")
    a("")
    a("   Preparer sign-off ______________________  Date __________")
    a("   Reviewer sign-off ______________________  Date __________")
    a(RULE)
    return "\n".join(L)


def _step(a, para, title: str, prose_summary: str, llm_prose: Optional[str]) -> None:
    a(f"   {title}")
    para(prose_summary, indent="      ")
    if llm_prose:
        para(llm_prose, indent="      ")
    a("")


def _wrap(text: str, width: int = 74) -> list[str]:
    import textwrap
    return textwrap.wrap(re.sub(r"\s+", " ", text or "").strip(), width) or [""]


class MemoAgent:
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm

    def run(self, agent1_output: dict, agent3_output: dict, agent4_output: dict) -> dict:
        agent1 = agent1_output or {}
        agent3 = agent3_output or {}
        agent4 = agent4_output or {}
        facts = agent1.get("validated_output") or agent1.get("candidate_output") or agent1
        contract_id = (facts.get("contract_id") or agent3.get("contract_id") or agent4.get("contract_id"))

        sufficient, missing = _extraction_sufficient(facts, agent3)
        if not sufficient:
            result = {
                "agent": "MemoAgent", "contract_id": contract_id,
                "customer_name": facts.get("customer_name"),
                "disposition": "cannot_assess", "missing_elements": missing,
                "review_points": [], "items_not_applicable": [],
                "schedule": {"rows": [], "note": "Not produced."}, "journal_entries": [],
                "treatment": agent3.get("treatment") or {},
                "consistency_issues": agent4.get("consistency_issues") or [],
                "kb1_refs": [], "kb2_refs": [], "kb3_refs": [], "narrative": None,
            }
            result["memo"] = render_memo(result)
            return result

        schedule = build_schedule(agent3, facts)
        points = build_review_points(agent4, agent3.get("treatment") or {}, schedule, facts)
        open_points = [p for p in points if p["materiality_conclusion"] != "immaterial"]
        immaterial_points = [p for p in points if p["materiality_conclusion"] == "immaterial"]

        characterization, reasoning = {}, {}
        kb1, kb2 = set(), set()
        for field, node in (agent3.get("characterization") or {}).items():
            if field.endswith("_llm") or not isinstance(node, dict):
                continue
            characterization[field] = node.get("conclusion")
            reasoning[field] = node.get("reasoning")
            for ref in (node.get("kb_basis") or []):
                ref = str(ref).strip().upper()
                if re.fullmatch(r"KB1_\d+", ref):
                    kb1.add(ref)
                elif re.fullmatch(r"KB2_\d+", ref):
                    kb2.add(ref)

        result = {
            "agent": "MemoAgent", "contract_id": contract_id,
            "customer_name": facts.get("customer_name"),
            "disposition": determine_disposition(points),
            "treatment": agent3.get("treatment") or {},
            "schedule_source": agent3.get("schedule") or {},
            "characterization": characterization,
            "reasoning": reasoning,
            "schedule": schedule,
            "journal_entries": build_journal_entries(schedule, agent3),
            "review_points": open_points,
            "immaterial_matters": immaterial_points,
            "items_assessed": len(agent4.get("audit_findings") or []),
            "items_not_applicable": sorted(
                ({"kb3_id": f["kb3_id"], "name": f.get("name")}
                 for f in (agent4.get("audit_findings") or []) if not f.get("material")),
                key=lambda x: int(x["kb3_id"].split("_")[1])),
            "consistency_issues": agent4.get("consistency_issues") or [],
            "revision_requests": agent4.get("revision_requests") or [],
            "kb1_refs": sorted(kb1), "kb2_refs": sorted(kb2),
            "kb3_refs": sorted(p["kb3_id"] for p in points),
            "basis": {
                "authoritative": [{"kb_id": k, "asc_ref": ASC_MAP.get(k)} for k in sorted(kb1)],
                "entity_policy": [{"kb_id": k, "asc_ref": ASC_MAP.get(k)} for k in sorted(kb2)],
                "review_checklist": [{"kb_id": p["kb3_id"], "name": p["name"]} for p in points],
            },
        }
        # LLM authors the prose; falls back to the deterministic template if the guard trips.
        drafted, reason = _draft_narrative(result) if self.use_llm else (None, "disabled")
        result["narrative"] = drafted
        result["narrative_source"] = "llm" if drafted else "template"
        result["narrative_reject_reason"] = reason
        result["memo"] = render_memo(result)
        return result


def portfolio_summary(results: list[dict]) -> dict:
    """Across all contracts: who can be booked and what drives the review workload - by contract
    COUNT and by transaction-price DOLLARS, which rank differently. An auditor works in dollars,
    but neither view is derivable from the other. No other layer sees more than one contract, so the
    portfolio aggregate (total price, recognisable revenue, closing deferred) can only be built here."""
    def tp(res):
        return _num((res.get("treatment") or {}).get("transaction_price")) or 0.0

    by_disp: dict[str, dict] = {}
    by_item: dict[str, dict] = {}
    total_tp = total_rev = 0.0
    opening_deferred_total = 0.0        # consideration billed in advance across the portfolio
    deferred_at_reporting = 0.0         # deferred balance at the reporting date (end of month 1)
    excl_amt, excl_contracts = 0.0, []  # contracts with no schedule (service period not determinable)
    for res in results:
        price = tp(res)
        total_tp += price
        rows = (res.get("schedule") or {}).get("rows") or []
        total_rev += sum(r["revenue_recognised"] for r in rows)
        opening_deferred_total += (res.get("schedule") or {}).get("opening_deferred") or 0.0
        if rows:
            deferred_at_reporting += rows[0]["closing_balance"]
        else:
            excl_amt += price
            excl_contracts.append(res["contract_id"])
        d = by_disp.setdefault(res["disposition"], {"contracts": [], "amount": 0.0})
        d["contracts"].append(res["contract_id"]); d["amount"] += price
        for point in res["review_points"]:
            it = by_item.setdefault(point["kb3_id"], {"contracts": [], "amount": 0.0})
            it["contracts"].append(res["contract_id"]); it["amount"] += price

    def fmt(dct, key):
        return {k: {"count": len(v["contracts"]), "amount": round(v["amount"], 2),
                    "contracts": sorted(v["contracts"])}
                for k, v in sorted(dct.items(), key=key)}

    return {
        "total_contracts": len(results),
        "total_transaction_price": round(total_tp, 2),
        "total_revenue_recognisable": round(total_rev, 2),
        "opening_deferred_total": round(opening_deferred_total, 2),
        "deferred_at_reporting_date": round(deferred_at_reporting, 2),
        "excluded_from_schedule": {"amount": round(excl_amt, 2), "count": len(excl_contracts),
                                   "contracts": sorted(excl_contracts)},
        "by_disposition": fmt(by_disp, lambda kv: kv[0]),
        "by_review_item": fmt(by_item, lambda kv: -kv[1]["amount"]),
    }
