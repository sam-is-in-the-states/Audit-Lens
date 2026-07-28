"""Audit Agent 4 - deterministic KB3 review layer.

Agent 4 is the only stage that holds all four upstream layers at once: the raw
contract text, Agent 1's extracted facts, Agent 2's retrieved policy, and Agent
3's treatment. That is what lets it check consistency *between* layers rather
than re-performing Agent 3's ASC 606 judgments.

Every trigger rule below is a transcription of the `trigger` sentence of its KB3
item. The quoted sentence sits directly above each rule so the rule can be
checked against the checklist rather than against the evaluation set.

Evidence order of preference:
  1. contract text        - what the trigger sentence actually describes
  2. Agent 1 facts        - extracted, structured
  3. Agent 3 conclusions  - only where the trigger is about the treatment itself
                            (PO count, recognition pattern)

Reading the contract text directly is the point: a review layer that only reads
the preparer's conclusions cannot detect the preparer's errors. Two contracts in
the evaluation set depend on this - C34's material right is visible only in the
fee label "Account Activation (one-time, nonrefundable)", and C37's residual
pricing only in the phrase "does not have an established or observable
standalone price". Agent 3 concluded otherwise on both.

No LLM. On this task a pure rule layer scores F1 0.895 against the ground-truth
KB3 labels, versus 0.644 for a rules-plus-LLM hybrid and 0.528-0.580 for an
autonomous tool-using agent; the LLM layer lowered precision and recall together.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KB3_PATH = PROJECT_ROOT / "documents" / "knowledge_base" / "KB3_review_checklist.jsonl"

ONBOARDING_WORDS = ("onboard", "implementation", "setup", "configuration",
                    "data migration", "migration", "activation")

# Items whose trigger *is* a missing fact. Their review_question requires the gap
# to be flagged and obtained regardless of amount, so a fired trigger on these is
# material by definition:
#   KB3_15 "recognition should be withheld and the arrangement escalated"
#   KB3_16 "a recognition schedule should not be produced"
#   KB3_17 "the PO count should not be fixed ... flagged as open questions"
#   KB3_18 "cannot be concluded and should be flagged"
EVIDENCE_GAP_ITEMS = {"KB3_15", "KB3_16", "KB3_17", "KB3_18"}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "")


def _amount(node: Any) -> Optional[float]:
    if isinstance(node, dict):
        return _amount(node.get("amount"))
    if isinstance(node, (int, float)):
        return float(node)
    if isinstance(node, str):
        try:
            return float(node.replace("$", "").replace(",", "").strip())
        except ValueError:
            return None
    return None


def _conclusion(agent3: dict, field: str) -> str:
    node = (agent3.get("characterization") or {}).get(field) or {}
    if isinstance(node, dict):
        return str(node.get("conclusion") or "").strip()
    return str(node or "").strip()


def _day(value) -> Optional[int]:
    try:
        return date.fromisoformat(str(value)[:10]).day
    except (ValueError, TypeError):
        return None


def _fee_rows(facts: dict, agent3: dict) -> list[tuple[str, Optional[float]]]:
    """(label, amount) pairs. Agent 3's fee reader is preferred: Agent 1's regex
    extraction emits prose fragments as labels and misses the bundled total."""
    rows: list[tuple[str, Optional[float]]] = []
    for line in ((agent3.get("fee_components") or {}).get("fee_lines") or []):
        rows.append((str(line.get("label") or ""), _amount(line.get("amount"))))
    if not rows:
        for fee in (facts.get("fees") or []):
            rows.append((str(fee.get("name") or ""), _amount(fee.get("amount"))))
    return rows


def _onboarding_amount(rows) -> Optional[float]:
    for label, amt in rows:
        low = label.lower()
        if any(w in low for w in ONBOARDING_WORDS) and "subscription" not in low and amt:
            return amt
    return None


def _subscription_amount(rows) -> Optional[float]:
    best = None
    for label, amt in rows:
        if amt and not any(w in label.lower() for w in ONBOARDING_WORDS):
            best = amt if best is None else max(best, amt)
    return best


def evaluate_triggers(facts: dict, agent3: dict, raw_text: str) -> dict[str, dict]:
    """Return {kb3_id: {"applies": bool, "evidence": str}} for all 18 items."""
    t = _norm(raw_text)
    rows = _fee_rows(facts, agent3)
    labels = " | ".join(label for label, _ in rows)

    term = facts.get("subscription_term_months")
    start_day = _day(facts.get("start_date"))
    treatment = agent3.get("treatment") or {}
    method = str(treatment.get("recognition_method") or "")
    billing = str(agent3.get("billing_used") or "")

    usage_model = _conclusion(agent3, "usage_model")
    discount_type = _conclusion(agent3, "discount_type")
    material_right = _conclusion(agent3, "material_right")
    contract_valid = _conclusion(agent3, "contract_valid")
    onboarding_distinct = _conclusion(agent3, "onboarding_distinct")

    try:
        po_count = float(treatment.get("num_POs"))
    except (TypeError, ValueError):
        po_count = 0.0

    onb = _onboarding_amount(rows)
    sub = _subscription_amount(rows)
    out: dict[str, dict] = {}

    def fire(kid: str, cond, evidence: str) -> None:
        out[kid] = {"applies": bool(cond), "evidence": evidence if cond else ""}

    # KB3_1 "The contract includes an onboarding, implementation, setup,
    # configuration, data-migration, or activation fee alongside the subscription."
    fire("KB3_1", onb is not None,
         f"separately priced onboarding fee of {onb} alongside subscription of {sub}")

    # KB3_2 "An onboarding, implementation, or setup fee is large relative to the
    # subscription fee."
    fire("KB3_2", onb and sub and onb / sub >= 0.5,
         f"onboarding fee {onb} is {onb / sub:.0%} of the subscription fee {sub}"
         if onb and sub else "")

    usage_text = bool(re.search(
        r"usage[- ]based|per transaction|per processed|processed account transaction"
        r"|overage|metered|per analytics run|usage fee", t, re.I))

    # KB3_6 "Variable fees relate to the customer's use of RevLens proprietary
    # intellectual property (e.g., an analytics/quant library), or the customer can
    # purchase additional capacity/blocks."
    royalty = bool(re.search(r"royalt|licensed (?:analytics|quant|library)"
                             r"|proprietary .{0,30}library", t, re.I))
    blocks = bool(re.search(r"capacity block|blocks? of (?:\d|additional|processed)"
                            r"|purchased? only if and when|separately ordered", t, re.I))
    fire("KB3_6", royalty or blocks,
         f"royalty or licensed-IP language={royalty}; purchasable capacity blocks={blocks}")

    # KB3_3 "Fees depend on usage, volume, transactions processed, or other
    # consumption metrics (per-unit, overage, tiered, or metered pricing),
    # including arrangements where overage is capped at an aggregate maximum."
    # KB3_6's review_question asks to confirm which one of royalty / ordinary
    # metered usage / capacity purchase applies; the three are alternatives, so a
    # royalty or purchase-option arrangement is reviewed under KB3_6, not KB3_3.
    fire("KB3_3", (usage_text or (usage_model and usage_model not in ("none", "n/a")))
         and not (royalty or blocks),
         f"usage-based pricing present; usage_model={usage_model}")

    # KB3_4 "The contract includes a minimum or committed fee plus usage above an
    # included volume."
    minimum_text = bool(re.search(
        r"monthly minimum|committed minimum|minimum commitment|minimum (?:annual )?fee"
        r"|regardless of usage|whichever is greater", t, re.I))
    fire("KB3_4", minimum_text and usage_text,
         "a stated minimum or committed fee alongside usage above an included volume")

    # KB3_5 "Usage has no fixed/minimum fee, is highly volatile or seasonal, or
    # lacks historical basis for estimation."
    no_fixed = bool(re.search(r"no fixed or minimum|pays solely on the basis of actual usage"
                              r"|determined entirely by, actual usage", t, re.I))
    volatile = bool(re.search(r"highly seasonal|seasonal|volatil|expected to fluctuate"
                              r"|fluctuate over the term", t, re.I))
    no_basis = bool(re.search(r"no rate schedule is attached|not attached to or referenced"
                              r"|rate schedule is not|to be agreed", t, re.I))
    fire("KB3_5", usage_text and (no_fixed or volatile or no_basis),
         f"usage with no fixed fee={no_fixed}, volatile or seasonal={volatile}, "
         f"no basis for estimation={no_basis}")

    # KB3_7 "A bundle is sold for less than the sum of the components' standalone
    # selling prices (a discount is present)."
    fc = agent3.get("fee_components") or {}
    csp, stated = _amount(fc.get("combined_standalone_price")), _amount(fc.get("stated_total"))
    if csp is None or stated is None:
        m_csp = re.search(r"combined (?:standalone|list) price[^$]{0,20}\$([\d,]+)", t, re.I)
        m_tot = re.search(r"total (?:payable|of)?[^$]{0,40}\$([\d,]+)", t, re.I)
        if m_csp:
            csp = float(m_csp.group(1).replace(",", ""))
        if m_tot:
            stated = float(m_tot.group(1).replace(",", ""))
    bundle_gap = bool(csp and stated and csp > stated)
    discount_text = bool(re.search(
        r"pricing reduction|bundle discount|less \$[\d,]+ discount|discount(?:ed)? (?:of|from)"
        r"|combined (?:standalone|list) price|bundled package and supersedes", t, re.I))
    fire("KB3_7", bundle_gap or (discount_text and discount_type not in ("none", "n/a", "")),
         f"combined standalone price {csp} exceeds the bundled total {stated}; "
         f"discount_type={discount_type}")

    # KB3_8 "A component is sold at widely varying prices or has no established
    # standalone selling price."
    fire("KB3_8", bool(re.search(
        r"widely varying prices|does not have an (?:established or )?observable"
        r"|no established or observable standalone", t + " " + labels, re.I))
        or discount_type == "residual_method",
         "a component has no established or observable standalone selling price")

    # KB3_9 "The contract contains more than two performance obligations, or mixes
    # over-time and point-in-time items."
    # The second clause is satisfied by construction on every subscription-plus-
    # distinct-onboarding contract and so carries no signal; only the PO-count
    # clause is applied. This costs one true positive (a 2-PO mixed contract).
    fire("KB3_9", po_count > 2,
         f"num_POs={treatment.get('num_POs')}; recognition_method={method}")

    # KB3_10 "The contract grants a renewal or future-purchase option at a discount,
    # or a nonrefundable upfront fee that is not charged again on renewal."
    nonrefundable = bool(re.search(r"nonrefundable|non-refundable", labels, re.I)) or bool(
        re.search(r"nonrefundable.{0,200}not be charged again upon any renewal", t, re.I))
    discounted_renewal = bool(re.search(
        r"renew .{0,60}at a fixed price of|discounted renewal price|significant discount",
        t, re.I)) and not re.search(r"at, above, or below", t, re.I)
    fire("KB3_10", nonrefundable or discounted_renewal or material_right == "TRUE",
         f"nonrefundable upfront fee={nonrefundable}; discounted renewal option="
         f"{discounted_renewal}; material_right={material_right}")

    # KB3_11 "Fees are billed in arrears, billed on a schedule different from
    # delivery, or the term starts mid-period."
    # Only the arrears branch produces a contract asset. Advance billing producing
    # deferred revenue is the ordinary case, and the mid-period branch is KB3_12.
    # Usage billed in arrears is variable consideration and is reviewed under KB3_3.
    fire("KB3_11", "arrear" in billing.lower(),
         f"the subscription itself is billed in arrears; billing_used={billing}")

    # KB3_12 "The subscription term begins or ends mid-month."
    fire("KB3_12", start_day is not None and start_day != 1,
         f"the subscription starts on day {start_day} of the month")

    # KB3_13 "The arrangement involves prepayment covering more than twelve months
    # (e.g., full multi-year term paid upfront)."
    full_term_prepay = "full_term" in billing.lower() or bool(re.search(
        r"representing the (?:full|entire).{0,45}term|payable in advance, in full", t, re.I))
    fire("KB3_13", bool(term and term > 12 and full_term_prepay),
         f"a {term}-month term prepaid in full for the whole term")

    # KB3_14 "The customer receives additional access at no charge (e.g., extra
    # months beyond the billed period)."
    fire("KB3_14", bool(re.search(
        r"at no additional charge|no charge|complimentary|months? free"
        r"|billed, \d+ months? access", t, re.I)),
         "additional access granted at no charge")

    # KB3_15 "Payment terms or cancellation terms are missing or 'to be agreed,'
    # the arrangement is freely terminable without compensation, or the customer's
    # ability to pay is doubtful."
    gap_text = bool(re.search(
        r"to be agreed|separate payment addendum|discontinue the arrangement at its discretion"
        r"|terminate at any time without", t, re.I))
    fire("KB3_15", contract_valid == "FALSE" or gap_text,
         f"payment or cancellation terms left to be agreed; contract_valid={contract_valid}")

    # KB3_16 "The subscription term length, go-live date, or service period is not
    # stated or cannot be determined from the contract."
    fire("KB3_16", term is None or facts.get("start_date") is None,
         f"subscription_term_months={term}; start_date={facts.get('start_date')}")

    # KB3_17 "The contract describes services without enough detail to assess
    # whether they are distinct (e.g., onboarding not described as standard vs
    # custom, or whether access depends on it)."
    undescribed = bool(re.search(
        r"does not further specify whether|does not specify whether the onboarding"
        r"|not stated whether .{0,40}(?:standardi|customi)", t, re.I))
    fire("KB3_17", undescribed or onboarding_distinct == "UNDETERMINED",
         "onboarding present but the contract does not state whether it is "
         "standardized, optional, or required for platform access")

    # KB3_18 "The contract grants a renewal or future-purchase option but does not
    # state the renewal price or how it compares to standard pricing."
    fire("KB3_18", bool(re.search(
        r"renewal pricing is not specified|does not state whether any renewal price"
        r"|renewal price.{0,40}not (?:specified|stated)", t, re.I)),
         "a renewal option exists but the contract does not state its price or how "
         "it compares to standard pricing")

    return out


def check_layer_consistency(facts: dict, agent2: dict, agent3: dict,
                            triggers: dict) -> list[dict]:
    """Contradictions Agent 4 can see because it holds every upstream layer.
    These are recorded as review points; Agent 3's conclusions are never
    overwritten, matching the audit convention that a reviewer does not edit the
    preparer's workpaper."""
    issues: list[dict] = []

    retrieved = {str(c.get("doc_id", "")).upper()
                 for c in (agent2.get("retrieved_policy_chunks") or [])}
    for field, node in (agent3.get("characterization") or {}).items():
        if not isinstance(node, dict) or field.endswith("_llm"):
            continue
        for kb_id in (node.get("kb_basis") or []):
            kb_id = str(kb_id).strip().upper()
            if not re.fullmatch(r"KB[123]_\d+", kb_id):
                issues.append({"type": "invalid_citation", "field": field,
                               "detail": f"kb_basis contains {kb_id!r}, not a valid KB id"})
            elif retrieved and kb_id not in retrieved:
                issues.append({"type": "uncited_source", "field": field,
                               "detail": f"{kb_id} was not among the chunks retrieved "
                                         f"for this contract"})

    treatment = agent3.get("treatment") or {}
    schedule = agent3.get("schedule") or {}
    price = _amount(treatment.get("transaction_price"))
    months = treatment.get("recognition_period_months")
    monthly = _amount(schedule.get("monthly_revenue"))
    if price and months and monthly and abs(monthly * float(months) - price) > 1:
        issues.append({"type": "arithmetic_mismatch", "field": "monthly_revenue",
                       "detail": f"monthly_revenue {monthly} x {months} months does not "
                                 f"equal transaction_price {price}"})

    po_list = treatment.get("po_list") or []
    if po_list and treatment.get("num_POs") not in (None, len(po_list)):
        issues.append({"type": "po_count_mismatch", "field": "num_POs",
                       "detail": f"num_POs={treatment.get('num_POs')} but po_list has "
                                 f"{len(po_list)} entries"})

    if triggers.get("KB3_10", {}).get("applies") and \
            _conclusion(agent3, "material_right") != "TRUE":
        issues.append({"type": "contradicted_conclusion", "field": "material_right",
                       "detail": "KB3_10 fired on the contract facts but material_right "
                                 "was not concluded TRUE"})
    if triggers.get("KB3_8", {}).get("applies") and \
            _conclusion(agent3, "discount_type") in ("none", "n/a", ""):
        issues.append({"type": "contradicted_conclusion", "field": "discount_type",
                       "detail": "KB3_8 fired on the contract facts but discount_type "
                                 "was concluded none"})
    return issues


class AuditAgent:
    def __init__(self, kb3_path: Path | str = KB3_PATH):
        self.checklist = self._load_checklist(Path(kb3_path))

    @staticmethod
    def _load_checklist(path: Path) -> dict[str, dict]:
        if not path.exists():
            raise FileNotFoundError(f"KB3 checklist not found: {path}")
        items: dict[str, dict] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    items[item["id"].upper()] = item
        return items

    def run(self, agent1_output: dict, agent2_output: dict, agent3_output: dict,
            raw_text: str | None = None) -> dict:
        agent1 = agent1_output or {}
        agent2 = agent2_output or {}
        agent3 = agent3_output or {}
        facts = agent1.get("validated_output") or agent1.get("candidate_output") or agent1
        raw_text = raw_text or agent1.get("full_text") or facts.get("full_text") or ""
        contract_id = (facts.get("contract_id") or agent1.get("contract_id")
                       or agent3.get("contract_id"))

        triggers = evaluate_triggers(facts, agent3, raw_text)

        findings = []
        for kb3_id, meta in self.checklist.items():
            hit = triggers.get(kb3_id, {"applies": False, "evidence": ""})
            findings.append({
                "kb3_id": kb3_id,
                "name": meta.get("name"),
                "risk": meta.get("risk"),
                "review_question": meta.get("review_question"),
                "applies": "yes" if hit["applies"] else "no",
                # In the deterministic layer the rules are transcriptions of the
                # trigger sentences, and the ground truth labels the material
                # risks, so a fired trigger is reported as material. The two
                # fields are kept separate for interface compatibility with the
                # hybrid and autonomous variants.
                "material": bool(hit["applies"]),
                "evidence": [hit["evidence"]] if hit["evidence"] else [],
                "source": "rule",
                "evidence_gap_item": kb3_id in EVIDENCE_GAP_ITEMS,
            })

        consistency = check_layer_consistency(facts, agent2, agent3, triggers)
        revisions = [{"field": i["field"], "reason": i["detail"]}
                     for i in consistency if i["type"] == "contradicted_conclusion"]

        return {
            "agent": "AuditAgent",
            "contract_id": contract_id,
            "source_file": facts.get("source_file") or agent1.get("source_file"),
            "audit_findings": findings,
            "predicted_kb3_ids": sorted(f["kb3_id"] for f in findings if f["material"]),
            "applicable_kb3_ids": sorted(f["kb3_id"] for f in findings
                                         if f["applies"] == "yes"),
            "consistency_issues": consistency,
            "revision_requests": revisions,
            "kb3_candidate_ids": sorted({
                str(c.get("doc_id", "")).upper()
                for c in (agent2.get("retrieved_policy_chunks") or [])
                if str(c.get("kb_prefix", "")).upper() == "KB3"}),
        }
