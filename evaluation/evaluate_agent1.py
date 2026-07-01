"""
Evaluate Agent 1's extraction against ground truth, per the Agent 1 output schema.

Agent 1 extracts FACTS only (no ASC 606 judgments), so this scores three things:
  1. Accuracy      - for the fact fields that can be compared to ground truth, how
                     often does Agent 1's value match?
  2. Completeness  - for each fact field, on how many of the 50 contracts did Agent 1
                     produce a non-empty value? (surfaces the [NEW] schema gaps)
  3. Sufficiency   - for each Agent 2 judgment, on how many of the relevant contracts did
                     Agent 1 supply the trigger fact Agent 2 needs to retrieve the right KB?

Input:  app/llm/extracted_contracts.json  (Agent 1 output; nested schema)
        evaluation/GroundTruth.xlsx

Run from the evaluation/ folder:
    py evaluate_agent1.py                 # evaluate the current extracted_contracts.json
    py evaluate_agent1.py --input PATH    # evaluate a different Agent 1 output file
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from load_ground_truth import load_ground_truth, _NUMERIC_FIELDS

HERE = Path(__file__).parent
DEFAULT_INPUT = HERE.parent / "app" / "llm" / "extracted_contracts.json"
CONTRACT_IDS = [f"C{i:02d}" for i in range(1, 51)]

# Fact fields we can compare to ground truth (facts only; judgments are excluded).
COMPARE_FIELDS = [
    "customer_name",
    "term_months",
    "start_date",
    "annual_subscription",
    "onboarding_fee",
    "usage_rate",
    "discount_amount",
]

_EMPTY = {"", "n/a", "na", "none", "null", "nil"}
_MISSING = "__MISSING__"   # Agent 1's schema has no place for this / did not attempt it
NUM_ABS_TOL = 1.0
NUM_REL_TOL = 0.005

_IMPL_KEYWORDS = {"implementation", "setup", "onboarding", "training", "activation", "migration", "data migration"}


# ── value extraction: nested Agent 1 record -> flat GT-comparable fields ──────

def _amount(node):
    """C stores money as {'amount': x, 'currency': ...}; return the number or None."""
    if isinstance(node, dict):
        return node.get("amount")
    if isinstance(node, (int, float)):
        return node
    return None


def _pick_fees(fees):
    """Split fees[] into (subscription_amount, onboarding_amount) by fee name/type."""
    sub, onb = None, None
    for f in fees or []:
        name = (f.get("name") or "").lower()
        ftype = (f.get("fee_type") or "").lower()
        amt = _amount(f.get("amount"))
        if amt is None:
            continue
        is_impl = ftype == "implementation" or any(k in name for k in _IMPL_KEYWORDS)
        if is_impl:
            if onb is None:
                onb = amt
        else:
            if sub is None or amt > sub:   # subscription = largest non-impl fee
                sub = amt
    return sub, onb


def map_to_gt_fields(record):
    """Translate one Agent 1 record into the flat fields we compare to ground truth.
    A field is _MISSING when Agent 1's schema does not provide it at all."""
    vo = record.get("validated_output") or record.get("candidate_output") or {}
    fees = vo.get("fees") or []
    usage = vo.get("usage_fee") or {}
    discount = vo.get("discount") or {}

    sub_amt, onb_amt = _pick_fees(fees)

    return {
        "customer_name":       vo.get("customer_name"),
        "term_months":         vo.get("subscription_term_months"),
        "start_date":          _norm_date(vo.get("start_date") or vo.get("effective_date")),
        "annual_subscription": sub_amt,
        "onboarding_fee":      onb_amt,
        "usage_rate":          _amount(usage.get("overage_rate")) if usage else None,
        # discount block is [NEW] in the schema; if absent entirely, mark MISSING (not wrong)
        "discount_amount":     discount.get("discount_amount") if discount else _MISSING,
    }


def _norm_date(v):
    if v is None:
        return None
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d")
    return str(v).strip()


# ── comparison (same normalisation rules as the baseline evaluation) ─────────

def _to_float(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.replace("$", "").replace(",", "").strip()
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _norm(v):
    return str(v).strip().lower() if v is not None else ""


def _is_empty(v):
    return v is None or (isinstance(v, str) and v.strip().lower() in _EMPTY)


def compare(field, predicted, truth):
    """Return True/False if comparable, or None if skipped (not attempted, or no GT)."""
    if predicted is _MISSING:
        return None                      # Agent 1's schema doesn't provide it -> not scored as wrong
    if _is_empty(truth):
        return None                      # ground truth has nothing to compare
    if _is_empty(predicted):
        return False                     # GT expects a value, Agent 1 gave none -> wrong

    if field in _NUMERIC_FIELDS:
        p, t = _to_float(predicted), _to_float(truth)
        if p is None or t is None:
            return _norm(predicted) == _norm(truth)
        return abs(p - t) <= NUM_ABS_TOL or (t != 0 and abs(p - t) / abs(t) <= NUM_REL_TOL)

    return _norm(predicted) == _norm(truth)


# ── dimension 3: trigger facts Agent 2 needs to retrieve the right KB ────────
# Each entry: which contracts are relevant (from GT), and whether Agent 1 supplied the
# trigger fact that lets Agent 2 retrieve the correct guidance.
def sufficiency(records_by_cid, gt):
    def vo(c):
        r = records_by_cid.get(c, {})
        return r.get("validated_output") or r.get("candidate_output") or {}

    rows = []

    # onboarding distinct (KB1_3/KB2_2-3): relevant = contracts with an onboarding fee
    relevant = [c for c in CONTRACT_IDS if (gt[c]["characterisation"].get("onboarding_fee") or 0) > 0]
    supplied = [c for c in relevant if vo(c).get("implementation_required_for_platform") is not None]
    rows.append(("onboarding distinct", "implementation_required_for_platform", "KB1_3/KB2_2-3", len(supplied), len(relevant)))

    # variable consideration (KB1_8/KB2_4-7): relevant = usage contracts
    relevant = [c for c in CONTRACT_IDS if gt[c]["characterisation"].get("usage_model") not in (None, "none")]
    supplied = [c for c in relevant if vo(c).get("usage_fee")]
    rows.append(("variable consideration", "usage_fee present", "KB1_8/KB2_4-7", len(supplied), len(relevant)))
    # + the rate specifically (needed to actually measure it)
    supplied_rate = [c for c in relevant if _amount((vo(c).get("usage_fee") or {}).get("overage_rate"))]
    rows.append(("  usage rate for measurement", "usage_fee.overage_rate", "KB2_4", len(supplied_rate), len(relevant)))

    # discount allocation (KB1_14/KB2_8): relevant = contracts with a discount
    relevant = [c for c in CONTRACT_IDS if (gt[c]["characterisation"].get("discount_amount") or 0) > 0]
    supplied = [c for c in relevant if (vo(c).get("discount") or {}).get("discount_amount") is not None]
    rows.append(("discount allocation", "discount.discount_amount", "KB1_14/KB2_8", len(supplied), len(relevant)))

    # material right (KB1_15/KB2_10): relevant = contracts GT flags as material right
    relevant = [c for c in CONTRACT_IDS if gt[c]["characterisation"].get("material_right") is True]
    supplied = [c for c in relevant if _amount((vo(c).get("renewal_terms") or {}).get("renewal_price"))]
    rows.append(("material right", "renewal_terms.renewal_price", "KB1_15/KB2_10", len(supplied), len(relevant)))

    df = pd.DataFrame(rows, columns=["agent2_judgment", "trigger_fact", "kb", "supplied", "relevant"])
    df["coverage_%"] = (df["supplied"] / df["relevant"].replace(0, pd.NA) * 100).round(1)
    return df


# ── main evaluation ──────────────────────────────────────────────────────────

def evaluate(input_path):
    gt = load_ground_truth()
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    by_cid = {rec["file_name"].split("_")[0]: rec for rec in data}

    acc_rows = []       # per contract/field accuracy
    comp_rows = []      # per contract/field attempted-or-not
    for cid in CONTRACT_IDS:
        rec = by_cid.get(cid)
        if rec is None:
            continue
        mapped = map_to_gt_fields(rec)
        truth = gt[cid]["characterisation"]
        for field in COMPARE_FIELDS:
            p = mapped.get(field, _MISSING)
            t = truth.get(field)
            attempted = not (p is _MISSING or _is_empty(p))
            comp_rows.append({"field": field, "attempted": attempted})
            res = compare(field, p, t)
            if res is not None:
                acc_rows.append({"field": field, "correct": res})

    acc = pd.DataFrame(acc_rows)
    comp = pd.DataFrame(comp_rows)

    # 1. Accuracy per field
    acc_tbl = (acc.groupby("field")["correct"].agg(correct="sum", scored="count").reset_index())
    acc_tbl["accuracy_%"] = (acc_tbl["correct"] / acc_tbl["scored"] * 100).round(1)

    # 2. Completeness per field (attempted on how many of 50)
    comp_tbl = (comp.groupby("field")["attempted"].agg(attempted="sum", total="count").reset_index())
    comp_tbl["coverage_%"] = (comp_tbl["attempted"] / comp_tbl["total"] * 100).round(1)

    # 3. Sufficiency for Agent 2
    suff_tbl = sufficiency(by_cid, gt)

    pd.set_option("display.width", 0)
    print("\n=== 1. Accuracy (fact fields comparable to ground truth) ===")
    print(acc_tbl.to_string(index=False))

    print("\n=== 2. Completeness (fields Agent 1 produced, out of 50) ===")
    print(comp_tbl.to_string(index=False))

    print("\n=== 3. Sufficiency for Agent 2 (did Agent 1 supply the KB trigger fact?) ===")
    print(suff_tbl.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT),
                    help="path to Agent 1 output json (default: app/llm/extracted_contracts.json)")
    args = ap.parse_args()
    evaluate(args.input)


if __name__ == "__main__":
    main()