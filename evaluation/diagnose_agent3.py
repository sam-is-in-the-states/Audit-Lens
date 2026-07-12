"""
diagnose_agent3.py - attribute each Agent 3 error to Agent 1, 2, or 3.

For every scored (contract, field) Agent 3 got wrong vs ground truth:
  Agent 1 - the fact this field needs is missing/empty in Agent 1's extraction
  Agent 2 - the fact was there, but the KB policy this field needs was not retrieved
  Agent 3 - fact present and policy retrieved, but the output is still wrong

Reads: app/llm/extracted_contracts.json, evaluation/results/treatment_outputs.json,
       evaluation/results/policy_retrieval_results.json (optional), GroundTruth.xlsx
Run from evaluation/:  py -3.12 diagnose_agent3.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from load_ground_truth import load_ground_truth
from evaluate_agent3 import flatten, compare, field_ground_truth, ALL_FIELDS

EXTRACTED = ROOT / "app" / "llm" / "extracted_contracts.json"
AGENT3 = HERE / "results" / "treatment_outputs.json"
AGENT2 = HERE / "results" / "policy_retrieval_results.json"
OUT = HERE / "results" / "error_attribution.csv"
CONTRACT_IDS = [f"C{i:02d}" for i in range(1, 51)]

# Which KB policies each field needs (for Agent 2 attribution; KB1/KB2 only).
FIELD_KB = {
    "onboarding_distinct": {"KB1_3", "KB1_4", "KB2_2", "KB2_3"},
    "usage_model": {"KB1_8", "KB1_9", "KB1_10", "KB1_11", "KB2_4", "KB2_5", "KB2_6", "KB2_7"},
    "discount_type": {"KB1_12", "KB1_13", "KB1_14", "KB2_8", "KB2_9"},
    "material_right": {"KB1_15", "KB1_16", "KB2_10", "KB2_11"},
    "recognition_method": {"KB1_5", "KB1_6", "KB1_7", "KB2_1", "KB2_4"},
    "transaction_price": {"KB1_12", "KB1_14", "KB2_8"},
    "opening_deferred": {"KB1_17", "KB2_12"},
}


def _amt(node):
    return node.get("amount") if isinstance(node, dict) else node


def agent1_fact_missing(field, vo):
    """Did Agent 1 fail to provide the fact this field depends on? -> (bool, reason)."""
    fees = vo.get("fees") or []
    has_fee_amount = any(_amt(f.get("amount")) for f in fees)

    if field == "onboarding_distinct":
        ot = vo.get("onboarding_terms") or {}
        signal = (vo.get("implementation_required_for_platform") is not None
                  or ot.get("required_for_platform") is not None
                  or ot.get("required") is not None)
        if not signal:
            return True, "no onboarding distinctness signal (optional/standardized/required_for_platform)"
    elif field == "usage_model":
        if not vo.get("usage_fee"):
            return True, "usage_fee not extracted"
    elif field == "discount_type":
        dt = vo.get("discount_terms") or {}
        if not (dt.get("has_discount") or dt.get("amount") is not None):
            return True, "discount not extracted (has_discount=False)"
    elif field == "material_right":
        rt = vo.get("renewal_terms") or {}
        if _amt(rt.get("renewal_price")) is None and not rt.get("renewal_pricing_basis"):
            return True, "no renewal price/basis to judge material right"

    # schedule / derived fields depend on term, billing and fee amounts
    if field in ("recognition_period_months", "recognition_method",
                 "monthly_revenue", "opening_deferred"):
        if vo.get("subscription_term_months") is None:
            return True, "subscription term not extracted"
    if field == "opening_deferred":
        if not any((f.get("payment_timing") or f.get("billing_frequency")) for f in fees):
            return True, "no billing signal (Agent 1 has no billing field)"
    if field in ("transaction_price", "monthly_revenue", "opening_deferred") and not has_fee_amount:
        return True, "no fee amounts extracted"

    return False, ""


def load_agent2_missing():
    """Per-contract set of GT-expected KB1/KB2 ids that Agent 2 failed to retrieve."""
    if not AGENT2.exists():
        return {}
    out = {}
    for r in json.loads(AGENT2.read_text(encoding="utf-8")):
        cid = str(r.get("ground_truth_contract_id", "")).upper()
        out[cid] = {i for i in r.get("missing_kb_ids", []) if i.startswith(("KB1_", "KB2_"))}
    return out


def attribute(field, vo, agent2_missing):
    miss, reason = agent1_fact_missing(field, vo)
    if miss:
        return "Agent 1", reason
    gap = FIELD_KB.get(field, set()) & agent2_missing
    if gap:
        return "Agent 2", f"policy not retrieved: {sorted(gap)}"
    return "Agent 3", "fact + policy present, output still wrong"


def main():
    gt = load_ground_truth()
    a1 = {r["file_name"].split("_")[0]: (r.get("validated_output") or r.get("candidate_output") or {})
          for r in json.loads(EXTRACTED.read_text(encoding="utf-8"))}
    a3 = {o["source_file"].split("_")[0]: o
          for o in json.loads(AGENT3.read_text(encoding="utf-8"))}
    a2_missing = load_agent2_missing()

    rows = []
    for cid in CONTRACT_IDS:
        out = a3.get(cid)
        if out is None:
            continue
        pred = flatten(out)
        truth, onb_fee = field_ground_truth(cid, gt)
        vo = a1.get(cid, {})
        for field in ALL_FIELDS:
            if field == "onboarding_distinct" and not (onb_fee or 0) > 0:
                continue
            result = compare(field, pred[field], truth[field])
            if result is None or result:      # skip un-scored and correct ones
                continue
            cause, reason = attribute(field, vo, a2_missing.get(cid, set()))
            rows.append({
                "contract_id": cid,
                "type": gt[cid]["characterisation"].get("type"),
                "field": field,
                "predicted": pred[field],
                "truth": truth[field],
                "cause": cause,
                "reason": reason,
            })

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    pd.set_option("display.width", 0)
    print(f"Total errors: {len(df)}")
    print("\n=== errors by cause ===")
    print(df["cause"].value_counts().to_string())
    print("\n=== errors by field x cause ===")
    print(df.groupby(["field", "cause"]).size().unstack(fill_value=0).to_string())
    print(f"\nWritten: {OUT}")


if __name__ == "__main__":
    main()
