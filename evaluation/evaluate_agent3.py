"""
Evaluate Agent 3 (Treatment Agent) against ground truth.

Two modes:
  py evaluate_agent3.py --run          # run Agent 3 over all contracts (calls the
                                       # LLM), save its outputs, then evaluate.
  py evaluate_agent3.py                 # evaluate the saved outputs only (no LLM).
  py evaluate_agent3.py --run --limit 5 # run only the first 5 (quick check).

The generated Agent 3 outputs are saved to evaluation/results/treatment_outputs.json,
which is the file Agent 4 consumes.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from load_ground_truth import load_ground_truth

EXTRACTED_PATH = ROOT / "app" / "llm" / "extracted_contracts.json"
OUTPUTS_PATH = HERE / "results" / "treatment_outputs.json"      # <- Agent 4 reads this
SUMMARY_PATH = HERE / "results" / "treatment_summary.csv"       # per-contract detail + field-level metrics

CONTRACT_IDS = [f"C{i:02d}" for i in range(1, 51)]

# Fields we score. Numeric ones use a tolerance; the rest are exact (normalized).
NUMERIC_FIELDS = {
    "num_POs", "transaction_price", "recognition_period_months",
    "monthly_revenue", "opening_deferred",
}
CATEGORICAL_FIELDS = {
    "onboarding_distinct", "usage_model", "discount_type",
    "material_right", "recognition_method",
}
ALL_FIELDS = list(CATEGORICAL_FIELDS) + list(NUMERIC_FIELDS)

NUM_ABS_TOL = 1.0
NUM_REL_TOL = 0.005
_EMPTY = {"", "n/a", "na", "none", "null", "nil"}


# ── generation: run Agent 3 and save the outputs Agent 4 will use ────────────

def generate_outputs(limit=None):
    from app.llm.treatment_agent import run as run_agent3
    from app.llm.policy_retrieval import PolicyRetrievalAgent

    extracted = json.loads(EXTRACTED_PATH.read_text(encoding="utf-8"))
    agent2 = PolicyRetrievalAgent()

    outputs = []
    for i, record in enumerate(extracted):
        if limit is not None and i >= limit:
            break
        facts = record.get("validated_output") or record.get("candidate_output") or {}
        agent2_output = agent2.run(facts)
        result = run_agent3(record, agent2_output)
        outputs.append(result)
        print(f"  ran {result.get('source_file')}")

    OUTPUTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUTS_PATH.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(f"Saved {len(outputs)} Agent 3 outputs -> {OUTPUTS_PATH}")
    return outputs


# ── flatten one Agent 3 output into the fields we compare to ground truth ────

def flatten(out):
    ch = out.get("characterization") or {}
    tr = out.get("treatment") or {}
    sc = out.get("schedule") or {}

    def conclusion(field):
        node = ch.get(field) or {}
        return node.get("conclusion") if isinstance(node, dict) else node

    return {
        "onboarding_distinct": conclusion("onboarding_distinct"),
        "usage_model": conclusion("usage_model"),
        "discount_type": conclusion("discount_type"),
        "material_right": conclusion("material_right"),
        "recognition_method": tr.get("recognition_method"),
        "num_POs": tr.get("num_POs"),
        "transaction_price": tr.get("transaction_price"),
        "recognition_period_months": tr.get("recognition_period_months"),
        "monthly_revenue": sc.get("monthly_revenue"),
        "opening_deferred": sc.get("opening_deferred"),
    }


# ── comparison helpers ───────────────────────────────────────────────────────

def _to_float(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.replace("$", "").replace(",", "").strip())
        except ValueError:
            return None
    return None


def _norm(v):
    return str(v).strip().lower() if v is not None else ""


def _is_empty(v):
    return v is None or (isinstance(v, str) and v.strip().lower() in _EMPTY)


def compare(field, predicted, truth):
    """True / False if scoreable, or None to skip (no ground truth to compare)."""
    if field in NUMERIC_FIELDS:
        t = _to_float(truth)
        if t is None:
            return None                       # GT has no number here -> skip
        p = _to_float(predicted)
        if p is None:
            return False
        return abs(p - t) <= NUM_ABS_TOL or (t != 0 and abs(p - t) / abs(t) <= NUM_REL_TOL)

    # categorical
    if _is_empty(truth):
        return None                           # nothing to compare against
    return _norm(predicted) == _norm(truth)


def field_ground_truth(cid, gt):
    """Pull the ground-truth value for each scored field from characterisation/treatment."""
    char = gt[cid]["characterisation"]
    treat = gt[cid]["treatment"]
    return {
        "onboarding_distinct": char.get("onboarding_distinct"),
        "usage_model": char.get("usage_model"),
        "discount_type": char.get("discount_type"),
        "material_right": char.get("material_right"),
        "recognition_method": treat.get("recognition_method"),
        "num_POs": treat.get("num_POs"),
        "transaction_price": treat.get("transaction_price"),
        "recognition_period_months": treat.get("recognition_period_months"),
        "monthly_revenue": treat.get("monthly_revenue"),
        "opening_deferred": treat.get("opening_deferred"),
    }, char.get("onboarding_fee")


def binary_metrics(detail, field, positive="true"):
    """Precision / recall / F1 for a binary judgment, positive class = TRUE.

    Accuracy alone hides how the model does on the rare TRUE cases (most contracts
    are FALSE). This reports whether those positives are actually caught.
    """
    rows = detail[detail["field"] == field]
    if rows.empty:
        return None

    def is_pos(v):
        return _norm(v) == positive

    tp = sum(is_pos(r.predicted) and is_pos(r.truth) for r in rows.itertuples())
    fp = sum(is_pos(r.predicted) and not is_pos(r.truth) for r in rows.itertuples())
    fn = sum(not is_pos(r.predicted) and is_pos(r.truth) for r in rows.itertuples())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "field": field, "TRUE_cases": tp + fn, "TP": tp, "FP": fp, "FN": fn,
        "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
    }


# ── evaluation ───────────────────────────────────────────────────────────────

def evaluate(outputs):
    gt = load_ground_truth()
    by_cid = {}
    for out in outputs:
        cid = (out.get("source_file") or "").split("_")[0].upper()
        if cid in CONTRACT_IDS:
            by_cid[cid] = out

    detail_rows = []
    for cid in CONTRACT_IDS:
        out = by_cid.get(cid)
        if out is None:
            continue
        pred = flatten(out)
        truth, onb_fee = field_ground_truth(cid, gt)

        for field in ALL_FIELDS:
            # onboarding_distinct is only a real judgment when there is an
            # onboarding fee; skip the trivial "no onboarding" contracts.
            if field == "onboarding_distinct" and not (onb_fee or 0) > 0:
                continue
            result = compare(field, pred[field], truth[field])
            if result is None:
                continue
            detail_rows.append({
                "contract_id": cid,
                "type": gt[cid]["characterisation"].get("type"),
                "field": field,
                "predicted": pred[field],
                "truth": truth[field],
                "correct": result,
            })

    detail = pd.DataFrame(detail_rows)

    # Per-field accuracy.
    per_field = (detail.groupby("field")["correct"]
                 .agg(correct="sum", scored="count").reset_index())
    per_field["accuracy_%"] = (per_field["correct"] / per_field["scored"] * 100).round(1)

    # Precision / recall / F1 for the imbalanced binary judgments.
    binary = [m for m in (binary_metrics(detail, f)
                          for f in ("material_right", "onboarding_distinct")) if m]

    overall = round(detail["correct"].mean() * 100, 1)

    # One flat CSV: per-contract detail, with the field-level metrics merged on as
    # extra columns (accuracy for every field; recall/F1 for the binary judgments).
    acc_map = dict(zip(per_field["field"], per_field["accuracy_%"]))
    recall_map = {b["field"]: b["recall"] for b in binary}
    f1_map = {b["field"]: b["f1"] for b in binary}
    detail["field_accuracy_%"] = detail["field"].map(acc_map)
    detail["field_recall"] = detail["field"].map(recall_map)
    detail["field_f1"] = detail["field"].map(f1_map)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(SUMMARY_PATH, index=False)

    pd.set_option("display.width", 0)
    print("\n=== Agent 3 per-field accuracy ===")
    print(per_field.to_string(index=False))
    if binary:
        print("\n=== Binary judgments (positive class = TRUE) ===")
        print(pd.DataFrame(binary).to_string(index=False))
    print(f"\nOverall accuracy: {overall}%  ({int(detail['correct'].sum())}/{len(detail)})")
    print(f"Report: {SUMMARY_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true",
                    help="run Agent 3 over the contracts (calls the LLM) and save outputs")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N contracts")
    args = ap.parse_args()

    if args.run:
        outputs = generate_outputs(limit=args.limit)
    else:
        if not OUTPUTS_PATH.exists():
            print(f"No saved outputs at {OUTPUTS_PATH}. Run with --run first.")
            return
        outputs = json.loads(OUTPUTS_PATH.read_text(encoding="utf-8"))

    evaluate(outputs)


if __name__ == "__main__":
    main()