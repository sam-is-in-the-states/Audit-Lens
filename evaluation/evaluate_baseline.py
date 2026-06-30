"""
Evaluate the baseline (single-prompt) extraction + treatment against GroundTruth.xlsx
across all 50 contracts.

The baseline produces both contract facts (characterisation) and ASC 606 judgments
(treatment) in one prompt, so both groups are scored separately by accuracy.

run_all()  reads each contract from contract_txt/, runs the baseline prompt, and saves
           the raw model outputs to results/baseline_outputs.json (calls the LLM, slow).
evaluate() loads those outputs, compares each field against ground truth with simple
           normalisation, and prints accuracy per group and per field as tables.

Run from the evaluation/ folder:
    py evaluate_baseline.py              # run the baseline on 50 contracts, then evaluate
    py evaluate_baseline.py --eval-only  # re-evaluate the saved outputs without the LLM
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.llm.client import get_llm_response
from app.prompts import baseline

from load_ground_truth import (
    load_ground_truth,
    CHARACTERISATION_FIELDS,
    TREATMENT_FIELDS,
    _NUMERIC_FIELDS,
)

HERE = Path(__file__).parent
TXT_DIR = HERE / "contract_txt"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
OUTPUTS_JSON = RESULTS / "baseline_outputs.json"

CONTRACT_IDS = [f"C{i:02d}" for i in range(1, 51)]

# Baseline does not emit type/variant_note/difficulty or risk_flags/review_questions/notes.
SCORE_CHAR = [f for f in CHARACTERISATION_FIELDS if f not in {"type", "variant_note", "difficulty"}]
SCORE_TREAT = [f for f in TREATMENT_FIELDS if f not in {"expected_risk_flags", "review_questions", "notes"}]

# Numeric tolerance (monthly_revenue etc. carry decimals like 1833.33).
NUM_ABS_TOL = 1.0
NUM_REL_TOL = 0.005

# Strings that all mean "not applicable / none / empty" and should compare as equal.
_EMPTY_TOKENS = {"", "n/a", "na", "none", "null", "nil"}


def run_one(contract_id):
    """Run the baseline prompt on one contract's text (reuses A's prompt + LLM client)."""
    contract_text = (TXT_DIR / f"{contract_id}.txt").read_text(encoding="utf-8")
    messages = [
        {"role": "system", "content": baseline.SYSTEM_PROMPT},
        {"role": "user", "content": baseline.USER_PROMPT.format(contract_text=contract_text)},
    ]
    return json.loads(get_llm_response(messages))


def run_all():
    """Run the baseline on all 50 contracts and save outputs to baseline_outputs.json."""
    outputs = {}
    for cid in CONTRACT_IDS:
        try:
            outputs[cid] = run_one(cid)
            print(f"  {cid} ok")
        except Exception as e:
            outputs[cid] = {"_error": str(e)}
            print(f"  {cid} failed: {e}")
    OUTPUTS_JSON.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(f"Saved outputs to {OUTPUTS_JSON}")
    return outputs


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


def _norm_str(v):
    return str(v).strip().lower() if v is not None else ""


def _is_empty(v):
    """True if the value means 'none / not applicable / empty'."""
    if v is None:
        return True
    if isinstance(v, str) and v.strip().lower() in _EMPTY_TOKENS:
        return True
    return False


def values_match(field, predicted, truth):
    """Return True if the predicted value matches the ground-truth value for this field."""
    # Treat n/a / none / null / "" / None as the same "empty" value on both sides.
    if _is_empty(predicted) and _is_empty(truth):
        return True
    if _is_empty(predicted) != _is_empty(truth):
        return False

    # num_POs is always a small integer (1/2/3); tolerance-based matching would give
    # credit for being off by 1 PO, so use exact integer comparison instead.
    if field == "num_POs":
        p, t = _to_float(predicted), _to_float(truth)
        if p is None or t is None:
            return _norm_str(predicted) == _norm_str(truth)
        return round(p) == round(t)

    if field in _NUMERIC_FIELDS:
        p, t = _to_float(predicted), _to_float(truth)
        if p is None or t is None:
            return _norm_str(predicted) == _norm_str(truth)
        return abs(p - t) <= NUM_ABS_TOL or (t != 0 and abs(p - t) / abs(t) <= NUM_REL_TOL)

    if field == "po_list":
        def as_list(x):
            if isinstance(x, list):
                return sorted(_norm_str(i) for i in x)
            if isinstance(x, str):
                return sorted(_norm_str(i) for i in x.replace(";", ",").split(",") if i.strip())
            return []
        return as_list(predicted) == as_list(truth)

    if field in ("material_right", "onboarding_distinct"):
        def norm_bool(x):
            if isinstance(x, bool):
                return x
            s = str(x).strip().upper()
            if s == "TRUE":
                return True
            if s == "FALSE":
                return False
            return s
        return norm_bool(predicted) == norm_bool(truth)

    return _norm_str(predicted) == _norm_str(truth)


def evaluate():
    """Compare saved baseline outputs against ground truth and print tables."""
    gt = load_ground_truth()
    if not OUTPUTS_JSON.exists():
        raise FileNotFoundError(
            f"{OUTPUTS_JSON} not found. Run without --eval-only first to generate outputs."
        )
    outputs = json.loads(OUTPUTS_JSON.read_text(encoding="utf-8"))

    fields = [("characterisation", f) for f in SCORE_CHAR] + [("treatment", f) for f in SCORE_TREAT]
    rows = []          # one row per (contract, field): correct or not
    n_failed = 0

    for cid in CONTRACT_IDS:
        out = outputs.get(cid, {})
        if "_error" in out:
            n_failed += 1
            continue
        char = gt[cid]["characterisation"]
        truth_all = {**char, **gt[cid]["treatment"]}
        for group, field in fields:
            t = truth_all.get(field)
            p = out.get(field)
            rows.append({
                "contract": cid,
                "type": char.get("type"),
                "difficulty": char.get("difficulty"),
                "group": group,
                "field": field,
                "predicted": p,
                "ground_truth": t,
                "correct": values_match(field, p, t),
            })

    df = pd.DataFrame(rows)

    def acc_by(keys):
        t = (df.groupby(keys)["correct"]
               .agg(correct="sum", total="count")
               .reset_index())
        t["accuracy_%"] = (t["correct"] / t["total"] * 100).round(1)
        return t

    field_order = {f: i for i, (_, f) in enumerate(fields)}
    diff_order = {"E": 0, "M": 1, "H": 2}
    type_order = {"simple_sub": 0, "sub_onboarding": 1, "sub_usage": 2,
                  "sub_discount": 3, "ambiguous": 4, "combo_sub_onboard_usage": 5}

    pd.set_option("display.width", 0)

    if n_failed:
        print(f"({n_failed} contract(s) returned invalid JSON and were skipped)")

    # 1. Overall, by group (extraction vs judgment)
    print("\n=== Accuracy by group ===")
    print(acc_by("group").to_string(index=False))

    # 2. By difficulty — shows the baseline degrading as contracts get harder
    print("\n=== Accuracy by difficulty (E/M/H) ===")
    t = acc_by(["difficulty", "group"])
    t = t.sort_values(["difficulty", "group"], key=lambda s: s.map(diff_order) if s.name == "difficulty" else s)
    print(t.to_string(index=False))

    # 3. By contract type — shows which contract structures the baseline handles or fails
    print("\n=== Accuracy by contract type ===")
    t = acc_by(["type", "group"])
    t = t.sort_values(["type", "group"], key=lambda s: s.map(type_order) if s.name == "type" else s)
    print(t.to_string(index=False))

    # 4. Per-field accuracy, weakest first — the baseline's specific blind spots
    print("\n=== Accuracy by field (weakest first) ===")
    t = acc_by(["group", "field"]).sort_values("accuracy_%")
    print(t.to_string(index=False))

    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true",
                    help="skip the LLM and re-evaluate the saved baseline_outputs.json")
    args = ap.parse_args()
    if not args.eval_only:
        run_all()
    evaluate()


if __name__ == "__main__":
    main()