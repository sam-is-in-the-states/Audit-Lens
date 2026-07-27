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
AGENT2_CACHE_PATH = HERE / "results" / "agent2_output_cache.json"  # cached Agent 2 outputs so Agent 3 can be iterated without re-running Agent 2

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
# scored on ALL 50 contracts: 'none' / 'n/a' are real answers here (correctly saying
# "no discount" / "no usage" / "no onboarding" counts), not rows to skip.
FULL_CATEGORICAL = {"onboarding_distinct", "usage_model", "discount_type"}
ALL_FIELDS = list(CATEGORICAL_FIELDS) + list(NUMERIC_FIELDS)

NUM_ABS_TOL = 1.0
NUM_REL_TOL = 0.005
_EMPTY = {"", "n/a", "na", "none", "null", "nil"}


# generation: run Agent 3 and save the outputs Agent 4 will use 

def generate_outputs(limit=None, refresh_agent2=False):
    from app.llm.treatment_agent import run as run_agent3
    from app.llm.policy_retrieval import PolicyRetrievalAgent

    extracted = json.loads(EXTRACTED_PATH.read_text(encoding="utf-8"))

    # Agent 2's retrieval only depends on Agent 1's facts, so once cached we can iterate on
    # Agent 3 WITHOUT re-running Agent 2. Rebuild the cache with --refresh-agent2 after Agent 1 changes.
    if refresh_agent2 or not AGENT2_CACHE_PATH.exists():
        a2_cache = {}
    else:
        a2_cache = json.loads(AGENT2_CACHE_PATH.read_text(encoding="utf-8"))
        print(f"Using cached Agent 2 outputs ({len(a2_cache)} contracts). Pass --refresh-agent2 to rebuild.")

    agent2 = None  # built lazily - only if some contract is not already cached

    outputs = []
    for i, record in enumerate(extracted):
        if limit is not None and i >= limit:
            break
        facts = record.get("validated_output") or record.get("candidate_output") or {}
        key = record.get("file_name") or str(i)
        if key in a2_cache:
            agent2_output = a2_cache[key]
        else:
            if agent2 is None:
                agent2 = PolicyRetrievalAgent()
            agent2_output = agent2.run(facts)
            a2_cache[key] = agent2_output
        result = run_agent3(record, agent2_output)
        outputs.append(result)
        print(f"  ran {result.get('source_file')}")

    AGENT2_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    AGENT2_CACHE_PATH.write_text(json.dumps(a2_cache, indent=2), encoding="utf-8")

    OUTPUTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUTS_PATH.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(f"Saved {len(outputs)} Agent 3 outputs -> {OUTPUTS_PATH}")
    return outputs


# flatten one Agent 3 output into the fields we compare to ground truth

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


# comparison helpers 

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


_ABSENT = {"", "n/a", "na", "none", "null", "nil", "nan"}


def _norm_cat(v):
    """Categorical normaliser that folds the whole 'nothing here' family (None, '', 'n/a',
    'none', ...) into one token, so 'no onboarding'/'no usage'/'no discount' agree regardless
    of how each side spells it. NOTE: 'FALSE' is NOT absent (it means present-but-not-distinct)."""
    s = _norm(v)
    return "absent" if s in _ABSENT else s


def _canon_method(v):
    """recognition_method is free text in the ground truth ('both over_time_ratable',
    'sub:over_time; onboarding:point_in_time', 'UNDETERMINED', ...). Compare the SET of
    recognition patterns mentioned, ignoring wording and per-component labels, so a
    semantically identical answer is not marked wrong on phrasing alone."""
    s = _norm(v)
    toks = set()
    if "undetermined" in s:
        toks.add("undetermined")
    if "point_in_time" in s or "point in time" in s:
        toks.add("point_in_time")
    if "over_time" in s or "over time" in s or "ratable" in s:
        toks.add("over_time")
    if "usage" in s or "as usage occurs" in s:
        toks.add("usage")
    if "input_method" in s or "input method" in s:
        toks.add("input_method")
    return toks


def _is_empty(v):
    return v is None or (isinstance(v, str) and v.strip().lower() in _EMPTY)


def compare(field, predicted, truth):
    """True / False if scoreable, or None to skip (no ground truth to compare)."""
    if field == "num_POs":
        # exact integer match: being off by one PO is a real error, not "close enough"
        t = _to_float(truth)
        if t is None:
            return None
        p = _to_float(predicted)
        if p is None:
            return False
        return round(p) == round(t)

    if field in NUMERIC_FIELDS:
        t = _to_float(truth)
        if t is None:
            return None                       # GT has no number here -> skip
        p = _to_float(predicted)
        if p is None:
            return False
        return abs(p - t) <= NUM_ABS_TOL or (t != 0 and abs(p - t) / abs(t) <= NUM_REL_TOL)

    if field == "recognition_method":
        if _is_empty(truth):
            return None
        return _canon_method(predicted) == _canon_method(truth)

    # these are scored on all 50 - "none"/"n/a"/None all mean "nothing here" and must agree
    if field in FULL_CATEGORICAL:
        return _norm_cat(predicted) == _norm_cat(truth)

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


# evaluation

def _is_nondefault(field, truth):
    """Is this a case that needs a real judgment (not the trivial majority answer)?
    Plain accuracy is dominated by simple contracts; the non-default subset is where the
    KB / judgment actually matters."""
    if field == "num_POs":
        return _to_float(truth) not in (None, 1.0)
    if field in NUMERIC_FIELDS:
        return True
    if field == "recognition_method":
        return _canon_method(truth) != {"over_time"}
    if field == "material_right":
        return _norm(truth) == "true"
    return _norm_cat(truth) not in ("absent", "false")


def score_report(pred_by_cid, gt, title=""):
    """Per-field accuracy on all 50 AND on the non-default subset. Identical logic for
    baseline and pipeline so the two are strictly comparable."""
    rows = []
    tot_c = tot_n = nd_c = nd_n = 0
    for field in ALL_FIELDS:
        oc = on = hc = hn = 0
        for cid in CONTRACT_IDS:
            pred = pred_by_cid.get(cid)
            if pred is None:
                continue
            truth, _ = field_ground_truth(cid, gt)
            r = compare(field, pred.get(field), truth[field])
            if r is None:
                continue
            oc += 1 if r else 0; on += 1
            tot_c += 1 if r else 0; tot_n += 1
            if _is_nondefault(field, truth[field]):
                hc += 1 if r else 0; hn += 1
                nd_c += 1 if r else 0; nd_n += 1
        rows.append({
            "field": field,
            "acc_%": round(100 * oc / on, 1) if on else None,
            "all": f"{oc}/{on}",
            "nondefault_%": round(100 * hc / hn, 1) if hn else None,
            "nondefault": f"{hc}/{hn}",
        })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 0)
    if title:
        print(f"\n=== {title}: per-field (all 50 + non-default subset) ===")
    print(df.to_string(index=False))
    print(f"OVERALL: {round(100 * tot_c / tot_n, 1)}%  ({tot_c}/{tot_n})   "
          f"NON-DEFAULT: {round(100 * nd_c / nd_n, 1)}%  ({nd_c}/{nd_n})")
    return df


def evaluate(outputs):
    gt = load_ground_truth()
    by_cid = {}
    for out in outputs:
        cid = (out.get("source_file") or "").split("_")[0].upper()
        if cid in CONTRACT_IDS:
            by_cid[cid] = out

    # Column order for the wide per-contract report (matches the ground-truth layout).
    WIDE_ORDER = [
        "onboarding_distinct", "usage_model", "discount_type", "material_right",
        "num_POs", "transaction_price", "recognition_method",
        "recognition_period_months", "monthly_revenue", "opening_deferred",
    ]

    detail_rows = []   # long form, used for the metrics
    wide_rows = []     # one row per contract, predicted vs ground truth side by side
    for cid in CONTRACT_IDS:
        out = by_cid.get(cid)
        if out is None:
            continue
        pred = flatten(out)
        truth, onb_fee = field_ground_truth(cid, gt)

        row = {"contract_id": cid, "type": gt[cid]["characterisation"].get("type")}
        wrong = []
        for field in ALL_FIELDS:
            result = compare(field, pred[field], truth[field])
            if result is not None:
                detail_rows.append({
                    "contract_id": cid,
                    "type": gt[cid]["characterisation"].get("type"),
                    "field": field,
                    "predicted": pred[field],
                    "truth": truth[field],
                    "correct": result,
                })
                if result is False:
                    wrong.append(field)
        # wide row: predicted next to ground truth for every scored field
        for field in WIDE_ORDER:
            row[field] = pred.get(field)
            row[field + "__gt"] = truth.get(field)
        row["n_wrong"] = len(wrong)
        row["wrong_fields"] = ", ".join(wrong)
        wide_rows.append(row)

    detail = pd.DataFrame(detail_rows)

    # Per-field accuracy.
    per_field = (detail.groupby("field")["correct"]
                 .agg(correct="sum", scored="count").reset_index())
    per_field["accuracy_%"] = (per_field["correct"] / per_field["scored"] * 100).round(1)

    # Precision / recall / F1 for the imbalanced binary judgments.
    binary = [m for m in (binary_metrics(detail, f)
                          for f in ("material_right", "onboarding_distinct")) if m]

    overall = round(detail["correct"].mean() * 100, 1)

    # One row per contract: predicted vs ground truth side by side, in GT column order,
    # plus n_wrong / wrong_fields so mismatches are easy to scan.
    wide = pd.DataFrame(wide_rows)
    col_order = ["contract_id", "type", "n_wrong", "wrong_fields"]
    for field in WIDE_ORDER:
        col_order += [field, field + "__gt"]
    wide = wide[[c for c in col_order if c in wide.columns]]

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(SUMMARY_PATH, index=False)

    pd.set_option("display.width", 0)
    print("\n=== Agent 3 per-field accuracy ===")
    print(per_field.to_string(index=False))
    if binary:
        print("\n=== Binary judgments (positive class = TRUE) ===")
        print(pd.DataFrame(binary).to_string(index=False))
    print(f"\nOverall accuracy: {overall}%  ({int(detail['correct'].sum())}/{len(detail)})")
    print(f"Report: {SUMMARY_PATH}")
    # unified report (same function baseline uses) - overall + non-default subset
    score_report({c: flatten(o) for c, o in by_cid.items()}, gt, "AGENT 3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true",
                    help="run Agent 3 over the contracts (calls the LLM) and save outputs")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N contracts")
    ap.add_argument("--refresh-agent2", action="store_true",
                    help="rebuild the cached Agent 2 outputs (do this after Agent 1 changes)")
    args = ap.parse_args()

    if args.run:
        outputs = generate_outputs(limit=args.limit, refresh_agent2=args.refresh_agent2)
    else:
        if not OUTPUTS_PATH.exists():
            print(f"No saved outputs at {OUTPUTS_PATH}. Run with --run first.")
            return
        outputs = json.loads(OUTPUTS_PATH.read_text(encoding="utf-8"))

    evaluate(outputs)


if __name__ == "__main__":
    main()