"""
Load GroundTruth.xlsx into a structured dict keyed by contract_id (e.g. 'C01').

Column categories (from spreadsheet colour coding):
  - Contract characterisation fields: facts and interpretive conclusions about the contract
    structure (some require applying ASC 606 criteria to contract language, e.g.
    onboarding_distinct, material_right, discount_type).
  - Revenue treatment fields: ASC 606 judgments that flow from the characterisation
    fields plus RAG-retrieved guidance (num_POs, recognition_method, opening_deferred, …).
  - Notes fields: reviewer annotations not needed for model evaluation (KB chunks, notes).
"""

import re
from pathlib import Path

import pandas as pd

GT_PATH = Path(__file__).parent / "GroundTruth.xlsx"

# Contract characterisation fields
# These describe WHAT the contract contains and HOW it is structured.
# Some are direct facts (term_months, fee amounts, billing timing).
# Others require applying ASC 606 criteria to contract language
# (onboarding_distinct → KB1_3/KB1_4; material_right → KB1_15; discount_type → KB1_14).
# Agent 1's job is to produce all of these so Agent 2 can reason about treatment.
CHARACTERISATION_FIELDS = [
    "customer_name",       # direct fact
    "type",                # contract category label (for evaluation grouping)
    "variant_note",        # human description of contract variant
    "difficulty",          # E / M / H difficulty label
    "term_months",         # direct fact – subscription length in months
    "start_date",          # direct fact – effective date
    "billing",             # direct fact – payment timing/frequency pattern
    "annual_subscription", # direct fact – fixed subscription fee (annual equivalent)
    "onboarding_fee",      # direct fact – one-time implementation/onboarding fee amount
    "onboarding_distinct", # interpretive – TRUE/FALSE/UNDETERMINED/n/a
                           #   requires KB1_3/KB1_4 + KB2_2/KB2_3 applied to contract language
    "usage_model",         # direct fact – type of usage pricing structure
    "base_fee",            # direct fact – base/minimum fee in usage arrangements
    "usage_rate",          # direct fact – per-unit rate
    "discount_amount",     # direct fact – discount vs sum of list prices
    "discount_type",       # interpretive – how discount is classified under ASC 606
                           #   requires KB1_14/KB2_8 (pro-rata vs directed vs residual)
                           #   or KB1_15/KB2_10 (material right / renewal option)
    "material_right",      # interpretive – TRUE/FALSE/UNDETERMINED
                           #   requires KB1_15/KB2_10 applied to renewal/option terms
]

# Revenue treatment fields
# ASC 606 accounting outcomes derived by Agent 2 using characterisation fields + RAG.
TREATMENT_FIELDS = [
    "num_POs",                  # KB1_3/KB1_4/KB1_15 + KB2_1/KB2_2/KB2_3
    "po_list",                  # same as above
    "transaction_price",        # KB1_8/KB1_9/KB1_12/KB1_14 + KB2_8/KB2_9
    "recognition_method",       # KB1_5/KB1_6/KB1_7/KB1_10/KB1_11 + KB2_1/KB2_4–KB2_7
    "recognition_period_months",# KB2_13 (proration for mid-period starts)
    "monthly_revenue",          # derived from transaction_price / recognition_period_months
    "opening_deferred",         # KB1_17/KB2_12 – contract liability on day 1
    "expected_risk_flags",      # risk signals for auditor review
    "review_questions",         # open questions for accounting team
    "notes",                    # reviewer annotations
]

# Fields Agent 2 needs from characterisation in order to retrieve the right KB chunks
# and produce each treatment field.
TREATMENT_DEPENDENCIES: dict[str, list[str]] = {
    "num_POs": [
        "onboarding_distinct",  # drives PO count for implementation (KB2_2/KB2_3)
        "material_right",       # renewal option as separate PO (KB1_15/KB2_10)
        "usage_model",          # royalty model may split POs (KB1_11)
        "discount_type",        # residual/directed affects PO structure
    ],
    "po_list": [
        "onboarding_distinct", "material_right", "usage_model", "discount_type",
    ],
    "transaction_price": [
        "annual_subscription",  # fixed fee component
        "onboarding_fee",       # one-time fee component (if distinct PO)
        "discount_amount",      # reduces consideration (KB1_14/KB2_8)
        "usage_model",          # variable consideration constraint (KB1_8/KB1_9)
    ],
    "recognition_method": [
        "usage_model",          # determines fixed vs variable recognition pattern
        "onboarding_distinct",  # non-distinct onboarding bundled into subscription
        "billing",              # right-to-invoice expedient depends on billing pattern
    ],
    "recognition_period_months": [
        "term_months",          # base period
        "start_date",           # proration for mid-month start (KB2_13)
    ],
    "monthly_revenue": [
        "annual_subscription", "onboarding_fee", "discount_amount",
        "term_months", "start_date", "usage_model", "onboarding_distinct",
    ],
    "opening_deferred": [
        "billing",              # upfront billing creates deferred; arrears creates contract asset
        "annual_subscription",  # amount billed upfront
        "onboarding_fee",       # if non-distinct, included in deferred balance
        "start_date",           # affects proration of first-period balance (KB2_13)
    ],
    "expected_risk_flags": [
        "material_right",       # undisclosed material right is a risk
        "onboarding_distinct",  # UNDETERMINED onboarding treatment is a risk
        "discount_type",        # residual / renewal-option discount needs scrutiny
        "usage_model",          # variable consideration constraint risk (KB1_9)
    ],
}

# Numeric fields
_NUMERIC_FIELDS = {
    "term_months", "annual_subscription", "onboarding_fee", "base_fee",
    "usage_rate", "discount_amount", "transaction_price",
    "recognition_period_months", "monthly_revenue", "opening_deferred", "num_POs",
}


def load_ground_truth(path: str | Path = GT_PATH) -> dict[str, dict]:
    """
    Returns dict keyed by contract_id ('C01' … 'C50'), each value:
        {
          "characterisation": { field: value, … },
          "treatment":        { field: value, … },
        }
    """
    df = pd.read_excel(path, skiprows=3, header=0)
    df = df.rename(columns={"Customer Name": "customer_name"})
    df = df[
        df["contract_id"].notna()
        & df["contract_id"].astype(str).str.match(r"^C\d+$")
    ]
    df = df.set_index("contract_id")

    result: dict[str, dict] = {}
    for cid, row in df.iterrows():
        result[str(cid)] = {
            "characterisation": {
                f: _clean(f, row.get(f)) for f in CHARACTERISATION_FIELDS
            },
            "treatment": {
                f: _clean(f, row.get(f)) for f in TREATMENT_FIELDS
            },
        }
    return result


def _clean(field: str, val):
    """Normalise a GT cell value for comparison."""
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if val == "":
        return None

    if field in _NUMERIC_FIELDS:
        if isinstance(val, (int, float)):
            return float(val)
        s = re.sub(r"[$,\s]", "", str(val))
        try:
            return float(s)
        except ValueError:
            return None

    # material_right: Excel boolean TRUE/FALSE, or string "UNDETERMINED"
    if field == "material_right":
        if isinstance(val, bool):
            return val
        s = str(val).strip().upper()
        if s == "TRUE":
            return True
        if s == "FALSE":
            return False
        return s  # "UNDETERMINED"

    # po_list: semicolon-separated string → list
    if field == "po_list":
        if isinstance(val, str):
            return sorted(x.strip() for x in val.split(";") if x.strip())
        return val

    # start_date: normalise to ISO string
    if field == "start_date":
        if hasattr(val, "strftime"):
            return val.strftime("%Y-%m-%d")
        return str(val).strip()

    return str(val).strip()


if __name__ == "__main__":
    gt = load_ground_truth()
    print(f"Loaded {len(gt)} contracts\n")
    for cid in ["C01", "C11", "C31"]:
        print(f"=== {cid} ===")
        print("Characterisation:", gt[cid]["characterisation"])
        print("Treatment:       ", gt[cid]["treatment"])
        print()