"""Reasoning-quality counter for Agent 3's saved outputs (no LLM, no control-flow change).

Program-level validation of the kb_basis citations the LLM produced, plus a separate recall
check on the deterministic onboarding rule's own citations. Reports counts only; it never
changes a conclusion.

    py validate_reasoning.py
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
outs = json.loads((ROOT / "evaluation/results/treatment_outputs.json").read_text(encoding="utf-8"))
cache = json.loads((ROOT / "evaluation/results/agent2_output_cache.json").read_text(encoding="utf-8"))
cache_by = {k.split("_")[0].upper(): v for k, v in cache.items()}

# The LLM-generated judgments (onboarding_distinct itself is set by the rule and is checked
# separately; onboarding_distinct_llm keeps the model's original onboarding answer).
LLM_FIELDS = ["usage_model", "discount_type", "material_right", "contract_valid",
              "additional_distinct_services", "transaction_price", "onboarding_distinct_llm"]
VALID_ID = re.compile(r"^KB[123]_\d+$")          # KB1/KB2/KB3 exist; a bare "KB2" or "KB4" is malformed
PLACEHOLDER = re.compile(r"KB[123]_\.\.\.")
RULE_IDS = {"KB2_1", "KB2_2", "KB2_3"}


def cid(o):
    return (o.get("source_file") or "").split("_")[0].upper()


def retrieved_ids(a2):
    if not isinstance(a2, dict):
        return set()
    return {c.get("doc_id") for c in (a2.get("retrieved_policy_chunks") or []) if c.get("doc_id")}


empty_kb = na_kb = placeholder_reasoning = 0
ids_total = na_ids = malformed_ids = hallucinated_ids = valid_ids = 0
rule_cited = rule_retrieved = 0

for o in outs:
    ret = retrieved_ids(cache_by.get(cid(o)))
    ch = o.get("characterization") or {}
    for f in LLM_FIELDS:
        node = ch.get(f)
        if not isinstance(node, dict):
            continue
        kb = node.get("kb_basis")
        if PLACEHOLDER.search(str(node.get("reasoning") or "")):
            placeholder_reasoning += 1
        if not kb:
            empty_kb += 1
            continue
        if any(str(i).strip().lower() == "n/a" for i in kb):
            na_kb += 1
        for i in kb:
            ids_total += 1
            s = str(i).strip()
            if s.lower() == "n/a":
                na_ids += 1
            elif not VALID_ID.match(s):
                malformed_ids += 1
            else:
                valid_ids += 1
                if s not in ret:
                    hallucinated_ids += 1
    # deterministic onboarding rule: is its KB2_1/2/3 citation actually in Agent 2's retrieval?
    for i in ((ch.get("onboarding_distinct") or {}).get("kb_basis") or []):
        if i in RULE_IDS:
            rule_cited += 1
            if i in ret:
                rule_retrieved += 1

print(f"=== kb_basis quality: LLM judgments (50 contracts x {len(LLM_FIELDS)} fields) ===")
print(f"  empty kb_basis:                 {empty_kb} judgments")
print(f"  kb_basis containing 'n/a':      {na_kb} judgments ({na_ids} ids)")
print(f"  malformed ids (e.g. KB2, KB4):  {malformed_ids} ids")
hrate = (100 * hallucinated_ids / valid_ids) if valid_ids else 0
print(f"  hallucinated ids (valid form, not in retrieval): {hallucinated_ids}/{valid_ids} ({hrate:.1f}%)")
print(f"  (leftover 'KBx_...' placeholder in reasoning:    {placeholder_reasoning} judgments)")
print("=== rule / retrieval consistency (exempted from the counts above) ===")
rr = (100 * rule_retrieved / rule_cited) if rule_cited else 0
print(f"  rule cited KB2_1/2/3 on {rule_cited} contracts; Agent 2 retrieved that id on {rule_retrieved} ({rr:.1f}% recall)")
