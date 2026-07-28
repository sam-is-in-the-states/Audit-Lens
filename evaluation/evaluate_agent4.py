"""Run the AuditAgent against the plain-text contracts and evaluate KB3 picks.

This script loads each contract text in `contract_txt/`, runs the
`AuditAgent` (extraction + retrieval + optional LLM refinement) and compares
the returned `KB3` ids against the Ground Truth `KB3 Chunks` column in
`GroundTruth.xlsx`.

Outputs:
  - evaluation/results/audit_agent_results.json
  - evaluation/results/audit_agent_summary.csv
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))          # so `app` / `evaluation` import when run as a script
TXT_DIR = PROJECT_ROOT / "evaluation" / "contract_txt"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Agent 4 audits the REAL upstream outputs (same data the end-to-end pipeline feeds it),
# read from the cached agent outputs instead of rebuilding a minimal Agent 1 and re-running
# Agent 2/3 from raw text. Iterating Agent 4 then does not disturb or degrade the upstream.
EXTRACTED_PATH = PROJECT_ROOT / "app" / "llm" / "extracted_contracts.json"          # Agent 1
AGENT2_CACHE_PATH = RESULTS_DIR / "agent2_output_cache.json"                         # Agent 2
TREATMENT_PATH = RESULTS_DIR / "treatment_outputs.json"                              # Agent 3

from app.llm.audit_agent import AuditAgent
from evaluation.evaluate_agent2 import (
    load_ground_truth,
    parse_kb_ids,
    calculate_metrics,
)


def _cid(name: str) -> str:
    return (name or "").split("_")[0].upper()


def _load_caches():
    """Load the cached upstream agent outputs, indexed by contract id."""
    extracted = json.loads(EXTRACTED_PATH.read_text(encoding="utf-8"))
    agent2_cache = json.loads(AGENT2_CACHE_PATH.read_text(encoding="utf-8"))
    treatment = json.loads(TREATMENT_PATH.read_text(encoding="utf-8"))
    a1_by = {_cid(rec.get("file_name", "")): rec for rec in extracted}
    a2_by = {_cid(k): v for k, v in agent2_cache.items()}
    a3_by = {_cid(o.get("source_file", "")): o for o in treatment}
    return a1_by, a2_by, a3_by


def run_one_contract(contract_id, *, agent, ground_truth, a1_by, a2_by, a3_by):
    """Audit one contract from the cached upstream outputs and score its KB3 picks."""
    rec = a1_by.get(contract_id) or {}
    facts = rec.get("validated_output") or rec.get("candidate_output") or {}
    raw_text = rec.get("full_text", "") or ""
    source_file = rec.get("file_name") or f"{contract_id}.txt"
    facts = {**facts, "contract_id": contract_id, "source_file": source_file, "full_text": raw_text}
    agent2_output = a2_by.get(contract_id) or {}
    agent3_output = a3_by.get(contract_id) or {}
    if not agent3_output:
        print(f"  skip {contract_id}: missing agent3 cache")
        return None

    out = agent.run(facts, agent2_output, agent3_output, raw_text=raw_text)

    if contract_id in ground_truth["contract_id"].values:
        gt_row = ground_truth[ground_truth["contract_id"] == contract_id].iloc[0]
        expected_kb3 = parse_kb_ids(gt_row.get("KB3 Chunks"))
    else:
        expected_kb3 = set()
    retrieved_kb3 = set(out.get("predicted_kb3_ids", []))
    metrics = calculate_metrics(expected_kb3, retrieved_kb3)

    print(f"  {contract_id}: predicted={sorted(retrieved_kb3)} "
          f"recall={metrics['recall']:.2f} prec={metrics['precision']:.2f}")
    return {
        "source_file": source_file,
        "contract_id": contract_id,
        "expected_kb3_ids": sorted(expected_kb3),
        "retrieved_kb3_ids": sorted(retrieved_kb3),
        "matched_kb3_ids": metrics["matched"],
        "missing_kb3_ids": metrics["missing"],
        "unexpected_kb3_ids": metrics["unexpected"],
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "f1": metrics["f1"],
        "audit_findings": out.get("audit_findings", []),
        "applicable_kb3_ids": out.get("applicable_kb3_ids", []),
        "consistency_issues": out.get("consistency_issues", []),
        "revision_requests": out.get("revision_requests", []),
        "kb3_candidate_ids": out.get("kb3_candidate_ids", []),
    }


def run_all(limit: int | None = None) -> list[dict[str, Any]]:
    agent = AuditAgent()
    ground_truth = load_ground_truth()
    a1_by, a2_by, a3_by = _load_caches()

    contract_ids = sorted(a3_by)
    if limit is not None:
        contract_ids = contract_ids[:limit]

    # Deterministic rule layer - no network calls, so run serially.
    results = [r for r in (run_one_contract(cid, agent=agent, ground_truth=ground_truth,
                                            a1_by=a1_by, a2_by=a2_by, a3_by=a3_by)
                           for cid in contract_ids) if r is not None]

    # Save JSON
    json_path = RESULTS_DIR / "audit_agent_results.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary CSV
    rows = []
    for r in results:
        rows.append(
            {
                "source_file": r["source_file"],
                "contract_id": r["contract_id"],
                "expected_kb3_ids": "; ".join(r["expected_kb3_ids"]),
                "retrieved_kb3_ids": "; ".join(r["retrieved_kb3_ids"]),
                "matched_kb3_ids": "; ".join(r["matched_kb3_ids"]),
                "missing_kb3_ids": "; ".join(r["missing_kb3_ids"]),
                "unexpected_kb3_ids": "; ".join(r["unexpected_kb3_ids"]),
                "recall": r["recall"],
                "precision": r["precision"],
                "f1": r["f1"],
            }
        )

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "audit_agent_summary.csv"
    df.to_csv(csv_path, index=False)

    # Write per-contract reasoning CSV: mapping KB3_ID -> LLM explanation
    reasoning_rows = []
    for r in results:
        findings = r.get("audit_findings", [])
        reasoning = {f.get("kb3_id"): f.get("reason") for f in findings}
        reasoning_rows.append(
            {
                "source_file": r["source_file"],
                "contract_id": r["contract_id"],
                "expected_kb3_ids": "; ".join(r["expected_kb3_ids"]),
                "retrieved_kb3_ids": "; ".join(r["retrieved_kb3_ids"]),
                "reasoning_json": json.dumps(reasoning, ensure_ascii=False),
            }
        )

    import csv

    reasoning_path = RESULTS_DIR / "audit_agent_reasoning.csv"
    with reasoning_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["source_file", "contract_id", "expected_kb3_ids", "retrieved_kb3_ids", "reasoning_json"])
        writer.writeheader()
        for row in reasoning_rows:
            writer.writerow(row)

    print(f"Wrote reasoning CSV -> {reasoning_path}")

    print(f"Wrote {json_path} and {csv_path}")
    _report(results)
    return results


def _report(results: list[dict[str, Any]]) -> None:
    """Metrics that fit a deterministic KB3 review layer - and why each is here.

    The task is multi-label detection over an 18-item checklist against sparse ground truth
    (~1.4 material KB3 items per contract). Autonomous-loop metrics (fallback %, step cap,
    tool-call count, materiality-basis mix) measure an LLM agent's control flow, not review
    quality, so they are not reported for the rule layer.
    """
    if not results:
        return
    tp = fp = fn = 0
    ptp: dict[str, int] = defaultdict(int)
    pfp: dict[str, int] = defaultdict(int)
    pfn: dict[str, int] = defaultdict(int)
    for r in results:
        e, p = set(r["expected_kb3_ids"]), set(r["retrieved_kb3_ids"])
        tp += len(e & p); fp += len(p - e); fn += len(e - p)
        for k in e & p: ptp[k] += 1
        for k in p - e: pfp[k] += 1
        for k in e - p: pfn[k] += 1
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0

    # (1) Detection quality, micro-pooled over every (contract, KB3-item) decision. Micro (not
    # per-contract mean) so each risk weighs equally regardless of how many a contract carries.
    # Precision AND recall are reported, not just F1: for an audit layer the two errors differ
    # in cost - a false negative is a missed risk (potential misstatement), a false positive only
    # costs reviewer time - so the operating point matters. This layer leans recall by design.
    print("\n=== AGENT 4 - KB3 review detection (micro, pooled over contract x item) ===")
    print(f"  precision={P:.3f}  recall={R:.3f}  f1={F:.3f}   (TP={tp} FP={fp} FN={fn})")

    # (2) Per-item, because an 18-item checklist's aggregate hides which triggers over/under-fire;
    # this is the diagnostic that says which rules to trust.
    print("\n=== per KB3 item (items with any error) ===")
    for k in sorted(set(list(ptp) + list(pfp) + list(pfn)), key=lambda x: int(x.split("_")[1])):
        if pfp[k] + pfn[k]:
            pp = ptp[k] / (ptp[k] + pfp[k]) if ptp[k] + pfp[k] else 0.0
            rr = ptp[k] / (ptp[k] + pfn[k]) if ptp[k] + pfn[k] else 0.0
            print(f"  {k:8} TP={ptp[k]} FP={pfp[k]} FN={pfn[k]}  prec={pp:.2f} rec={rr:.2f}")

    # (3) Contract-level exact match, because a reviewer consumes the whole KB3 set for a contract
    # at once; getting every item right on a contract is the operationally meaningful bar, and it
    # is stricter than per-item accuracy.
    exact = sum(1 for r in results if set(r["expected_kb3_ids"]) == set(r["retrieved_kb3_ids"]))
    le1 = sum(1 for r in results
              if len(set(r["expected_kb3_ids"]) ^ set(r["retrieved_kb3_ids"])) <= 1)
    print(f"\n=== contract-level ===\n  exact-set match: {exact}/{len(results)}   "
          f"within-1-item: {le1}/{len(results)}")

    # (4) Cross-layer review value: Agent 4 is the only stage holding all upstream layers, so it
    # flags contradictions Agent 3's own output hides (e.g. C34 nonrefundable material right, C37
    # residual pricing). Not a KB3-detection metric, but the distinctive value of this stage.
    revs = [(r["contract_id"], x) for r in results for x in r.get("revision_requests", [])]
    issues = sum(len(r.get("consistency_issues", [])) for r in results)
    print(f"\n=== cross-layer review (Agent 4's distinctive value) ===")
    print(f"  revision requests: {len(revs)}   consistency issues flagged: {issues}")
    for cid, x in revs:
        print(f"    {cid}: {x.get('field')} - {x.get('reason')}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only run the first N contracts (smoke test)")
    args = ap.parse_args()

    run_all(limit=args.limit)
