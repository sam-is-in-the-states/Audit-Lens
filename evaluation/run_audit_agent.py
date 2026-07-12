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
import re
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TXT_DIR = PROJECT_ROOT / "evaluation" / "contract_txt"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

from app.llm.audit_agent import AuditAgent
from evaluation.evaluate_agent2 import (
    load_ground_truth,
    parse_kb_ids,
    calculate_metrics,
)


def run_all() -> list[dict[str, Any]]:
    agent = AuditAgent()

    ground_truth = load_ground_truth()

    results = []

    txt_paths = sorted(TXT_DIR.glob("C*.txt"))

    for i, path in enumerate(txt_paths, start=1):
        txt = path.read_text(encoding="utf-8")
        out = agent.run_on_raw_text(txt, source_file=path.name)

        # Determine expected KB3 ids from Ground Truth by contract id
        # Extract contract id from filename (C01 etc.)
        match = re.match(r"(C\d+)", path.stem, flags=re.IGNORECASE)
        contract_id = match.group(1).upper() if match else None

        if contract_id and contract_id in ground_truth["contract_id"].values:
            gt_row = ground_truth[ground_truth["contract_id"] == contract_id].iloc[0]
            expected_kb3 = parse_kb_ids(gt_row.get("KB3 Chunks"))
        else:
            expected_kb3 = set()

        retrieved_kb3 = set(out.get("kb3_selected_ids", []))

        metrics = calculate_metrics(expected_kb3, retrieved_kb3)

        result = {
            "source_file": path.name,
            "contract_id": contract_id,
            "expected_kb3_ids": sorted(expected_kb3),
            "retrieved_kb3_ids": sorted(retrieved_kb3),
            "matched_kb3_ids": metrics["matched_kb_ids"],
            "missing_kb3_ids": metrics["missing_kb_ids"],
            "unexpected_kb3_ids": metrics["unexpected_kb_ids"],
            "recall": metrics["recall_at_k"],
            "precision": metrics["precision_at_k"],
            "f1": metrics["f1_at_k"],
        }

        results.append(result)

        # Log progress after processing this contract so users can track progress
        print(
            f"Processed {i}/{len(txt_paths)}: {path.name} | "
            f"expected={len(expected_kb3)} retrieved={len(retrieved_kb3)} "
            f"matched={len(metrics['matched_kb_ids'])} "
            f"recall={metrics['recall_at_k']:.3f} precision={metrics['precision_at_k']:.3f}"
        )

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

    print(f"Wrote {json_path} and {csv_path}")
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    run_all()
