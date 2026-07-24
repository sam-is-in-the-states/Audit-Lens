"""
Evaluate Agent 5 (Evaluation Agent) against Agent 3 and Agent 4 outputs.

Two modes:
  py evaluate_agent5.py --run          # run Agent 5 over all contracts (full pipeline),
                                       # save outputs, then evaluate.
  py evaluate_agent5.py                 # evaluate the saved outputs only (no LLM).
  py evaluate_agent5.py --run --limit 5 # run only the first 5 (quick check).

The generated Agent 5 outputs are saved to evaluation/results/evaluation_outputs.json.
Summary report saved to evaluation/results/evaluation_summary.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from app.pipeline.run_pipeline import EndToEndPipeline

EXTRACTED_PATH = ROOT / "app" / "llm" / "extracted_contracts.json"
TREATMENT_OUTPUTS_PATH = HERE / "results" / "treatment_outputs.json"
AUDIT_OUTPUTS_PATH = HERE / "results" / "audit_agent_results.json"
EVALUATION_OUTPUTS_PATH = HERE / "results" / "evaluation_outputs.json"
EVALUATION_SUMMARY_PATH = HERE / "results" / "evaluation_summary.csv"

CONTRACT_IDS = [f"C{i:02d}" for i in range(1, 51)]


def generate_outputs(limit=None):
    """Run full pipeline and save Agent 5 outputs."""
    pipeline = EndToEndPipeline()
    extracted = json.loads(EXTRACTED_PATH.read_text(encoding="utf-8"))

    outputs = []
    for i, record in enumerate(extracted):
        if limit is not None and i >= limit:
            break

        # Get the contract path from extracted data
        # support multiple formats: top-level `source_file`, or nested
        # `validated_output.source_file` / `candidate_output.source_file`,
        # or fallback to `file_name`.
        contract_path = (
            record.get("source_file")
            or (record.get("validated_output") or {}).get("source_file")
            or (record.get("candidate_output") or {}).get("source_file")
            or record.get("file_name")
        )

        if not contract_path:
            print(f"  Skipping record {i}: no source_file or file_name")
            continue

        # If the value is just a filename (e.g., C01.pdf), try to locate a
        # local text copy under evaluation/contract_txt by replacing .pdf -> .txt
        # or using the same name if present.
        from pathlib import Path

        candidate_path = Path(contract_path)
        resolved_path = None
        # If an absolute/relative path exists as-is, use it
        if candidate_path.exists():
            resolved_path = str(candidate_path)
        else:
            # look for file in evaluation/contract_txt
            local_dir = HERE / "contract_txt"
            try_paths = [local_dir / candidate_path.name]
            if candidate_path.suffix.lower() == ".pdf":
                try_paths.append(local_dir / (candidate_path.stem + ".txt"))
                # also support files named like C41_GranitePeak.pdf -> C41.txt
                short_stem = candidate_path.stem.split("_")[0]
                try_paths.append(local_dir / (short_stem + ".txt"))

            for p in try_paths:
                if p.exists():
                    resolved_path = str(p)
                    break

        if not resolved_path:
            print(f"  Skipping record {i}: contract file not found ({contract_path})")
            continue

        try:
            # Run full pipeline using the resolved path
            result = pipeline.run_contract(resolved_path)

            # Extract Agent 5 output
            agent5_output = result.get("agent5_output")
            if agent5_output:
                outputs.append({
                    "contract_id": result.get("source_file"),
                    "confidence": agent5_output.get("overall_confidence"),
                    "issues_count": len(agent5_output.get("issues", [])),
                    "memo_preview": agent5_output.get("memo", "")[:200],
                    "full_output": agent5_output,
                })
                print(f"  ✓ Evaluated {result.get('source_file')} - Confidence: {agent5_output.get('overall_confidence')}%")
        except Exception as e:
            print(f"  ✗ Error processing {contract_path}: {str(e)}")

    EVALUATION_OUTPUTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVALUATION_OUTPUTS_PATH.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(f"\nSaved {len(outputs)} Agent 5 evaluations → {EVALUATION_OUTPUTS_PATH}")
    return outputs


def evaluate_outputs():
    """Evaluate saved Agent 5 outputs and generate summary."""
    if not EVALUATION_OUTPUTS_PATH.exists():
        print(f"Error: {EVALUATION_OUTPUTS_PATH} not found. Run with --run first.")
        return

    outputs = json.loads(EVALUATION_OUTPUTS_PATH.read_text(encoding="utf-8"))

    summary_rows = []
    for output in outputs:
        full_output = output.get("full_output", {})
        trace = full_output.get("trace", {})
        stats = trace.get("summary_statistics", {})

        row = {
            "contract_id": output.get("contract_id"),
            "overall_confidence": output.get("confidence"),
            "issues_count": output.get("issues_count"),
            "total_kb3_evaluated": stats.get("total_kb3_items_evaluated", 0),
            "kb3_triggered": stats.get("items_triggered_by_agent4", 0),
            "pass_count": stats.get("results", {}).get("PASS", 0),
            "fail_count": stats.get("results", {}).get("FAIL", 0),
            "undetermined_acceptable": stats.get("results", {}).get("UNDETERMINED_ACCEPTABLE", 0),
            "undetermined_problematic": stats.get("results", {}).get("UNDETERMINED_PROBLEMATIC", 0),
        }
        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)

    # Save to CSV
    EVALUATION_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(EVALUATION_SUMMARY_PATH, index=False)

    print(f"\n{'='*80}")
    print("AGENT 5 EVALUATION SUMMARY")
    print(f"{'='*80}\n")
    print(df.to_string(index=False))

    # Calculate aggregate metrics
    avg_confidence = df["overall_confidence"].mean()
    total_issues = df["issues_count"].sum()
    total_pass = df["pass_count"].sum()
    total_fail = df["fail_count"].sum()

    print(f"\n{'='*80}")
    print("AGGREGATE METRICS")
    print(f"{'='*80}")
    print(f"Average Confidence Score:      {avg_confidence:.1f}%")
    print(f"Total Issues Found:            {total_issues}")
    print(f"Total KB3 Items PASS:          {total_pass}")
    print(f"Total KB3 Items FAIL:          {total_fail}")
    print(f"Contracts with Issues:         {len(df[df['issues_count'] > 0])}/{len(df)}")
    print(f"\nSummary saved to: {EVALUATION_SUMMARY_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Run and evaluate Agent 5 (Evaluation Agent) on contract pipeline."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run full pipeline including Agent 5 (calls LLM if needed)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N contracts (for quick testing)",
    )
    args = parser.parse_args()

    if args.run:
        print("Running full pipeline with Agent 5...")
        generate_outputs(limit=args.limit)

    print("\nEvaluating saved outputs...")
    evaluate_outputs()


if __name__ == "__main__":
    main()
