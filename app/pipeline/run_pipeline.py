"""End-to-end Audit-Lens pipeline: one contract PDF in, a revenue-recognition memo out.

Frontend usage - build the pipeline once, then call run_contract per upload:

    from app.pipeline.run_pipeline import EndToEndPipeline

    pipeline = EndToEndPipeline()                 # construct once (loads the KB / retriever)
    result = pipeline.run_contract(pdf_path)      # pdf_path: a saved .pdf or .txt path

    memo_text = result["memo"]                    # the finished memo, ready to display / download
    status    = result["recognition_status"]      # one-line disposition for a badge
    result["agent5_output"]                       # full structured memo (schedule, JEs, review points)
    result["agent1_output"] ... ["agent4_output"] # per-agent traces, if the UI wants to show them

run_contract never raises on a normal contract; on unreadable input it raises FileNotFoundError /
ValueError, which the caller should surface to the user.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.llm.contract_label_extraction import (
    build_rule_based_candidate,
    extract_pdf_raw_data,
    validate_structured_output,
)
from app.llm.policy_retrieval import PolicyRetrievalAgent
from app.llm.treatment_agent import run as run_agent3
from app.llm.audit_agent import AuditAgent
from app.llm.memo_agent import MemoAgent, DISPOSITION_TEXT


class EndToEndPipeline:
    def __init__(
        self,
        top_k: int = 20,
        chunk_size: int = 500,
        chunk_overlap: int = 75,
        kb_quotas: dict | None = None,
    ):
        self.agent2 = PolicyRetrievalAgent(
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            kb_quotas=kb_quotas,
        )
        self.agent4 = AuditAgent()   # rule-based KB3 review; reads the contract text directly
        self.agent5 = MemoAgent()

    def run_contract(self, contract_path: str) -> dict[str, Any]:
        path = Path(contract_path)
        if not path.exists():
            raise FileNotFoundError(f"Contract file not found: {path}")

        if path.suffix.lower() == ".pdf":
            raw = extract_pdf_raw_data(path)
            raw_text = raw["full_text"]
        elif path.suffix.lower() in {".txt", ".md"}:
            raw_text = path.read_text(encoding="utf-8")
            raw = {"full_text": raw_text, "tables": []}
        else:
            raise ValueError("Unsupported contract file type. Use .pdf or .txt")

        agent1_candidate = build_rule_based_candidate(raw)
        validated_output, validation_errors = validate_structured_output(agent1_candidate)
        # mode="json" so dates etc. become JSON-safe primitives (matches the UI and what the
        # downstream agents expect); a raw model_dump() leaves date objects that break json.dumps.
        facts = validated_output.model_dump(mode="json") if validated_output else agent1_candidate
        agent1_output = {
            "candidate_output": agent1_candidate,
            "validated_output": facts if validated_output else None,
            "validation_errors": validation_errors,
            "source_file": path.name,
        }

        agent2_output = self.agent2.run(facts)

        agent3_output = run_agent3(facts, agent2_output)
        agent4_output = self.agent4.run(agent1_output, agent2_output, agent3_output, raw_text=raw_text)
        agent5_output = self.agent5.run(agent1_output, agent3_output, agent4_output)

        disposition = agent5_output.get("disposition", "cannot_assess")
        return {
            "source_file": path.name,
            "contract_path": str(path),
            # convenience: what the frontend usually shows first
            "memo": agent5_output.get("memo", ""),
            "recognition_status": DISPOSITION_TEXT.get(disposition, disposition),
            "disposition": disposition,
            # full per-agent trace
            "agent1_output": agent1_output,
            "agent2_output": agent2_output,
            "agent3_output": agent3_output,
            "agent4_output": agent4_output,
            "agent5_output": agent5_output,
        }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the full Audit-Lens pipeline on one contract file."
    )
    parser.add_argument("contract", help="Path to a contract PDF or text file")
    parser.add_argument(
        "--out",
        default="pipeline_result.json",
        help="Path to write the JSON pipeline result",
    )
    args = parser.parse_args()

    pipeline = EndToEndPipeline()
    result = pipeline.run_contract(args.contract)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote end-to-end pipeline result to {out_path}")


if __name__ == "__main__":
    main()
