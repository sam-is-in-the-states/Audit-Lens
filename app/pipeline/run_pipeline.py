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
from app.llm.evaluation_agent import EvaluationAgent


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
        self.agent4 = AuditAgent(
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            kb_quotas=kb_quotas,
        )
        self.agent5 = EvaluationAgent()

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
        agent1_output = {
            "candidate_output": agent1_candidate,
            "validated_output": validated_output.model_dump() if validated_output else None,
            "validation_errors": validation_errors,
            "source_file": path.name,
        }

        facts = validated_output.model_dump() if validated_output else agent1_candidate
        agent2_output = self.agent2.run(facts)

        agent3_output = run_agent3(facts, agent2_output)
        agent4_output = self.agent4.run(facts, agent2_output, agent3_output, raw_text=raw_text)
        agent5_output = self.agent5.run(agent3_output, agent4_output, facts)

        return {
            "source_file": path.name,
            "contract_path": str(path),
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
