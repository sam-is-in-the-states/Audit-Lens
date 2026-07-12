"""Audit Agent: combine extraction + policy retrieval + optional LLM re-ranking

Given raw contract text (or an extracted contract dict), return the list of
KB3 policy IDs that an auditor should check for that contract.

This agent uses the rule-based extractor as a quick way to produce the
structured contract representation expected by the retrieval agent, then
calls PolicyRetrievalAgent to find candidate KB chunks. Optionally an LLM
(via `app.llm.client.get_llm_response`) is asked to classify which KB3
documents are truly relevant; the LLM step is a helpful refinement but
is not required for the evaluation runner.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set

from .policy_retrieval import PolicyRetrievalAgent
from .contract_label_extraction import build_rule_based_candidate
from .client import get_llm_response


class AuditAgent:
    def __init__(
        self,
        top_k: int = 20,
        chunk_size: int = 500,
        chunk_overlap: int = 75,
        kb_quotas: dict | None = None,
        llm_temperature: float = 0.0,
    ):
        self.retriever = PolicyRetrievalAgent(
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            kb_quotas=kb_quotas,
        )
        self.llm_temperature = llm_temperature

    def run_on_raw_text(self, raw_text: str, source_file: str | None = None, top_k: int | None = None) -> Dict[str, Any]:
        """Run the audit agent on raw contract text and return KB3 ids.

        Returns a dict with keys:
          - source_file
          - agent1_output
          - retrieved_policy_chunks
          - kb3_candidate_ids
          - kb3_selected_ids
        """
        raw = {"full_text": raw_text, "tables": []}
        agent1 = build_rule_based_candidate(raw)
        if source_file:
            agent1["source_file"] = source_file

        retrieved = self.retriever.run(agent1, top_k=top_k)

        retrieved_chunks: List[dict] = retrieved["retrieved_policy_chunks"]

        kb3_candidate_ids: Set[str] = {
            chunk["doc_id"].upper()
            for chunk in retrieved_chunks
            if str(chunk.get("kb_prefix", "")).upper() == "KB3"
        }

        kb3_selected_ids: Set[str]

        # Ask the LLM to select which KB3 ids are relevant to the contract.
        prompt = self._build_llm_prompt(agent1, retrieved_chunks)
        try:
            reply = get_llm_response(
                [
                    {"role": "system", "content": "You are an accounting audit assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.llm_temperature,
            )

            # Expect either a JSON array of ids or a plain list. Try to parse JSON first.
            parsed = None
            try:
                parsed = json.loads(reply)
            except Exception:
                # Fallback: pull KB ids using a simple regex-like scan
                parsed = [tok.strip().upper() for tok in reply.replace(",", " ").split() if tok.upper().startswith("KB3_")]

            if isinstance(parsed, list):
                kb3_selected_ids = {str(i).upper() for i in parsed}
            else:
                kb3_selected_ids = set()
        except Exception:
            kb3_selected_ids = set(kb3_candidate_ids)
        

        return {
            "agent": "AuditAgent",
            "source_file": source_file,
            "agent1_output": agent1,
            "retrieved_policy_chunks": retrieved_chunks,
            "kb3_candidate_ids": sorted(kb3_candidate_ids),
            "kb3_selected_ids": sorted(kb3_selected_ids),
        }

    def _build_llm_prompt(self, agent1: dict, retrieved_chunks: List[dict]) -> str:
        contract_summary = agent1.get("services_summary") or agent1.get("contract_id") or "(no summary)"
        parts = [f"Contract summary: {contract_summary}\n\n"]
        parts.append("Candidate KB3 documents and short excerpts:\n")

        for chunk in retrieved_chunks:
            if str(chunk.get("kb_prefix", "")).upper() != "KB3":
                continue
            parts.append(
                f"ID: {chunk['doc_id']}\nTitle: {chunk.get('title')}\nText: {chunk.get('text')[:800].strip()}\n---\n"
            )

        # Provide clear selection rule and examples to the LLM so it returns
        # a clean JSON array of KB3 ids only.
        parts.append(
            "\nInstruction:\n"
            "Select only the KB3 policy IDs that are directly relevant for an auditor to review based on the contract facts shown above.\n"
            "- Include a KB3 id when the contract contains the issue described by that KB3 policy (for example, an onboarding fee and ambiguity about distinctness should include KB3_1).\n"
            "- Do not include KB1 or KB2 ids here — return only KB3_* ids.\n"
            "- Answer with a JSON array only (for example: [\"KB3_1\", \"KB3_5\"]). Do not add any other text.\n\n"
        )

        parts.append("Examples:\n")
        parts.append(
            "Example 1 - Simple subscription only:\n"
            "  Contract facts: Hosted subscription, standard support, no onboarding fee, fixed annual subscription.\n"
            "  Expected output: []\n"
        )
        parts.append(
            "Example 2 - Onboarding fee present and onboarding required:\n"
            "  Contract facts: One-time onboarding fee, onboarding required before access, no evidence onboarding is standardized.\n"
            "  Relevant checklist: KB3_1 (bundled_onboarding_distinctness), KB3_2 may be considered if fee is very large.\n"
            "  Expected output: [\"KB3_1\"]\n"
        )
        parts.append(
            "Example 3 - Usage-based fees and royalty question:\n"
            "  Contract facts: Variable fees based on processed transactions; contract mentions use of proprietary analytics library.\n"
            "  Relevant checklist: KB3_3 (variable_consideration_present), KB3_6 (usage_royalty_vs_ordinary_usage).\n"
            "  Expected output: [\"KB3_3\", \"KB3_6\"]\n"
        )

        parts.append(
            "Now review the candidate KB3 documents above and return a JSON array with the KB3 ids that are relevant for auditing this contract. Only include the IDs in the array."
        )

        return "\n".join(parts)


def load_contract_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("txt", help="path to contract text file")
    args = ap.parse_args()

    txt = load_contract_text(Path(args.txt))
    agent = AuditAgent()
    out = agent.run_on_raw_text(txt, source_file=Path(args.txt).name)
    print(json.dumps(out, indent=2, ensure_ascii=False))
