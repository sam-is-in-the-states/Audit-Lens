"""Audit Agent 4: audit Agent 3 outputs using KB3 review checklist triggers.

Agent 4 inspects Agent 1 facts, Agent 2 retrieval metadata, and Agent 3 treatment
results to decide which KB3 checklist items should be reviewed. The output is a
list of audit findings plus a predicted set of KB3 ids.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

# Avoid importing `contract_label_extraction` at module load because it pulls
# in heavy PDF deps (PyMuPDF / fitz) that may not be installed in CI/test
# environments. We build a lightweight agent1 candidate from raw text when
# needed.
from .policy_retrieval import PolicyRetrievalAgent
from .client import get_llm_response

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KB3_CHECKLIST_PATH = PROJECT_ROOT / "documents" / "knowledge_base" / "KB3_review_checklist.jsonl"


class AuditAgent:
    def __init__(
        self,
        top_k: int = 20,
        chunk_size: int = 500,
        chunk_overlap: int = 75,
        kb_quotas: dict | None = None,
    ):
        self.retriever = PolicyRetrievalAgent(
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            kb_quotas=kb_quotas,
        )
        self.checklist = self._load_kb3_checklist(KB3_CHECKLIST_PATH)

    def _load_kb3_checklist(self, path: Path) -> dict[str, dict[str, str]]:
        if not path.exists():
            raise FileNotFoundError(f"KB3 checklist file not found: {path}")

        items: dict[str, dict[str, str]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                items[item["id"].upper()] = item
        return items

    def run(
        self,
        agent1_output: dict[str, Any],
        agent2_output: dict[str, Any],
        agent3_output: dict[str, Any],
        raw_text: str | None = None,
    ) -> dict[str, Any]:
        """Audit the pipeline output and return KB3 review findings."""
        # Normalize the inputs to the shapes we expect.
        agent1 = agent1_output or {}
        agent2 = agent2_output or {}
        agent3 = agent3_output or {}

        findings = self._build_findings(agent1, agent2, agent3, raw_text)

        # Run a single LLM call that evaluates all KB3 checklist items at once.
        # The model should return a JSON object mapping KB3 ids to {applies, explanation, confidence}.
        try:
            llm_map = self._llm_check_all_kb3(list(self.checklist.keys()), agent1, agent3, raw_text)
        except Exception as exc:
            llm_map = {}

        # Apply LLM results to findings.
        for kb3_id, kb_meta in self.checklist.items():
            entry = next((f for f in findings if f["kb3_id"] == kb3_id), None)
            if entry is None:
                entry = {
                    "kb3_id": kb3_id,
                    "name": kb_meta["name"],
                    "trigger_applies": False,
                    "risk": kb_meta["risk"],
                    "trigger": kb_meta["trigger"],
                    "review_question": kb_meta["review_question"],
                    "reason": "",
                    "evidence": [],
                }
                findings.append(entry)

            raw_result = llm_map.get(kb3_id.upper())
            # Normalize possible response shapes: dict, JSON-string, or simple token
            parsed = {}
            if raw_result is None:
                parsed = {}
            elif isinstance(raw_result, dict):
                parsed = raw_result
            elif isinstance(raw_result, str):
                # Try to parse JSON string, otherwise treat as simple applies token
                try:
                    parsed = json.loads(raw_result)
                except Exception:
                    parsed = {"applies": raw_result}
            else:
                # Unexpected type — coerce to string
                parsed = {"applies": str(raw_result)}

            applies = str(parsed.get("applies") or parsed.get("decision") or "UNDETERMINED").upper()
            explanation = parsed.get("explanation") or parsed.get("reason") or parsed.get("explain") or ""
            confidence = parsed.get("confidence")
            entry["llm_applies"] = applies
            entry["llm_explanation"] = explanation
            entry["llm_confidence"] = confidence
            entry["trigger_applies"] = True if applies == "TRUE" else False
            if explanation:
                entry["reason"] = (entry.get("reason") or "") + " | LLM: " + explanation

        predicted_kb3_ids = sorted({f["kb3_id"] for f in findings if f["trigger_applies"]})
        kb3_candidate_ids = sorted(self._kb3_candidates(agent2))
        return {
            "agent": "AuditAgent",
            "contract_id": agent1.get("contract_id") or agent3.get("contract_id"),
            "source_file": agent1.get("source_file") or agent3.get("source_file"),
            "agent1_output": agent1,
            "agent2_output": agent2,
            "agent3_output": agent3,
            "audit_findings": findings,
            "kb3_candidate_ids": kb3_candidate_ids,
            "predicted_kb3_ids": predicted_kb3_ids,
        }

    def run_on_raw_text(
        self,
        raw_text: str,
        source_file: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        # Build a minimal Agent 1 candidate directly from raw text so we do not
        # import the full extraction module (which requires PyMuPDF). This is
        # sufficient for retrieval + audit evaluation runs on plain text.
        raw = {"full_text": raw_text, "tables": []}
        agent1 = {
            "full_text": raw_text,
            "services_summary": (raw_text or "")[:500],
            "hosted_service": bool(re.search(r"hosted service|hosted subscription|SaaS", raw_text, flags=re.I)),
            "implementation_services": bool(re.search(r"onboarding|implementation|setup|migration|training", raw_text, flags=re.I)),
            "onboarding_terms": {},
            "fees": [],
            "usage_fee": {},
            "discount_terms": {},
            "renewal_terms": {},
            "subscription_term_months": None,
            "start_date": None,
            "source_file": source_file,
        }

        agent2 = self.retriever.run(agent1, top_k=top_k)

        from .treatment_agent import run as run_agent3

        agent3 = run_agent3(agent1, agent2)
        return self.run(agent1, agent2, agent3, raw_text=raw_text)

    def _build_findings(
        self,
        agent1: dict[str, Any],
        agent2: dict[str, Any],
        agent3: dict[str, Any],
        raw_text: str | None,
    ) -> list[dict[str, Any]]:
        raw_text = raw_text or agent1.get("full_text", "") or ""
        evidence = []
        findings: list[dict[str, Any]] = []
        found_ids = set()

        def add_finding(kb3_id: str, trigger_applies: bool, reason: str, sources: list[str]) -> None:
            nonlocal findings, found_ids
            kb3_id = kb3_id.upper()
            if kb3_id not in self.checklist:
                return
            findings.append(
                {
                    "kb3_id": kb3_id,
                    "name": self.checklist[kb3_id]["name"],
                    "trigger_applies": trigger_applies,
                    "risk": self.checklist[kb3_id]["risk"],
                    "trigger": self.checklist[kb3_id]["trigger"],
                    "review_question": self.checklist[kb3_id]["review_question"],
                    "reason": reason,
                    "evidence": sorted(set(sources)),
                }
            )
            found_ids.add(kb3_id)

        has_onboarding = self._has_onboarding_fee(agent1)
        sub_amt, onb_amt = self._subscription_and_onboarding_amounts(agent1)
        usage_model = self._usage_model(agent1, agent3)
        discount_type = self._conclusion(agent3, "discount_type")
        material_right = self._conclusion(agent3, "material_right")
        contract_valid = self._conclusion(agent3, "contract_valid")
        billing_used = self._billing_used(agent1, agent3)
        term_months = self._term_months(agent1, agent3)
        start_date = self._start_date(agent1)
        schedule_start = self._schedule_start(agent3)
        has_discount = self._has_discount(agent1, discount_type)
        has_free_period = self._has_free_period(agent1, discount_type, raw_text)
        has_usage = self._has_usage(agent1, usage_model, raw_text)
        renewal_terms = agent1.get("renewal_terms") or {}
        auto_renewal = bool(renewal_terms.get("auto_renewal") is True)
        renewal_price = self._amount(renewal_terms.get("renewal_price"))
        po_list = list(agent3.get("treatment", {}).get("po_list") or [])
        num_pos = int(agent3.get("treatment", {}).get("num_POs") or len(po_list) or 0)
        recognition_method = agent3.get("treatment", {}).get("recognition_method")

        kb3_candidates = self._kb3_candidates(agent2)

        # Using agent3 plus agent1/raw-text when needed. If agent3 is silent,
        # agent1 facts and raw-text search supply the trigger evidence.
        add_finding(
            "KB3_1",
            bool(has_onboarding),
            "Onboarding or implementation fees are present alongside the subscription.",
            ["agent1: onboarding fee"] if has_onboarding else ["agent1: no onboarding fee"],
        )

        add_finding(
            "KB3_2",
            bool(onb_amt and sub_amt and onb_amt / max(sub_amt, 1) >= 0.5),
            "Onboarding/implementation fee is large relative to the annual subscription fee.",
            ["agent1: onboarding fee", "agent1: subscription fee"],
        )

        add_finding(
            "KB3_3",
            bool(has_usage),
            "The contract includes usage-, overage-, or volume-based fees.",
            ["agent3: usage model", "agent1: usage fee"],
        )

        add_finding(
            "KB3_4",
            usage_model in {
                "minimum_commitment",
                "usage_with_floor",
                "base_plus_overage",
                "base_plus_capped",
            },
            "The usage model includes a committed minimum or floor plus usage above an included volume.",
            ["agent3: usage_model"],
        )

        add_finding(
            "KB3_5",
            usage_model == "pure_usage",
            "The contract appears to include pure usage-based fees without a fixed or minimum amount.",
            ["agent3: usage_model"],
        )

        add_finding(
            "KB3_6",
            usage_model == "usage_royalty" or self._likely_usage_royalty(agent1, raw_text),
            "The usage fee appears tied to a license or proprietary intellectual property rather than ordinary metered usage.",
            ["agent3: usage_model", "raw_text: royalty wording"],
        )

        add_finding(
            "KB3_7",
            has_discount,
            "The contract includes a discount that may need allocation across performance obligations.",
            ["agent3: discount_type", "agent1: discount terms"],
        )

        add_finding(
            "KB3_8",
            discount_type == "residual_method",
            "The discount appears to use a residual pricing approach for one component.",
            ["agent3: discount_type"],
        )

        add_finding(
            "KB3_9",
            num_pos > 1,
            "The treatment includes multiple performance obligations, which raises allocation and recognition risk.",
            ["agent3: po_list", "agent3: num_POs"],
        )

        add_finding(
            "KB3_10",
            material_right == "TRUE" or discount_type in {"renewal_option", "upfront_fee_material_right"},
            "A material right may exist from a renewal option or an upfront fee waived on renewal.",
            ["agent3: material_right", "agent3: discount_type"],
        )

        add_finding(
            "KB3_11",
            billing_used in {
                "annual_upfront",
                "annual_upfront_each_year",
                "quarterly_upfront",
                "full_term_upfront",
            }
            or (start_date and start_date.day != 1)
            or agent3.get("billing_used") == "monthly_arrears",
            "Billing timing and service delivery may differ, affecting deferred revenue or contract assets.",
            ["agent3: billing_used", "agent1: billing terms"],
        )

        add_finding(
            "KB3_12",
            bool(start_date and start_date.day != 1) or bool(schedule_start and schedule_start.day != 1),
            "The service period begins mid-month, so proration should be reviewed.",
            ["agent1: start_date", "agent3: schedule"],
        )

        add_finding(
            "KB3_13",
            bool(term_months and term_months > 12 and billing_used in {"annual_upfront", "full_term_upfront"}),
            "A multi-year upfront prepayment may require a financing component check.",
            ["agent1: subscription term", "agent3: billing_used"],
        )

        add_finding(
            "KB3_14",
            has_free_period,
            "The contract contains free or promotional access periods that should be spread over the full access term.",
            ["agent3: discount_type", "raw_text: free period"],
        )

        add_finding(
            "KB3_15",
            contract_valid != "TRUE",
            "The contract validity or collectibility is unclear or false.",
            ["agent3: contract_valid"],
        )

        add_finding(
            "KB3_16",
            term_months is None or start_date is None,
            "The service period or contract term is missing or undetermined.",
            ["agent1: subscription_term_months", "agent1: start_date"],
        )

        add_finding(
            "KB3_17",
            self._conclusion(agent3, "onboarding_distinct") == "UNDETERMINED"
            or material_right == "UNDETERMINED"
            or recognition_method == "UNDETERMINED",
            "The contract facts are not sufficient to fix the number of performance obligations or onboarding distinctness.",
            ["agent3: onboarding_distinct", "agent3: material_right", "agent3: recognition_method"],
        )

        add_finding(
            "KB3_18",
            auto_renewal and material_right == "UNDETERMINED" and renewal_price is None,
            "A renewal option is present but the renewal price is not specified, so materiality should be reviewed.",
            ["agent1: renewal_terms", "agent3: material_right"],
        )

        # Preserve checklist order in output.
        ordered_findings = [f for f in findings if f["kb3_id"] in self.checklist]
        ordered_findings.sort(key=lambda item: list(self.checklist).index(item["kb3_id"]))
        return ordered_findings

    def _llm_check_kb3(self, kb3_id: str, finding: dict[str, Any], agent1: dict[str, Any], agent3: dict[str, Any], raw_text: str | None) -> dict[str, Any]:
        """Ask the LLM whether the KB3 checklist item applies to the current contract.

        Returns a dict with keys: 'applies' (TRUE/FALSE/UNDETERMINED) and 'explanation'.
        """
        kb = self.checklist.get(kb3_id.upper()) or {}
        system = "You are a compliance reviewer. Given a KB3 checklist item and contract facts, answer in JSON with keys: applies (TRUE/FALSE/UNDETERMINED), explanation (string), confidence (0-1)."
        prompt = (
            f"KB3 ID: {kb3_id}\n"
            f"Name: {kb.get('name')}\n"
            f"Trigger: {kb.get('trigger')}\n"
            f"Review question: {kb.get('review_question')}\n"
            f"Risk: {kb.get('risk')}\n\n"
            "Contract facts:\n"
            f"Agent1 summary: {json.dumps(agent1.get('services_summary') or '')}\n"
            f"Agent1 key fields: subscription_term_months={agent1.get('subscription_term_months')}, start_date={agent1.get('start_date')}, fees={agent1.get('fees') or []}\n"
            f"Agent3 characterization: {json.dumps(agent3.get('characterization') or {})}\n"
            f"Agent3 treatment summary: {json.dumps(agent3.get('treatment') or {})}\n\n"
            "Answer whether the KB3 trigger applies to the contract. Provide a short explanation and a confidence between 0 and 1."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        # Call LLM
        try:
            resp = get_llm_response(messages, max_tokens=800, response_format="json_object", temperature=0.0)
            # resp may be a dict-like or JSON string depending on client; ensure dict
            if isinstance(resp, str):
                try:
                    resp = json.loads(resp)
                except Exception:
                    # fallback: wrap in explanation
                    return {"applies": "UNDETERMINED", "explanation": resp}
            # Normalize keys
            applies = resp.get("applies") or resp.get("decision") or resp.get("applies_to_contract")
            explanation = resp.get("explanation") or resp.get("reason") or ""
            confidence = resp.get("confidence")
            return {"applies": str(applies).upper() if applies is not None else "UNDETERMINED", "explanation": explanation, "confidence": confidence}
        except Exception as exc:
            raise

    def _llm_check_all_kb3(self, kb3_ids: list[str], agent1: dict[str, Any], agent3: dict[str, Any], raw_text: str | None) -> dict[str, dict[str, Any]]:
        """Single LLM call to evaluate all KB3 items. Returns mapping KB3_ID -> {applies, explanation, confidence}.
        """
        kb_items = []
        for kid in kb3_ids:
            kb = self.checklist.get(kid.upper(), {})
            kb_items.append({
                "id": kid.upper(),
                "name": kb.get("name"),
                "trigger": kb.get("trigger"),
                "review_question": kb.get("review_question"),
                "risk": kb.get("risk"),
            })

        system = "You are a compliance reviewer. For each KB3 checklist item provided, decide whether it applies to the contract. Return a single JSON object mapping KB3 id to {applies, explanation, confidence}. Applies must be one of TRUE, FALSE, or UNDETERMINED. Confidence is a number 0-1."

        prompt = {
            "kb3_items": kb_items,
            "contract_facts": {
                "services_summary": agent1.get("services_summary"),
                "subscription_term_months": agent1.get("subscription_term_months"),
                "start_date": agent1.get("start_date"),
                "fees": agent1.get("fees"),
                "characterization": agent3.get("characterization"),
                "treatment": agent3.get("treatment"),
            },
            "raw_text_excerpt": (raw_text or "")[:2000],
        }

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
        ]

        resp = get_llm_response(messages, max_tokens=2000, response_format="json_object", temperature=0.0)
        # Normalize to dict
        if isinstance(resp, str):
            try:
                resp = json.loads(resp)
            except Exception:
                return {}
        # Uppercase keys
        out = {}
        for k, v in (resp.items() if isinstance(resp, dict) else []):
            out[str(k).upper()] = v
        return out

    def _conclusion(self, agent3: dict[str, Any], field: str) -> str:
        node = (agent3.get("characterization") or {}).get(field) or {}
        if isinstance(node, dict):
            return str(node.get("conclusion") or "").strip()
        return str(node or "").strip()

    def _amount(self, node: Any) -> Optional[float]:
        if isinstance(node, dict):
            return self._amount(node.get("amount"))
        if isinstance(node, (int, float)):
            return float(node)
        if isinstance(node, str):
            try:
                return float(node.replace("$", "").replace(",", "").strip())
            except ValueError:
                return None
        return None

    def _has_onboarding_fee(self, agent1: dict[str, Any]) -> bool:
        ons = self._find_fee_by_keywords(agent1, ["onboard", "implementation", "setup", "training", "migration", "activation"])
        if ons is not None:
            return True
        onboarding_terms = agent1.get("onboarding_terms") or {}
        return bool(self._amount(onboarding_terms.get("fee")))

    def _subscription_and_onboarding_amounts(self, agent1: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
        fees = agent1.get("fees") or []
        onb_amount = None
        sub_amount = None
        for fee in fees:
            amount = self._amount(fee.get("amount"))
            if amount is None:
                continue
            name = str(fee.get("name") or "").lower()
            if any(k in name for k in ("onboard", "implementation", "setup", "training", "migration", "activation")):
                onb_amount = onb_amount or amount
            else:
                if sub_amount is None or amount > sub_amount:
                    sub_amount = amount
        onboarding_terms = agent1.get("onboarding_terms") or {}
        if onb_amount is None:
            onb_amount = self._amount(onboarding_terms.get("fee"))
        if sub_amount is None:
            sub_amount = self._find_fee_by_keywords(agent1, ["subscription", "platform", "service"])
        return sub_amount, onb_amount

    def _find_fee_by_keywords(self, agent1: dict[str, Any], keywords: Iterable[str]) -> Optional[float]:
        for fee in agent1.get("fees") or []:
            amount = self._amount(fee.get("amount"))
            if amount is None:
                continue
            name = str(fee.get("name") or "").lower()
            if any(k in name for k in keywords):
                return amount
        return None

    def _usage_model(self, agent1: dict[str, Any], agent3: dict[str, Any]) -> str:
        usage = self._conclusion(agent3, "usage_model")
        if usage:
            return usage
        usage_fee = agent1.get("usage_fee") or {}
        if usage_fee.get("overage_rate") or usage_fee.get("included_units"):
            return "pure_usage"
        return "none"

    def _has_usage(self, agent1: dict[str, Any], usage_model: str, raw_text: str) -> bool:
        if usage_model != "none":
            return True
        usage_fee = agent1.get("usage_fee") or {}
        if usage_fee.get("overage_rate") or usage_fee.get("included_units"):
            return True
        return bool(re.search(r"usage|overage|metered|transaction(s)? processed|consumption", raw_text, flags=re.I))

    def _likely_usage_royalty(self, agent1: dict[str, Any], raw_text: str) -> bool:
        if "royalty" in str(raw_text).lower():
            return True
        usage_fee = agent1.get("usage_fee") or {}
        if usage_fee.get("unit_name") and "license" in str(usage_fee.get("unit_name")).lower():
            return True
        return False

    def _has_discount(self, agent1: dict[str, Any], discount_type: str) -> bool:
        if discount_type and discount_type != "none":
            return True
        discount_terms = agent1.get("discount_terms") or {}
        if discount_terms.get("has_discount") is True:
            return True
        if discount_terms.get("amount") or discount_terms.get("percentage"):
            return True
        return False

    def _has_free_period(self, agent1: dict[str, Any], discount_type: str, raw_text: str) -> bool:
        if discount_type == "free_period":
            return True
        if re.search(r"free (month|months|period|trial)|no charge.*month|complimentary months", raw_text, flags=re.I):
            return True
        return False

    def _billing_used(self, agent1: dict[str, Any], agent3: dict[str, Any]) -> str:
        billing = agent3.get("billing_used") or ""
        if billing:
            return billing
        fees = agent1.get("fees") or []
        timings = " ".join(str(f.get("payment_timing") or "") for f in fees).lower()
        freqs = " ".join(str(f.get("billing_frequency") or "") for f in fees).lower()
        if "arrear" in timings:
            return "monthly_arrears"
        if "quarter" in freqs:
            return "quarterly_upfront"
        if "annual" in freqs or "advance" in timings:
            return "annual_upfront"
        if "month" in freqs:
            return "monthly_arrears"
        return "annual_upfront"

    def _term_months(self, agent1: dict[str, Any], agent3: dict[str, Any]) -> Optional[int]:
        term = agent3.get("treatment", {}).get("recognition_period_months")
        if term:
            return int(term)
        value = agent1.get("subscription_term_months")
        if isinstance(value, (int, float)):
            return int(value)
        return None

    def _start_date(self, agent1: dict[str, Any]) -> Optional[Path]:
        date_str = agent1.get("start_date") or agent1.get("effective_date")
        if not date_str:
            return None
        try:
            from datetime import date
            if isinstance(date_str, date):
                return date_str
            return date.fromisoformat(str(date_str))
        except Exception:
            return None

    def _schedule_start(self, agent3: dict[str, Any]) -> Optional[Any]:
        schedule = agent3.get("schedule", {}).get("monthly_schedule") or []
        if not schedule:
            return None
        first = schedule[0]
        return self._parse_iso_date(first.get("period_start"))

    def _parse_iso_date(self, value: Any) -> Optional[Any]:
        if not value:
            return None
        try:
            from datetime import date
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            return None

    def _kb3_candidates(self, agent2: dict[str, Any]) -> list[str]:
        chunks = agent2.get("retrieved_policy_chunks") or []
        return [str(chunk.get("doc_id", "")).upper() for chunk in chunks if str(chunk.get("kb_prefix", "")).upper() == "KB3"]


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
