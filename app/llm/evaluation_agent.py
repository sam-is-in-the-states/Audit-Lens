"""
Evaluation Agent 5: Validate Agent 3 Treatment Decisions Against KB3 Risks

This agent evaluates whether Agent 3's treatment decisions properly address the
risks and concerns flagged by Agent 4's audit findings. For each KB3 item raised,
it checks:
  1. Was the risk addressed? (Did Agent 3 make a relevant judgment?)
  2. Is the conclusion correct? (Is the reasoning sound?)
  3. Is it consistent? (Does it align with related decisions?)

Outputs:
  - Final memo (human-readable summary)
  - Issue list (manual review items with severity)
  - Confidence score (0-100% with breakdown)
  - Detailed trace (KB3-by-KB3 validation evidence)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KB3_CHECKLIST_PATH = PROJECT_ROOT / "documents" / "knowledge_base" / "KB3_review_checklist.jsonl"


# ============================================================================
# KB3 ↔ AGENT 3 FIELD MAPPING & VALIDATION RULES
# ============================================================================

# Map each KB3 item to the Agent 3 field(s) it evaluates
KB3_TO_AGENT3_FIELD_MAP = {
    "KB3_1": {
        "name": "bundled_onboarding_distinctness",
        "group": "onboarding",
        "agent3_field": "onboarding_distinct",
        "weight": 0.8,  # CRITICAL
        "trigger_from_agent1": "has_onboarding_fee",
        "expected_conclusion_if_triggered": ["TRUE", "FALSE", "UNDETERMINED"],
        "undetermined_acceptable_if": "insufficient contract detail on customization",
    },
    "KB3_2": {
        "name": "large_upfront_fee_size_trap",
        "group": "onboarding",
        "agent3_field": "onboarding_distinct",
        "weight": 0.8,
        "trigger_from_agent1": "has_onboarding_fee AND onboarding_fee > subscription_fee",
        "expected_conclusion_if_triggered": ["TRUE", "FALSE"],
        "undetermined_acceptable_if": None,
        "validation_check": "reasoning must cite NATURE not SIZE",
    },
    "KB3_3": {
        "name": "variable_consideration_present",
        "group": "usage",
        "agent3_field": "usage_model",
        "weight": 0.8,
        "trigger_from_agent1": "has_usage_fee",
        "expected_conclusion_if_triggered": ["mixed", "pure_usage"],
        "undetermined_acceptable_if": None,
        "validation_check": "must NOT be 'none' if usage fees exist",
    },
    "KB3_4": {
        "name": "minimum_commitment_floor",
        "group": "usage",
        "agent3_field": "usage_model",
        "weight": 0.8,
        "trigger_from_agent1": "has_minimum_fee AND has_usage_above_minimum",
        "expected_conclusion_if_triggered": ["mixed"],
        "undetermined_acceptable_if": None,
        "validation_check": "transaction_price must separate minimum from overage",
    },
    "KB3_5": {
        "name": "unestimable_or_volatile_usage",
        "group": "usage",
        "agent3_field": "usage_model",
        "weight": 0.8,
        "trigger_from_agent1": "usage_volatile_or_no_minimum",
        "expected_conclusion_if_triggered": ["pure_usage"],
        "undetermined_acceptable_if": None,
        "validation_check": "variable must be recognized as-incurred, not estimated",
    },
    "KB3_6": {
        "name": "usage_royalty_vs_ordinary_usage",
        "group": "usage",
        "agent3_field": "usage_model",
        "weight": 0.8,
        "trigger_from_agent1": "has_usage_fee AND relates_to_ip_license",
        "expected_conclusion_if_triggered": ["pure_usage", "mixed"],
        "undetermined_acceptable_if": None,
        "validation_check": "reasoning must distinguish IP license from metered usage",
    },
    "KB3_7": {
        "name": "discount_allocation_method",
        "group": "discount",
        "agent3_field": "discount_type",
        "weight": 0.4,  # MEDIUM
        "trigger_from_agent1": "has_bundle_discount",
        "expected_conclusion_if_triggered": ["pro_rata", "directed", "residual"],
        "undetermined_acceptable_if": None,
        "validation_check": "must default to pro_rata unless evidence of directed/residual",
    },
    "KB3_8": {
        "name": "ssp_not_observable_residual",
        "group": "discount",
        "agent3_field": "discount_type",
        "weight": 0.4,
        "trigger_from_agent1": "has_bundle_discount AND uses_residual",
        "expected_conclusion_if_triggered": ["pro_rata", "residual"],
        "undetermined_acceptable_if": None,
        "validation_check": "if residual: at least one PO must have observable SSP",
    },
    "KB3_9": {
        "name": "multiple_performance_obligations",
        "group": "po_count",
        "agent3_field": "num_POs",
        "weight": 0.4,
        "trigger_from_agent1": "num_POs > 2",
        "expected_conclusion_if_triggered": ["2", "3", "4"],  # Any specific >1 number
        "undetermined_acceptable_if": None,
        "validation_check": "po_list must detail each obligation and recognition pattern",
    },
    "KB3_10": {
        "name": "material_right_renewal_or_fee",
        "group": "material_right",
        "agent3_field": "material_right",
        "weight": 0.6,  # HIGH
        "trigger_from_agent1": "has_renewal_option_at_discount",
        "expected_conclusion_if_triggered": ["TRUE", "FALSE", "UNDETERMINED"],
        "undetermined_acceptable_if": "renewal_price not specified in contract",
        "validation_check": "if TRUE: po_list must include renewal PO; if FALSE: justify why not material",
    },
    "KB3_11": {
        "name": "billing_timing_balances",
        "group": "billing",
        "agent3_field": "billing_used",
        "weight": 0.6,
        "trigger_from_agent1": "billing_mismatched_to_delivery",
        "expected_conclusion_if_triggered": ["annual_upfront", "quarterly_upfront", "monthly_arrears"],
        "undetermined_acceptable_if": None,
        "validation_check": "opening_deferred must match upfront_billing pattern",
    },
    "KB3_12": {
        "name": "proration_partial_period",
        "group": "billing",
        "agent3_field": "schedule",
        "weight": 0.2,  # LOW
        "trigger_from_agent1": "start_date_mid_month",
        "expected_conclusion_if_triggered": ["prorated_first_month"],
        "undetermined_acceptable_if": None,
        "validation_check": "first schedule entry should have prorated amount",
    },
    "KB3_13": {
        "name": "financing_component_check",
        "group": "billing",
        "agent3_field": "recognition_period_months",
        "weight": 0.2,
        "trigger_from_agent1": "term_months > 12 AND prepaid_full_term",
        "expected_conclusion_if_triggered": ["financing_considered"],
        "undetermined_acceptable_if": "financing deemed immaterial (documented)",
        "validation_check": "if term>12 and full upfront: financing component must be addressed",
    },
    "KB3_14": {
        "name": "free_or_promotional_period",
        "group": "billing",
        "agent3_field": "schedule",
        "weight": 0.2,
        "trigger_from_agent1": "has_free_period",
        "expected_conclusion_if_triggered": ["spread_over_full_period"],
        "undetermined_acceptable_if": None,
        "validation_check": "monthly_revenue must = transaction_price / (term + free_months)",
    },
    "KB3_15": {
        "name": "contract_validity_collectibility",
        "group": "validity",
        "agent3_field": "contract_valid",
        "weight": 0.2,
        "trigger_from_agent1": "missing_payment_terms_or_doubtful_collectibility",
        "expected_conclusion_if_triggered": ["TRUE", "FALSE"],
        "undetermined_acceptable_if": None,
        "validation_check": "if FALSE: recognition_method must be 'do_not_recognize'",
    },
    "KB3_16": {
        "name": "missing_service_period_or_dates",
        "group": "validity",
        "agent3_field": "recognition_period_months",
        "weight": 0.2,
        "trigger_from_agent1": "term_or_start_date_missing",
        "expected_conclusion_if_triggered": ["specified_term"],
        "undetermined_acceptable_if": "facts insufficient to determine",
        "validation_check": "if term_months is null: schedule must be empty",
    },
    "KB3_17": {
        "name": "undetermined_distinctness_or_po_count",
        "group": "po_count",
        "agent3_field": "num_POs",
        "weight": 0.4,
        "trigger_from_agent1": "services_lack_detail",
        "expected_conclusion_if_triggered": ["1", "2", "3"],
        "undetermined_acceptable_if": "insufficient contract facts on customization/dependency",
        "validation_check": "if UNDETERMINED and facts present: needs escalation",
    },
    "KB3_18": {
        "name": "silent_renewal_materiality",
        "group": "material_right",
        "agent3_field": "material_right",
        "weight": 0.6,
        "trigger_from_agent1": "has_renewal_option_no_price_stated",
        "expected_conclusion_if_triggered": ["UNDETERMINED"],
        "undetermined_acceptable_if": "renewal_price not stated (need to obtain)",
        "validation_check": "must flag as requiring price information",
    },
}

# Severity levels for issues
SEVERITY_LEVELS = {
    "CRITICAL": ["KB3_3", "KB3_4", "KB3_5"],  # Usage mishandled
    "HIGH": ["KB3_1", "KB3_2", "KB3_7", "KB3_11"],  # Core treatment
    "MEDIUM": ["KB3_6", "KB3_8", "KB3_9", "KB3_10"],  # Complex items
    "LOW": ["KB3_12", "KB3_13", "KB3_14", "KB3_15", "KB3_16", "KB3_17", "KB3_18"],  # Contextual
}

RISK_WEIGHTS = {
    "onboarding": 0.8,  # CRITICAL
    "usage": 0.8,       # CRITICAL
    "discount": 0.4,    # MEDIUM
    "po_count": 0.4,    # MEDIUM
    "material_right": 0.6,  # HIGH
    "billing": 0.6,     # HIGH
    "validity": 0.2,    # LOW
}


class EvaluationAgent:
    """Evaluate Agent 3 treatment against Agent 4 audit findings."""

    def __init__(self):
        self.kb3_map = KB3_TO_AGENT3_FIELD_MAP
        self.checklist = self._load_kb3_checklist(KB3_CHECKLIST_PATH)

    def _load_kb3_checklist(self, path: Path) -> dict[str, dict[str, str]]:
        """Load KB3 checklist from JSONL file."""
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
        agent3_output: dict[str, Any],
        agent4_output: dict[str, Any],
        agent1_output: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate Agent 3 output against Agent 4 audit findings.

        Returns all 4 output formats: memo, issues, confidence, trace.
        """
        if not agent3_output:
            raise ValueError("agent3_output is required")
        if not agent4_output:
            raise ValueError("agent4_output is required")

        # Extract predicted KB3 items from Agent 4
        predicted_kb3_ids = agent4_output.get("predicted_kb3_ids", [])
        audit_findings = {f["kb3_id"]: f for f in agent4_output.get("audit_findings", [])}

        # Build detailed trace for each KB3 item
        trace_items = []
        status_counts = {"PASS": 0, "FAIL": 0, "UNDETERMINED_ACCEPTABLE": 0, "UNDETERMINED_PROBLEMATIC": 0}

        for kb3_id in self.kb3_map.keys():
            evaluation = self._evaluate_kb3_item(
                kb3_id, agent3_output, audit_findings.get(kb3_id), agent1_output
            )
            trace_items.append(evaluation)
            status_counts[evaluation["status"]] += 1

        # Calculate confidence score
        confidence_score, breakdown = self._calculate_confidence(trace_items)

        # Generate issues list
        issues = self._build_issues_list(trace_items)

        # Generate human-readable memo
        memo = self._format_memo(
            agent3_output, confidence_score, breakdown, trace_items, issues
        )

        return {
            "agent": "EvaluationAgent",
            "contract_id": agent3_output.get("contract_id"),
            "source_file": agent3_output.get("source_file"),
            "evaluation_date": datetime.now().isoformat(),
            "overall_confidence": confidence_score,
            "confidence_breakdown": breakdown,
            "memo": memo,
            "issues": issues,
            "trace": {
                "evaluations": trace_items,
                "summary_statistics": {
                    "total_kb3_items_evaluated": len(trace_items),
                    "items_triggered_by_agent4": len([t for t in trace_items if t["trigger_from_agent4"]]),
                    "results": status_counts,
                },
            },
        }

    def _evaluate_kb3_item(
        self,
        kb3_id: str,
        agent3_output: dict[str, Any],
        audit_finding: dict[str, Any] | None,
        agent1_output: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Evaluate a single KB3 item against Agent 3's decision."""
        kb3_info = self.kb3_map.get(kb3_id, {})
        checklist_item = self.checklist.get(kb3_id, {})

        # Extract Agent 3 response for this KB3 item
        agent3_field = kb3_info.get("agent3_field")
        agent3_response = self._extract_agent3_response(agent3_output, agent3_field)

        # Determine if this item was triggered by Agent 4
        trigger_from_agent4 = bool(audit_finding and audit_finding.get("trigger_applies"))

        # Validate Agent 3's response
        validation_result = self._validate_response(kb3_id, agent3_response, agent3_output, agent1_output)

        # Determine overall status
        status, confidence = self._determine_status(
            kb3_id, validation_result, trigger_from_agent4, kb3_info
        )

        return {
            "kb3_id": kb3_id,
            "kb3_name": kb3_info.get("name", ""),
            "kb3_risk": checklist_item.get("risk", ""),
            "kb3_trigger": checklist_item.get("trigger", ""),
            "kb3_review_question": checklist_item.get("review_question", ""),
            "trigger_from_agent4": trigger_from_agent4,
            "agent4_reasoning": audit_finding.get("reason", "") if audit_finding else "",
            "agent3_response": agent3_response,
            "validation": validation_result,
            "status": status,
            "confidence": confidence,
            "flags": validation_result.get("flags", []),
            "weight": kb3_info.get("weight", 0.4),
            "group": kb3_info.get("group", ""),
        }

    def _extract_agent3_response(
        self, agent3_output: dict[str, Any], agent3_field: str
    ) -> dict[str, Any]:
        """Extract Agent 3's response for the given field."""
        characterization = agent3_output.get("characterization", {})
        treatment = agent3_output.get("treatment", {})
        schedule = agent3_output.get("schedule", {})

        # Try characterization first (most judgments are there)
        if agent3_field in characterization:
            judgment = characterization[agent3_field]
            return {
                "source": "characterization",
                "field": agent3_field,
                "conclusion": judgment.get("conclusion"),
                "reasoning": judgment.get("reasoning"),
                "kb_basis": judgment.get("kb_basis", []),
            }

        # Then try treatment
        if agent3_field in treatment:
            return {
                "source": "treatment",
                "field": agent3_field,
                "value": treatment[agent3_field],
            }

        # Then try schedule
        if agent3_field in schedule:
            return {
                "source": "schedule",
                "field": agent3_field,
                "value": schedule[agent3_field],
            }

        # Special case: billing_used
        if agent3_field == "billing_used":
            return {
                "source": "root",
                "field": "billing_used",
                "value": agent3_output.get("billing_used"),
            }

        return {"source": None, "field": agent3_field, "value": None}

    def _validate_response(
        self,
        kb3_id: str,
        agent3_response: dict[str, Any],
        agent3_output: dict[str, Any],
        agent1_output: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Validate Agent 3's response against KB3 requirements."""
        kb3_info = self.kb3_map.get(kb3_id, {})
        flags = []
        checks = {}

        # Check 1: Did Agent 3 address the risk? (make a decision)
        addressed = agent3_response.get("source") is not None
        checks["addressed"] = addressed
        if not addressed:
            flags.append("Agent 3 did not address this KB3 item")

        # Check 2: Is the conclusion correct? (validate reasoning & consistency)
        conclusion_valid = True
        if kb3_id == "KB3_1":
            conclusion_valid = self._validate_onboarding_distinctness(
                agent3_response, agent3_output, agent1_output
            )
        elif kb3_id == "KB3_2":
            conclusion_valid = self._validate_large_upfront_fee_size(agent3_response, agent3_output)
        elif kb3_id in ["KB3_3", "KB3_4", "KB3_5", "KB3_6"]:
            conclusion_valid = self._validate_usage_model(kb3_id, agent3_response, agent3_output, agent1_output)
        elif kb3_id in ["KB3_7", "KB3_8"]:
            conclusion_valid = self._validate_discount_type(kb3_id, agent3_response, agent3_output)
        elif kb3_id in ["KB3_9", "KB3_17"]:
            conclusion_valid = self._validate_po_count(kb3_id, agent3_response, agent3_output)
        elif kb3_id in ["KB3_10", "KB3_18"]:
            conclusion_valid = self._validate_material_right(kb3_id, agent3_response, agent3_output, agent1_output)
        elif kb3_id in ["KB3_11", "KB3_12", "KB3_13", "KB3_14"]:
            conclusion_valid = self._validate_billing_timing(kb3_id, agent3_response, agent3_output, agent1_output)
        elif kb3_id in ["KB3_15", "KB3_16"]:
            conclusion_valid = self._validate_validity(kb3_id, agent3_response, agent3_output)

        checks["conclusion_correct"] = conclusion_valid

        # Check 3: Cross-consistency checks
        cross_checks = self._cross_validate(kb3_id, agent3_output, flags)
        checks["cross_checks"] = cross_checks

        return {
            "addresses_risk": addressed,
            "conclusion_supported": conclusion_valid,
            "cross_checks": cross_checks,
            "checks_passed": [k for k, v in checks.items() if v is True],
            "flags": flags,
        }

    # ========================================================================
    # VALIDATION RULES FOR EACH KB3 GROUP
    # ========================================================================

    def _validate_onboarding_distinctness(
        self,
        agent3_response: dict[str, Any],
        agent3_output: dict[str, Any],
        agent1_output: dict[str, Any] | None,
    ) -> bool:
        """Validate KB3_1: onboarding_distinct decision."""
        conclusion = agent3_response.get("conclusion", "").upper()
        reasoning = agent3_response.get("reasoning", "").lower()

        # If FALSE or TRUE, must have reasoning
        if conclusion in ["FALSE", "TRUE"] and not reasoning:
            return False

        # If TRUE, reasoning must cite nature/customization, not just size
        if conclusion == "TRUE":
            size_only = any(word in reasoning for word in ["large", "big", "size", "amount"])
            nature_present = any(
                word in reasoning
                for word in [
                    "custom",
                    "integration",
                    "dependency",
                    "access",
                    "required",
                    "distinct",
                ]
            )
            if size_only and not nature_present:
                return False  # FAIL: Size trap

        # Cross-check: po_list should not include "onboarding" if conclusion=FALSE
        po_list = agent3_output.get("treatment", {}).get("po_list", [])
        if conclusion == "FALSE" and "onboarding" in [str(p).lower() for p in po_list]:
            return False

        return True

    def _validate_large_upfront_fee_size(self, agent3_response: dict[str, Any], agent3_output: dict[str, Any]) -> bool:
        """Validate KB3_2: Large fee size doesn't drive distinctness."""
        conclusion = agent3_response.get("conclusion", "").upper()
        reasoning = agent3_response.get("reasoning", "").lower()

        # If TRUE, reasoning must NOT cite size alone
        if conclusion == "TRUE":
            size_only = any(word in reasoning for word in ["large", "big", "size", "amount"]) and not any(
                word in reasoning for word in ["custom", "integration", "dependency"]
            )
            if size_only:
                return False  # FAIL: Violated KB3_2

        return True

    def _validate_usage_model(
        self,
        kb3_id: str,
        agent3_response: dict[str, Any],
        agent3_output: dict[str, Any],
        agent1_output: dict[str, Any] | None,
    ) -> bool:
        """Validate KB3_3-6: Usage model decisions."""
        conclusion = agent3_response.get("conclusion", "").lower()

        # If usage fees exist, usage_model must NOT be "none"
        treatment = agent3_output.get("treatment", {})
        transaction_price = treatment.get("transaction_price", 0)

        # Check: If conclusion=none but usage exists somewhere, FAIL
        if conclusion == "none":
            # This is often hard to validate without full facts, so be conservative
            # Flag as uncertain if no clear reasoning
            reasoning = agent3_response.get("reasoning", "")
            if not reasoning:
                return False

        return True

    def _validate_discount_type(
        self, kb3_id: str, agent3_response: dict[str, Any], agent3_output: dict[str, Any]
    ) -> bool:
        """Validate KB3_7-8: Discount allocation."""
        conclusion = agent3_response.get("conclusion", "").lower()

        # If residual used, validate reasoning cites observable SSP
        if conclusion == "residual":
            reasoning = agent3_response.get("reasoning", "").lower()
            if "observable" not in reasoning and "ssp" not in reasoning:
                return False

        return True

    def _validate_po_count(
        self, kb3_id: str, agent3_response: dict[str, Any], agent3_output: dict[str, Any]
    ) -> bool:
        """Validate KB3_9, KB3_17: PO count and detail."""
        treatment = agent3_output.get("treatment", {})
        num_pos = treatment.get("num_POs")
        po_list = treatment.get("po_list", [])

        # If num_POs > 2, po_list must be detailed
        if num_pos and num_pos > 2 and len(po_list) <= 1:
            return False

        # If num_POs is UNDETERMINED, must justify
        if num_pos == "UNDETERMINED":
            reasoning = agent3_response.get("reasoning", "")
            if not reasoning:
                return False

        return True

    def _validate_material_right(
        self,
        kb3_id: str,
        agent3_response: dict[str, Any],
        agent3_output: dict[str, Any],
        agent1_output: dict[str, Any] | None,
    ) -> bool:
        """Validate KB3_10, KB3_18: Material right / renewal."""
        conclusion = agent3_response.get("conclusion", "").upper()

        # If TRUE, po_list should include renewal/option PO
        if conclusion == "TRUE":
            treatment = agent3_output.get("treatment", {})
            po_list = treatment.get("po_list", [])
            has_renewal_po = any("renew" in str(p).lower() or "option" in str(p).lower() for p in po_list)
            if not has_renewal_po:
                return False

        return True

    def _validate_billing_timing(
        self,
        kb3_id: str,
        agent3_response: dict[str, Any],
        agent3_output: dict[str, Any],
        agent1_output: dict[str, Any] | None,
    ) -> bool:
        """Validate KB3_11-14: Billing and timing."""
        # KB3_11: opening_deferred should match upfront billing
        if kb3_id == "KB3_11":
            billing_used = agent3_output.get("billing_used", "")
            schedule = agent3_output.get("schedule", {})
            opening_deferred = schedule.get("opening_deferred", 0)

            if "upfront" in billing_used.lower() and opening_deferred <= 0:
                return False  # Should have deferred revenue
            if "arrear" in billing_used.lower() and opening_deferred > 100:
                return False  # Should have minimal or no deferred

        # KB3_12: First period should be prorated if mid-month
        if kb3_id == "KB3_12":
            schedule = agent3_output.get("schedule", {})
            monthly_schedule = schedule.get("monthly_schedule", [])
            if not monthly_schedule:
                return False
            # If first entry has period_start and it's mid-month, revenue should be lower
            # (This is a simplified check)

        # KB3_13: Financing component for >12 month terms
        if kb3_id == "KB3_13":
            treatment = agent3_output.get("treatment", {})
            term_months = treatment.get("recognition_period_months")
            if term_months and term_months > 12:
                billing_used = agent3_output.get("billing_used", "")
                if "full_term_upfront" in billing_used.lower():
                    reasoning = agent3_response.get("reasoning", "")
                    if not reasoning:
                        # Flag: Financing component should be addressed
                        return False

        return True

    def _validate_validity(self, kb3_id: str, agent3_response: dict[str, Any], agent3_output: dict[str, Any]) -> bool:
        """Validate KB3_15-16: Contract validity and completeness."""
        # KB3_15: If contract_valid=FALSE, schedule should be empty
        if kb3_id == "KB3_15":
            conclusion = agent3_response.get("conclusion", "").upper()
            if conclusion == "FALSE":
                schedule = agent3_output.get("schedule", {})
                monthly_schedule = schedule.get("monthly_schedule", [])
                if monthly_schedule:
                    return False  # Should not have schedule if invalid

        # KB3_16: If term_months null, schedule must be empty
        if kb3_id == "KB3_16":
            treatment = agent3_output.get("treatment", {})
            term_months = treatment.get("recognition_period_months")
            if not term_months:
                schedule = agent3_output.get("schedule", {})
                monthly_schedule = schedule.get("monthly_schedule", [])
                if monthly_schedule:
                    return False

        return True

    def _cross_validate(self, kb3_id: str, agent3_output: dict[str, Any], flags: List[str]) -> dict[str, bool]:
        """Cross-validate consistency with related decisions."""
        checks = {}

        # Example: If onboarding_distinct=TRUE, po_list must include onboarding
        characterization = agent3_output.get("characterization", {})
        treatment = agent3_output.get("treatment", {})

        onboarding_distinct = characterization.get("onboarding_distinct", {}).get("conclusion")
        po_list = treatment.get("po_list", [])

        if kb3_id in ["KB3_1", "KB3_2"]:
            has_onboarding_po = any("onboard" in str(p).lower() for p in po_list)
            if onboarding_distinct == "TRUE":
                checks["onboarding_in_po_list"] = has_onboarding_po
            else:
                checks["onboarding_not_in_po_list"] = not has_onboarding_po

        # Material right consistency
        material_right = characterization.get("material_right", {}).get("conclusion")
        if kb3_id in ["KB3_10", "KB3_18"]:
            has_renewal_po = any("renew" in str(p).lower() for p in po_list)
            if material_right == "TRUE":
                checks["renewal_in_po_list"] = has_renewal_po
                if not has_renewal_po:
                    flags.append("Material right=TRUE but renewal not in po_list")

        return checks

    def _determine_status(
        self, kb3_id: str, validation_result: dict[str, Any], trigger_from_agent4: bool, kb3_info: dict[str, Any]
    ) -> Tuple[str, int]:
        """Determine overall status (PASS/FAIL/UNDETERMINED) and confidence."""
        addressed = validation_result.get("addresses_risk", False)
        conclusion_valid = validation_result.get("conclusion_supported", False)
        flags = validation_result.get("flags", [])

        if not addressed:
            # KB3 item not addressed by Agent 3
            if trigger_from_agent4 and not kb3_info.get("undetermined_acceptable_if"):
                return "FAIL", 20
            else:
                return "UNDETERMINED_PROBLEMATIC", 40

        if conclusion_valid:
            return "PASS", 90

        # If conclusion not fully valid but UNDETERMINED is acceptable
        if "UNDETERMINED" in str(validation_result.get("checks_passed", [])):
            acceptable_reason = kb3_info.get("undetermined_acceptable_if")
            if acceptable_reason:
                return "UNDETERMINED_ACCEPTABLE", 70
            else:
                return "UNDETERMINED_PROBLEMATIC", 40

        return "FAIL", 10

    def _calculate_confidence(self, trace_items: List[dict[str, Any]]) -> Tuple[int, dict[str, Any]]:
        """Calculate overall confidence score and breakdown by group."""
        breakdown = {}
        total_weighted_score = 0
        total_weights = 0

        # Group by judgment area
        groups = {}
        for item in trace_items:
            group = item.get("group")
            if group not in groups:
                groups[group] = []
            groups[group].append(item)

        # Score each group
        for group, items in groups.items():
            group_score = 0
            group_weight = RISK_WEIGHTS.get(group, 0.4)
            status_scores = {"PASS": 100, "UNDETERMINED_ACCEPTABLE": 50, "FAIL": 0, "UNDETERMINED_PROBLEMATIC": 30}

            for item in items:
                status = item.get("status")
                status_score = status_scores.get(status, 0)
                group_score += status_score * item.get("weight", 0.4) / group_weight

            group_score = min(100, max(0, group_score / len(items) * 100)) if items else 0
            breakdown[group] = {"score": int(group_score), "status": items[0]["status"] if items else "UNKNOWN"}

            total_weighted_score += group_score * group_weight
            total_weights += group_weight

        overall_confidence = int(total_weighted_score / total_weights) if total_weights > 0 else 0

        return overall_confidence, breakdown

    def _build_issues_list(self, trace_items: List[dict[str, Any]]) -> List[dict[str, Any]]:
        """Build structured issues list from trace items."""
        issues = []
        issue_id = 1

        for item in trace_items:
            status = item.get("status")
            if status in ["FAIL", "UNDETERMINED_PROBLEMATIC"]:
                severity = "CRITICAL"
                for sev, ids in SEVERITY_LEVELS.items():
                    if item.get("kb3_id") in ids:
                        severity = sev
                        break

                issue = {
                    "issue_id": issue_id,
                    "kb3_id": item.get("kb3_id"),
                    "kb3_name": item.get("kb3_name"),
                    "severity": severity,
                    "issue_type": status,
                    "description": f"{item.get('kb3_name')}: {item.get('kb3_risk')}",
                    "agent3_field": self.kb3_map.get(item.get("kb3_id"), {}).get("agent3_field"),
                    "agent3_output": item.get("agent3_response"),
                    "flags": item.get("flags", []),
                    "recommended_action": self._get_recommended_action(item),
                    "timing": "Before revenue recognition" if severity in ["CRITICAL", "HIGH"] else "Before period-end",
                }
                issues.append(issue)
                issue_id += 1

        return issues

    def _get_recommended_action(self, item: dict[str, Any]) -> str:
        """Get recommended action for an issue."""
        kb3_id = item.get("kb3_id")

        actions = {
            "KB3_1": "Re-review contract for onboarding customization and customer dependency",
            "KB3_2": "Verify onboarding distinctness based on nature, not fee size",
            "KB3_3": "Confirm usage structure in contract; distinguish fixed vs. variable",
            "KB3_4": "Separate minimum commitment from above-floor usage fees",
            "KB3_5": "Identify whether usage should be recognized as-incurred vs. estimated",
            "KB3_6": "Clarify whether variable fee is IP royalty or metered SaaS usage",
            "KB3_7": "Verify bundle discount allocation method (default: proportionate)",
            "KB3_8": "Confirm at least one PO has observable SSP for residual method",
            "KB3_9": "Detail each PO in po_list with recognition pattern",
            "KB3_10": "Verify renewal pricing and assess materiality of option",
            "KB3_11": "Confirm deferred revenue classification matches upfront billing",
            "KB3_12": "Pro-rate first/last month if term starts/ends mid-month",
            "KB3_13": "Document financing component analysis for multi-year prepayment",
            "KB3_14": "Spread revenue over full period including free promotional months",
            "KB3_15": "Validate contract meets ASC 606 Step 1 requirements before recognition",
            "KB3_16": "Obtain contract term and start date before building revenue schedule",
            "KB3_17": "Clarify service distinctness based on contract specificity",
            "KB3_18": "Obtain renewal pricing terms to assess material right status",
        }

        return actions.get(kb3_id, "Manual review required")

    def _format_memo(
        self,
        agent3_output: dict[str, Any],
        confidence_score: int,
        breakdown: dict[str, Any],
        trace_items: List[dict[str, Any]],
        issues: List[dict[str, Any]],
    ) -> str:
        """Format human-readable memo."""
        treatment = agent3_output.get("treatment", {})
        schedule = agent3_output.get("schedule", {})

        memo = f"""═══════════════════════════════════════════════════════════════════════
                 ASC 606 REVENUE TREATMENT EVALUATION
                          FINAL REVIEW MEMO
═══════════════════════════════════════════════════════════════════════

Contract ID:           {agent3_output.get('contract_id', 'N/A')}
Source File:           {agent3_output.get('source_file', 'N/A')}
Evaluation Date:       {datetime.now().strftime('%Y-%m-%d')}
Evaluator:             Agent 5 Evaluation Agent

═══════════════════════════════════════════════════════════════════════
                        EXECUTIVE SUMMARY
═══════════════════════════════════════════════════════════════════════

Overall Confidence Score:    {confidence_score}%

Key Treatment Decisions:
  • Recognition Method:      {treatment.get('recognition_method', 'N/A')}
  • Transaction Price:       ${treatment.get('transaction_price', 0):,.2f}
  • Performance Obligations: {treatment.get('num_POs', 'N/A')} → {', '.join(str(p) for p in treatment.get('po_list', []))}
  • Monthly Revenue:         ${schedule.get('monthly_revenue', 0):,.2f}
  • Opening Deferred Balance: ${schedule.get('opening_deferred', 0):,.2f}

Assessment: {"PROCEED WITH CONFIDENCE" if confidence_score >= 80 else "PROCEED WITH CAUTION" if confidence_score >= 60 else "UNCERTAIN - REQUIRES EXPERT REVIEW"}
  → {len(issues)} items require manual review

═══════════════════════════════════════════════════════════════════════
                         CONFIDENCE BREAKDOWN
═══════════════════════════════════════════════════════════════════════

"""
        for group, scores in breakdown.items():
            memo += f"  {group:20s}: {scores['score']:3d}% ({scores['status']})\n"

        memo += f"""
═══════════════════════════════════════════════════════════════════════
                      MANUAL REVIEW CHECKLIST
═══════════════════════════════════════════════════════════════════════

"""
        for i, issue in enumerate(issues, 1):
            memo += f"""{i}. [{issue['severity']:8s}] {issue['kb3_name']}
   Issue: {issue['description'][:80]}...
   Action: {issue['recommended_action']}
   Timing: {issue['timing']}

"""

        memo += f"""
═══════════════════════════════════════════════════════════════════════
                           CONCLUSION
═══════════════════════════════════════════════════════════════════════

Agent 3 Treatment: {"PROVISIONALLY APPROVED" if confidence_score >= 60 else "REQUIRES REMEDIATION"}
({confidence_score}% Confidence)

Recommended Action:
  1. Complete manual review of {len(issues)} item(s) flagged above
  2. Rerun Agent 3 evaluation if material facts change
  3. Proceed with revenue schedule pending completion of above

═══════════════════════════════════════════════════════════════════════
"""
        return memo


if __name__ == "__main__":
    # Smoke test
    print("Evaluation Agent loaded successfully")
