import sys
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.llm.contract_label_extraction import (
    extract_pdf_raw_data,
    build_rule_based_candidate,
    validate_structured_output,
)

# App configuration
st.set_page_config(
    page_title="Audit-Lens",
    page_icon="📄",
    layout="wide",
)

EVALUATION_DIR_CANDIDATES = [
    PROJECT_ROOT / "evaluation" / "results",
    PROJECT_ROOT / "evaluation",
    Path(__file__).resolve().parent,
]

EVALUATION_JSON_NAMES = [
    "evaluation_outputs.json",
]

EVALUATION_CSV_NAMES = [
    "evaluation_summary.csv",
]

AGENT2_JSON_NAMES = [
    "agent2_final_evaluation.json",
    "policy_retrieval_results.json",
]

# Styling helpers
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .audit-card {
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.75rem;
            background: rgba(128, 128, 128, 0.035);
        }

        .audit-card h4 {
            margin: 0 0 0.35rem 0;
        }

        .muted {
            opacity: 0.72;
            font-size: 0.92rem;
        }

        .pill-pass {
            display: inline-block;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            background: rgba(30, 170, 90, 0.15);
            border: 1px solid rgba(30, 170, 90, 0.35);
            font-weight: 600;
        }

        .pill-fail {
            display: inline-block;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            background: rgba(220, 60, 60, 0.14);
            border: 1px solid rgba(220, 60, 60, 0.35);
            font-weight: 600;
        }

        .pill-review {
            display: inline-block;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            background: rgba(230, 160, 30, 0.14);
            border: 1px solid rgba(230, 160, 30, 0.38);
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Utility functions
def find_existing_file(
    names: list[str],
) -> Path | None:
    for directory in EVALUATION_DIR_CANDIDATES:
        for name in names:
            path = directory / name
            if path.exists():
                return path
    return None


@st.cache_data(show_spinner=False)
def load_evaluation_json(
    path_string: str,
) -> list[dict[str, Any]]:
    path = Path(path_string)

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "evaluation_outputs.json must contain a list of contract results."
        )

    return data


@st.cache_data(show_spinner=False)
def load_evaluation_summary(
    path_string: str,
) -> pd.DataFrame:
    return pd.read_csv(path_string)


@st.cache_data(show_spinner=False)
def load_json_file(
    path_string: str,
) -> Any:
    with Path(path_string).open("r", encoding="utf-8") as file:
        return json.load(file)


def safe_value(
    value: Any,
    fallback: str = "N/A",
) -> Any:
    if value in (None, "", [], {}):
        return fallback
    return value


def severity_rank(severity: str) -> int:
    order = {
        "CRITICAL": 0,
        "HIGH": 1,
        "MEDIUM": 2,
        "LOW": 3,
    }
    return order.get(str(severity).upper(), 99)


def severity_icon(severity: str) -> str:
    icons = {
        "CRITICAL": "🔴",
        "HIGH": "🟠",
        "MEDIUM": "🟡",
        "LOW": "🔵",
    }
    return icons.get(str(severity).upper(), "⚪")


def status_badge(status: str) -> str:
    normalized = str(status).upper()

    if normalized == "PASS":
        css_class = "pill-pass"
    elif normalized == "FAIL":
        css_class = "pill-fail"
    else:
        css_class = "pill-review"

    return (
        f'<span class="{css_class}">'
        f"{normalized}"
        f"</span>"
    )


def confidence_label(score: float) -> str:
    if score >= 90:
        return "High confidence"
    if score >= 75:
        return "Proceed with caution"
    return "Manual review recommended"


def build_issue_dataframe(
    issues: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = []

    for issue in issues:
        rows.append(
            {
                "Severity": str(
                    issue.get("severity", "UNKNOWN")
                ).upper(),
                "Issue": issue.get(
                    "kb3_name",
                    issue.get("description", "Unknown issue"),
                ),
                "Type": issue.get("issue_type", ""),
                "Recommended Action": issue.get(
                    "recommended_action",
                    "",
                ),
                "Timing": issue.get("timing", ""),
                "KB": issue.get("kb3_id", ""),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Severity",
                "Issue",
                "Type",
                "Recommended Action",
                "Timing",
                "KB",
            ]
        )

    frame = pd.DataFrame(rows)

    frame["_severity_rank"] = frame["Severity"].map(
        severity_rank
    )

    return (
        frame.sort_values(
            ["_severity_rank", "Issue"]
        )
        .drop(columns=["_severity_rank"])
        .reset_index(drop=True)
    )


def render_confidence_breakdown(
    breakdown: dict[str, Any],
) -> None:
    if not breakdown:
        st.info("No confidence breakdown is available.")
        return

    cols = st.columns(
        min(4, max(1, len(breakdown)))
    )

    for index, (group, details) in enumerate(
        breakdown.items()
    ):
        score = details.get("score", 0)
        status = details.get("status", "UNKNOWN")

        with cols[index % len(cols)]:
            st.markdown(
                f"""
                <div class="audit-card">
                    <div class="muted">
                        {group.replace("_", " ").title()}
                    </div>
                    <h4>{score}%</h4>
                    {status_badge(status)}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_pipeline_health(
    agent2_path: Path | None,
    evaluation_summary: pd.DataFrame | None,
) -> None:
    st.subheader("5-Agent Pipeline Health")
    st.caption(
        "Only metrics supported by saved evaluation artifacts are shown. "
        "Missing agent metrics are labeled rather than estimated."
    )

    columns = st.columns(5)

    # Agent 1
    with columns[0]:
        st.markdown("#### Agent 1")
        st.caption("Contract extraction")
        st.metric("Metric", "Validation")
        st.caption(
            "Per-upload schema validation is shown in Contract Review."
        )

    # Agent 2
    with columns[1]:
        st.markdown("#### Agent 2")
        st.caption("Policy retrieval")

        agent2_metric = None
        agent2_detail = "Final evaluation file not found."

        if agent2_path:
            try:
                agent2_data = load_json_file(
                    str(agent2_path)
                )

                heldout = (
                    agent2_data.get("heldout_metrics", {})
                    if isinstance(agent2_data, dict)
                    else {}
                )

                if heldout.get("f1") is not None:
                    agent2_metric = (
                        f"{float(heldout['f1']):.1%}"
                    )
                    agent2_detail = (
                        "Held-out F1 "
                        f"• Recall {float(heldout.get('recall', 0)):.1%} "
                        f"• Precision "
                        f"{float(heldout.get('precision', 0)):.1%}"
                    )
            except Exception:
                agent2_detail = (
                    "Agent 2 evaluation file could not be parsed."
                )

        st.metric(
            "Held-out F1",
            agent2_metric or "—",
        )
        st.caption(agent2_detail)

    # Agent 3
    with columns[2]:
        st.markdown("#### Agent 3")
        st.caption("Accounting treatment")
        st.metric("Accuracy", "—")
        st.caption(
            "Add an Agent 3 ground-truth evaluation file to show accuracy."
        )

    # Agent 4
    with columns[3]:
        st.markdown("#### Agent 4")
        st.caption("Risk / reasoning")
        st.metric("Accuracy", "—")
        st.caption(
            "Trigger decisions are available in Agent 5 trace, "
            "but no standalone accuracy file was provided."
        )

    # Agent 5
    with columns[4]:
        st.markdown("#### Agent 5")
        st.caption("Evaluation")

        if evaluation_summary is not None and not evaluation_summary.empty:
            mean_confidence = float(
                evaluation_summary[
                    "overall_confidence"
                ].mean()
            )
            st.metric(
                "Avg. confidence",
                f"{mean_confidence:.1f}%",
            )
            st.caption(
                f"{len(evaluation_summary)} evaluated contracts"
            )
        else:
            st.metric("Avg. confidence", "—")
            st.caption("Evaluation summary not found.")


def render_evaluation_dashboard(
    evaluation_data: list[dict[str, Any]],
    summary_df: pd.DataFrame | None,
) -> None:
    st.header("Evaluation Dashboard")
    st.caption(
        "Review Agent 5 outcomes, manual-review issues, and trace details "
        "without reading raw JSON."
    )

    if not evaluation_data:
        st.info("No saved evaluation results are available.")
        return

    # Portfolio summary
    confidences = [
        float(item.get("confidence", 0))
        for item in evaluation_data
    ]
    issue_counts = [
        int(item.get("issues_count", 0))
        for item in evaluation_data
    ]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Contracts evaluated",
        len(evaluation_data),
    )
    c2.metric(
        "Average confidence",
        f"{sum(confidences) / len(confidences):.1f}%",
    )
    c3.metric(
        "Manual review items",
        sum(issue_counts),
    )

    if summary_df is not None and not summary_df.empty:
        pass_total = int(summary_df["pass_count"].sum())
        fail_total = int(summary_df["fail_count"].sum())

        pass_rate = (
            pass_total / (pass_total + fail_total)
            if pass_total + fail_total
            else 0
        )

        c4.metric(
            "PASS rate",
            f"{pass_rate:.1%}",
        )
    else:
        c4.metric(
            "PASS rate",
            "—",
        )

    st.divider()

    # Contract selector
    contract_labels = {
        (
            f"{item.get('contract_id', 'Unknown')} "
            f"• {item.get('confidence', 0)}% confidence "
            f"• {item.get('issues_count', 0)} issue(s)"
        ): item
        for item in evaluation_data
    }

    selected_label = st.selectbox(
        "Select evaluated contract",
        list(contract_labels.keys()),
    )

    selected = contract_labels[selected_label]
    full_output = selected.get("full_output", {})

    confidence = float(
        full_output.get(
            "overall_confidence",
            selected.get("confidence", 0),
        )
    )

    issues = full_output.get("issues", [])

    # Executive summary
    st.subheader(
        f"Contract {selected.get('contract_id', '')}"
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Overall confidence",
        f"{confidence:.0f}%",
    )
    m2.metric(
        "Manual review items",
        len(issues),
    )

    critical_count = sum(
        str(issue.get("severity", "")).upper()
        == "CRITICAL"
        for issue in issues
    )

    m3.metric(
        "Critical issues",
        critical_count,
    )

    trace_summary = (
        full_output
        .get("trace", {})
        .get("summary_statistics", {})
    )

    triggered = int(
        trace_summary.get(
            "items_triggered_by_agent4",
            0,
        )
    )

    m4.metric(
        "Agent 4 triggers",
        triggered,
    )

    if confidence >= 90:
        st.success(
            f"{confidence_label(confidence)} — "
            "review the flagged exceptions below."
        )
    elif confidence >= 75:
        st.warning(
            f"{confidence_label(confidence)} — "
            "manual review is recommended before recognition."
        )
    else:
        st.error(
            f"{confidence_label(confidence)} — "
            "do not rely on the result without review."
        )

    st.subheader("Control / Confidence Breakdown")

    render_confidence_breakdown(
        full_output.get(
            "confidence_breakdown",
            {},
        )
    )

    # Manual review queue
    st.subheader("Manual Review Queue")

    issues_df = build_issue_dataframe(issues)

    if issues_df.empty:
        st.success("No manual-review issues were flagged.")
    else:
        severity_filter = st.multiselect(
            "Filter by severity",
            ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            default=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            key="evaluation_severity_filter",
        )

        filtered_issues = issues_df[
            issues_df["Severity"].isin(
                severity_filter
            )
        ]

        st.dataframe(
            filtered_issues,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Severity": st.column_config.TextColumn(
                    width="small",
                ),
                "Issue": st.column_config.TextColumn(
                    width="medium",
                ),
                "Type": st.column_config.TextColumn(
                    width="small",
                ),
                "Recommended Action": (
                    st.column_config.TextColumn(
                        width="large",
                    )
                ),
                "Timing": st.column_config.TextColumn(
                    width="medium",
                ),
                "KB": st.column_config.TextColumn(
                    width="small",
                ),
            },
        )

        for issue in sorted(
            issues,
            key=lambda item: severity_rank(
                item.get("severity", "")
            ),
        ):
            title = (
                f"{severity_icon(issue.get('severity', ''))} "
                f"{issue.get('severity', 'UNKNOWN')} — "
                f"{issue.get('kb3_name', 'Issue')}"
            )

            with st.expander(title):
                st.write(
                    issue.get(
                        "description",
                        "No description available.",
                    )
                )

                a1, a2 = st.columns(2)

                with a1:
                    st.markdown("**Recommended action**")
                    st.write(
                        safe_value(
                            issue.get(
                                "recommended_action"
                            )
                        )
                    )

                with a2:
                    st.markdown("**Timing**")
                    st.write(
                        safe_value(
                            issue.get("timing")
                        )
                    )

                st.caption(
                    f"Policy: {issue.get('kb3_id', 'N/A')} "
                    f"• Agent 3 field: "
                    f"{issue.get('agent3_field', 'N/A')}"
                )

    # Trace / explainability
    st.subheader("Agent Trace")

    trace = full_output.get("trace", {})
    evaluations = trace.get("evaluations", [])

    if evaluations:
        trace_rows = []

        for evaluation in evaluations:
            trace_rows.append(
                {
                    "Policy": evaluation.get(
                        "kb3_id",
                        "",
                    ),
                    "Risk": evaluation.get(
                        "kb3_name",
                        "",
                    ),
                    "Triggered by Agent 4": (
                        "Yes"
                        if evaluation.get(
                            "trigger_from_agent4"
                        )
                        else "No"
                    ),
                    "Agent 5 Status": evaluation.get(
                        "status",
                        "",
                    ),
                    "Confidence": evaluation.get(
                        "confidence",
                        "",
                    ),
                    "Group": evaluation.get(
                        "group",
                        "",
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(trace_rows),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander(
            "View detailed reasoning trace"
        ):
            selected_trace_policy = st.selectbox(
                "Policy trace",
                [
                    (
                        f"{entry.get('kb3_id', '')} — "
                        f"{entry.get('kb3_name', '')}"
                    )
                    for entry in evaluations
                ],
            )

            selected_index = [
                (
                    f"{entry.get('kb3_id', '')} — "
                    f"{entry.get('kb3_name', '')}"
                )
                for entry in evaluations
            ].index(selected_trace_policy)

            entry = evaluations[selected_index]

            st.markdown("**Risk**")
            st.write(entry.get("kb3_risk", ""))

            st.markdown("**Trigger**")
            st.write(entry.get("kb3_trigger", ""))

            st.markdown("**Agent 4 reasoning**")
            st.write(entry.get("agent4_reasoning", ""))

            st.markdown("**Agent 3 response**")
            st.json(entry.get("agent3_response", {}))

            st.markdown("**Agent 5 validation**")
            st.json(entry.get("validation", {}))

    # Memo/raw output
    with st.expander("View final review memo"):
        st.text(
            full_output.get(
                "memo",
                selected.get("memo_preview", ""),
            )
        )

    with st.expander("View raw Agent 5 JSON"):
        st.json(full_output)

# Header
st.title("Audit-Lens")
st.subheader("ASC 606 Contract Review & Agent Evaluation")

st.markdown(
    """
Upload a contract for structured extraction and validation, or use the
evaluation dashboard to inspect outputs from the multi-agent pipeline.
"""
)

evaluation_json_path = find_existing_file(
    EVALUATION_JSON_NAMES
)
evaluation_csv_path = find_existing_file(
    EVALUATION_CSV_NAMES
)
agent2_json_path = find_existing_file(
    AGENT2_JSON_NAMES
)

evaluation_data = None
evaluation_summary = None

if evaluation_json_path:
    try:
        evaluation_data = load_evaluation_json(
            str(evaluation_json_path)
        )
    except Exception as error:
        st.sidebar.error(
            f"Could not load evaluation JSON: {error}"
        )

if evaluation_csv_path:
    try:
        evaluation_summary = load_evaluation_summary(
            str(evaluation_csv_path)
        )
    except Exception as error:
        st.sidebar.error(
            f"Could not load evaluation CSV: {error}"
        )

# Main navigation
tab_review, tab_eval, tab_pipeline = st.tabs(
    [
        "📄 Contract Review",
        "📊 Evaluation Dashboard",
        "🧠 Pipeline Health",
    ]
)

# Contract review tab
with tab_review:
    st.header("Contract Review")

    uploaded_file = st.file_uploader(
        "Upload a contract PDF",
        type=["pdf"],
        key="contract_upload",
    )

    if uploaded_file:
        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

        if st.button(
            "Run Contract Review",
            type="primary",
        ):
            progress = st.progress(0)
            status = st.empty()

            try:
                upload_dir = (
                    Path(__file__).resolve().parents[1]
                    / "llm"
                    / "contracts"
                    / "uploaded_contracts"
                )

                upload_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                filename = uploaded_file.name
                saved_pdf = upload_dir / filename

                counter = 1

                while saved_pdf.exists():
                    stem = Path(filename).stem
                    suffix = Path(filename).suffix

                    saved_pdf = (
                        upload_dir
                        / f"{stem}_{counter}{suffix}"
                    )

                    counter += 1

                with saved_pdf.open("wb") as file:
                    file.write(
                        uploaded_file.getbuffer()
                    )

                pdf_path = str(saved_pdf)

                progress.progress(20)
                status.info(
                    "Step 1/5: PDF uploaded and saved."
                )

                raw = extract_pdf_raw_data(pdf_path)

                progress.progress(40)
                status.info(
                    "Step 2/5: Raw PDF data extracted."
                )

                candidate = build_rule_based_candidate(
                    raw
                )

                progress.progress(60)
                status.info(
                    "Step 3/5: Contract labels extracted."
                )

                validated, errors = (
                    validate_structured_output(
                        candidate
                    )
                )

                progress.progress(80)
                status.info(
                    "Step 4/5: Structured output validated."
                )

                result = {
                    "file_name": raw["file_name"],
                    "saved_pdf_path": str(saved_pdf),
                    "raw_text_preview": (
                        raw["full_text"][:1000]
                    ),
                    "section_names": list(
                        raw["sections"].keys()
                    ),
                    "tables": raw["tables"],
                    "candidate_output": candidate,
                    "validated_output": (
                        validated.model_dump(
                            mode="json"
                        )
                        if validated
                        else None
                    ),
                    "validation_errors": errors,
                }

                progress.progress(100)
                status.success("Review complete!")

                output = (
                    result["validated_output"]
                    or result["candidate_output"]
                )

                st.divider()

                # Executive summary
                st.subheader("Contract Summary")

                summary_cols = st.columns(5)

                summary_cols[0].metric(
                    "Validation",
                    (
                        "VALID"
                        if validated
                        else "INVALID"
                    ),
                )

                summary_cols[1].metric(
                    "Term",
                    (
                        f"{output.get('subscription_term_months')} mo"
                        if output.get(
                            "subscription_term_months"
                        )
                        else "N/A"
                    ),
                )

                summary_cols[2].metric(
                    "Hosted service",
                    str(
                        safe_value(
                            output.get(
                                "hosted_service"
                            )
                        )
                    ),
                )

                summary_cols[3].metric(
                    "Authorized users",
                    str(
                        safe_value(
                            output.get(
                                "authorized_users"
                            )
                        )
                    ),
                )

                summary_cols[4].metric(
                    "Implementation",
                    str(
                        safe_value(
                            output.get(
                                "implementation_services"
                            )
                        )
                    ),
                )

                left, right = st.columns(
                    [1.05, 1.4]
                )

                with left:
                    st.markdown(
                        "### Contract Details"
                    )

                    details = pd.DataFrame(
                        [
                            {
                                "Field": "Contract ID",
                                "Value": safe_value(
                                    output.get(
                                        "contract_id"
                                    )
                                ),
                            },
                            {
                                "Field": "Provider",
                                "Value": safe_value(
                                    output.get(
                                        "provider_name"
                                    )
                                ),
                            },
                            {
                                "Field": "Customer",
                                "Value": safe_value(
                                    output.get(
                                        "customer_name"
                                    )
                                ),
                            },
                            {
                                "Field": "Effective Date",
                                "Value": safe_value(
                                    output.get(
                                        "effective_date"
                                    )
                                ),
                            },
                            {
                                "Field": "Start Date",
                                "Value": safe_value(
                                    output.get(
                                        "start_date"
                                    )
                                ),
                            },
                            {
                                "Field": "End Date",
                                "Value": safe_value(
                                    output.get(
                                        "end_date"
                                    )
                                ),
                            },
                        ]
                    )

                    st.dataframe(
                        details,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        label="Download Review JSON",
                        data=json.dumps(
                            result,
                            indent=2,
                        ),
                        file_name=(
                            f"{Path(uploaded_file.name).stem}"
                            "_review.json"
                        ),
                        mime="application/json",
                    )

                with right:
                    st.markdown(
                        "### Revenue Terms"
                    )

                    terms_tabs = st.tabs(
                        [
                            "Fees",
                            "Discount",
                            "Usage",
                            "Renewal",
                            "Termination",
                            "Ambiguities",
                        ]
                    )

                    with terms_tabs[0]:
                        st.json(
                            output.get(
                                "fees",
                                [],
                            )
                        )

                    with terms_tabs[1]:
                        st.json(
                            output.get(
                                "discount_terms",
                                {},
                            )
                        )

                    with terms_tabs[2]:
                        st.json(
                            output.get(
                                "usage_fee",
                                {},
                            )
                        )

                    with terms_tabs[3]:
                        st.json(
                            output.get(
                                "renewal_terms",
                                {},
                            )
                        )

                    with terms_tabs[4]:
                        st.json(
                            output.get(
                                "termination_terms",
                                {},
                            )
                        )

                    with terms_tabs[5]:
                        ambiguities = output.get(
                            "ambiguous_clauses",
                            [],
                        )

                        if ambiguities:
                            st.json(ambiguities)
                        else:
                            st.success(
                                "No ambiguous clauses "
                                "were identified."
                            )

                with st.expander(
                    "Structured output / debugging"
                ):
                    if validated:
                        st.success(
                            "Schema validation: VALID"
                        )
                        st.json(
                            result[
                                "validated_output"
                            ]
                        )
                    else:
                        st.error(
                            "Schema validation: INVALID"
                        )
                        st.json(
                            result[
                                "candidate_output"
                            ]
                        )

                    if errors:
                        st.markdown(
                            "**Validation errors**"
                        )
                        st.json(errors)

                    st.markdown(
                        "**Raw text preview**"
                    )
                    st.text(
                        result["raw_text_preview"]
                    )

            except Exception as error:
                progress.progress(0)
                status.error("Review failed.")
                st.exception(error)

# Evaluation dashboard tab
with tab_eval:
    if evaluation_data is None:
        st.info(
            "No evaluation_outputs.json was found. "
            "Place it in evaluation/results/ or evaluation/ "
            "to populate this dashboard."
        )
    else:
        render_evaluation_dashboard(
            evaluation_data,
            evaluation_summary,
        )

# Pipeline health tab
with tab_pipeline:
    render_pipeline_health(
        agent2_json_path,
        evaluation_summary,
    )

    st.divider()

    st.subheader("Recommended Evaluation Metrics")

    st.markdown(
        """
**Agent 1 — Extraction**
- Schema-valid output rate
- Field-level accuracy for dates, fees, discount, usage, onboarding, and renewal
- Missing-field / hallucination rate

**Agent 2 — Policy Retrieval**
- Held-out Recall, Precision, and F1
- KB1 / KB2 / KB3 recall
- Average number of policies returned

**Agent 3 — Accounting Treatment**
- Treatment decision accuracy
- Performance-obligation count accuracy
- Transaction-price and recognition-method accuracy

**Agent 4 — Risk Detection**
- Risk-trigger precision / recall
- Critical-risk miss rate
- False-positive trigger rate

**Agent 5 — Evaluation**
- Average confidence
- PASS / FAIL / undetermined distribution
- Manual-review issue count and severity
"""
    )

    if evaluation_summary is not None:
        st.subheader("Agent 5 Portfolio Summary")

        chart_df = (
            evaluation_summary[
                [
                    "contract_id",
                    "pass_count",
                    "fail_count",
                    "undetermined_problematic",
                ]
            ]
            .set_index("contract_id")
        )

        st.bar_chart(
            chart_df,
            use_container_width=True,
        )

        with st.expander(
            "View evaluation summary table"
        ):
            st.dataframe(
                evaluation_summary,
                use_container_width=True,
                hide_index=True,
            )