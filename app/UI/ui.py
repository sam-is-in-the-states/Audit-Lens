import sys
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.llm.memo_agent import ASC_MAP, KB_LABEL, DISPOSITION_TEXT
from app.pipeline.run_pipeline import EndToEndPipeline

# App configuration
st.set_page_config(
    page_title="Rev-Lens",
    layout="wide",
)

# Completed analyses are kept here so an earlier contract can be reopened after a later one has
# been run. Regenerable - safe to delete or to keep out of version control.
HISTORY_DIR = PROJECT_ROOT / "analysis_history"

# Styling helpers
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_json_file(
    path_string: str,
) -> Any:
    with Path(path_string).open("r", encoding="utf-8") as file:
        return json.load(file)


# The pipeline loads and chunks the whole knowledge base on construction, so it is built once
# per session and reused across uploads rather than rebuilt on every rerun.
@st.cache_resource(show_spinner=False)
def get_pipeline() -> EndToEndPipeline:
    return EndToEndPipeline()


def saved_result_paths() -> list[Path]:
    """Every analysis available to open without calling the model: past runs from this machine,
    newest first, followed by any result file kept at the project root."""
    history = sorted(
        HISTORY_DIR.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return history + sorted(PROJECT_ROOT.glob("*_result.json"))


def analysis_label(path: Path) -> str:
    """Label a saved analysis without opening it - history filenames carry their own metadata."""
    if path.parent != HISTORY_DIR:
        return path.name

    parts = path.stem.split("__")
    if len(parts) != 3:
        return path.name

    source, contract_id, stamp = parts
    try:
        when = datetime.strptime(stamp, "%Y%m%d-%H%M%S").strftime("%b %d, %H:%M")
    except ValueError:
        when = stamp

    return f"{source}  ·  {contract_id}  ·  {when}"


def save_analysis(result: dict[str, Any]) -> Path:
    """Keep every completed run, so an earlier contract can be reopened after a later one is
    analysed. The filename carries the source, the contract id and the run time, which keeps
    the picker readable without reading all the files back."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    source = Path(str(result.get("source_file") or "contract")).stem
    contract_id = (
        (result.get("agent5_output") or {}).get("contract_id")
        or "unknown"
    )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    def clean(text: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-") or "unknown"

    target = HISTORY_DIR / f"{clean(source)}__{clean(contract_id)}__{stamp}.json"
    target.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return target


def save_uploaded_contract(uploaded) -> Path:
    """Persist an upload under a non-colliding name and return the path."""
    upload_dir = PROJECT_ROOT / "app" / "llm" / "contracts" / "uploaded_contracts"
    upload_dir.mkdir(parents=True, exist_ok=True)

    target = upload_dir / uploaded.name
    counter = 1
    while target.exists():
        stem = Path(uploaded.name).stem
        suffix = Path(uploaded.name).suffix
        target = upload_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    with target.open("wb") as handle:
        handle.write(uploaded.getbuffer())

    return target


# Revenue analysis rendering
#
# Turns one end-to-end pipeline result into the view an accountant expects: the recognition
# conclusion first, then the judgments behind it, the schedule, the entries, the open items,
# and the memo. Nothing here computes an accounting figure - every number is taken from the
# agent output as-is, so the screen and the memo can never disagree.

# Validated chart palette (single-series blue) and the reserved status colours. Status is
# always paired with an icon and a word - colour never carries the meaning on its own.
SERIES = "#2a78d6"
GRID = "#e1e0d9"
AXIS_INK = "#898781"

SEVERITY = {
    "CRITICAL": ("#d03b3b", "●", "Critical"),
    "HIGH": ("#ec835a", "●", "High"),
    "MEDIUM": ("#fab219", "●", "Medium"),
    "LOW": ("#0ca30c", "●", "Low"),
}

DISPOSITION_TONE = {
    "proceed": "#0ca30c",
    "review_before_close": "#fab219",
    "review_before_recognition": "#ec835a",
    "escalate": "#d03b3b",
    "cannot_assess": "#d03b3b",
}

# The interpretive fields. These are the calls a rule-based extractor cannot make: each one
# needs the contract read against the standard and the entity policy. Python owns every
# figure; these judgments are what the language model contributes.
JUDGMENT_LABELS = {
    "onboarding_distinct": (
        "Is onboarding a distinct performance obligation?",
        "Decides whether the onboarding fee is recognised separately at a point in time or "
        "bundled into the subscription and spread over the term.",
    ),
    "usage_model": (
        "How is usage-based consideration structured?",
        "Determines whether usage is variable consideration, a right to invoice, or a "
        "sales-based royalty - each recognised differently.",
    ),
    "discount_type": (
        "What kind of discount is present?",
        "Drives whether a discount is allocated pro-rata across obligations or directed to "
        "one of them.",
    ),
    "material_right": (
        "Does the renewal option create a material right?",
        "A renewal priced below standalone value is a separate performance obligation that "
        "must carry part of the transaction price.",
    ),
    "contract_valid": (
        "Does an enforceable contract exist?",
        "ASC 606 does not apply until the criteria in 606-10-25-1 are met.",
    ),
    "additional_distinct_services": (
        "Are there other distinct services?",
        "Separately priced and separately purchasable services are their own performance "
        "obligations.",
    ),
}

CONCLUSION_WORDS = {
    "TRUE": "Yes",
    "FALSE": "No",
    "n/a": "Not applicable",
    "UNDETERMINED": "Could not be determined",
    "none": "None",
}


def money(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "-"


def money_short(value: Any) -> str:
    try:
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return "-"


def fmt_date(value: Any) -> str:
    if not value:
        return "-"
    try:
        return pd.to_datetime(value).strftime("%m/%d/%Y")
    except (ValueError, TypeError):
        return str(value)


def humanize(name: Any) -> str:
    text = str(name or "").replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else "-"


def asc_text(text: Any) -> str:
    """Replace KB ids with the ASC paragraph they stand for, so no internal id reaches the
    screen. Repeated paragraphs collapse - two knowledge bases often cite the same one."""
    if not isinstance(text, str):
        text = (
            " ".join(str(x) for x in text)
            if isinstance(text, (list, tuple))
            else str(text or "")
        )

    def repl(match: re.Match) -> str:
        kb = match.group(0).upper()
        ref = ASC_MAP.get(kb)
        return f"ASC {ref}" if ref else KB_LABEL.get(kb, "the review checklist")

    out = re.sub(r"KB[123]_\d+", repl, text)
    return re.sub(r"(ASC [^,;()]+)(?:, \1)+", r"\1", out)


def asc_refs(kb_ids: Any) -> list[str]:
    """Dereference a kb_basis list to unique ASC paragraphs, order preserved."""
    seen: list[str] = []
    for kb in kb_ids or []:
        ref = ASC_MAP.get(str(kb).upper())
        label = f"ASC {ref}" if ref else KB_LABEL.get(str(kb).upper())
        if label and label not in seen:
            seen.append(label)
    return seen


def conclusion_text(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(humanize(v) for v in value) if value else "None"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return money(value)
    return CONCLUSION_WORDS.get(str(value), humanize(value))


def badge(color: str, text: str) -> str:
    return (
        f"<span style='display:inline-block;padding:0.15rem 0.6rem;border-radius:999px;"
        f"border:1px solid {color}55;background:{color}1f;font-weight:600;font-size:0.85rem;'>"
        f"{text}</span>"
    )


def render_headline(result: dict[str, Any]) -> None:
    """The conclusion, before any of the evidence for it."""
    agent3 = result.get("agent3_output") or {}
    agent5 = result.get("agent5_output") or {}
    treatment = agent5.get("treatment") or agent3.get("treatment") or {}
    disposition = agent5.get("disposition") or result.get("disposition") or ""

    # Clear sits beside the contract name so the screen can be emptied before a demo starts.
    title_col, clear_col = st.columns([6, 1])
    title_col.markdown(
        f"### {agent5.get('customer_name') or 'Contract'} "
        f"<span style='font-size:0.95rem;opacity:0.7;'>"
        f"&nbsp;{agent5.get('contract_id') or ''}</span>",
        unsafe_allow_html=True,
    )
    if clear_col.button("Clear", width="stretch", key="clear_analysis"):
        st.session_state.pop("analysis_result", None)
        st.rerun()

    st.markdown(
        badge(
            DISPOSITION_TONE.get(disposition, "#898781"),
            DISPOSITION_TEXT.get(disposition, humanize(disposition)),
        ),
        unsafe_allow_html=True,
    )
    st.write("")

    cols = st.columns(4)
    cols[0].metric(
        "Transaction price",
        money_short(treatment.get("transaction_price")),
    )
    cols[1].metric(
        "Performance obligations",
        treatment.get("num_POs", "-"),
    )

    months = treatment.get("recognition_period_months")
    cols[2].metric(
        "Recognition period",
        f"{months} months" if months else "Not determined",
    )

    rows = (agent5.get("schedule") or {}).get("rows") or []
    monthly = rows[0].get("revenue_recognised") if rows else None
    cols[3].metric(
        "Monthly revenue",
        money_short(monthly) if monthly else "-",
    )

    method = str(treatment.get("recognition_method") or "")
    if method:
        pretty = (
            "Could not be determined"
            if method == "UNDETERMINED"
            else method.replace("_", " ")
        )
        st.caption(f"Recognition method — {pretty}")

    po_list = treatment.get("po_list") or []
    if po_list:
        st.caption(
            "Performance obligations identified — "
            + ", ".join(humanize(p) for p in po_list)
        )


def render_contract_terms(result: dict[str, Any]) -> None:
    agent1 = result.get("agent1_output") or {}
    facts = (
        agent1.get("validated_output")
        or agent1.get("candidate_output")
        or {}
    )

    left, right = st.columns(2)

    with left:
        st.markdown("**Parties and term**")
        st.table(
            pd.DataFrame(
                [
                    ("Customer", facts.get("customer_name") or "-"),
                    ("Provider", facts.get("provider_name") or "-"),
                    ("Effective date", fmt_date(facts.get("effective_date"))),
                    ("Service start", fmt_date(facts.get("start_date"))),
                    (
                        "Subscription term",
                        f"{facts.get('subscription_term_months')} months"
                        if facts.get("subscription_term_months")
                        else "Not stated",
                    ),
                    (
                        "Hosted service",
                        "Yes" if facts.get("hosted_service") else "No",
                    ),
                ],
                columns=["Term", "Value"],
            ).set_index("Term")
        )

    with right:
        st.markdown("**Fees**")
        fees = facts.get("fees") or []

        if fees:
            st.table(
                pd.DataFrame(
                    [
                        (
                            fee.get("name") or "-",
                            money((fee.get("amount") or {}).get("amount")),
                            humanize(fee.get("billing_frequency")),
                        )
                        for fee in fees
                    ],
                    columns=["Fee", "Amount", "Billing"],
                ).set_index("Fee")
            )
        else:
            st.caption("No fee lines extracted.")

    renewal = facts.get("renewal_terms") or {}
    termination = facts.get("termination_terms") or {}
    onboarding = facts.get("onboarding_terms") or {}

    with st.expander("Renewal, termination and onboarding clauses"):
        st.table(
            pd.DataFrame(
                [
                    (
                        "Auto-renewal",
                        "Yes" if renewal.get("auto_renewal") else "No",
                    ),
                    (
                        "Renewal pricing basis",
                        renewal.get("renewal_pricing_basis") or "Not stated",
                    ),
                    (
                        "Renewal notice",
                        f"{renewal.get('notice_days')} days"
                        if renewal.get("notice_days")
                        else "-",
                    ),
                    (
                        "Termination for breach",
                        "Yes" if termination.get("termination_for_breach") else "No",
                    ),
                    (
                        "Termination for convenience",
                        "Yes" if termination.get("convenience_termination") else "No",
                    ),
                    (
                        "Onboarding fee",
                        money((onboarding.get("fee") or {}).get("amount"))
                        if onboarding.get("fee")
                        else "None stated",
                    ),
                    (
                        "Onboarding required for access",
                        {True: "Yes", False: "No"}.get(
                            onboarding.get("required_for_platform"),
                            "Not stated",
                        ),
                    ),
                ],
                columns=["Clause", "Value"],
            ).set_index("Clause")
        )

    errors = agent1.get("validation_errors") or []
    if errors:
        st.warning(
            f"Schema validation raised {len(errors)} issue(s): "
            + "; ".join(str(error) for error in errors)
        )


def render_judgments(result: dict[str, Any]) -> None:
    """The interpretive calls - where the language model earns its place in the pipeline."""
    characterization = (result.get("agent3_output") or {}).get("characterization") or {}

    st.caption(
        "Each judgment below was reached by reading the contract against ASC 606 and the "
        "entity policy. Every figure on this page is computed in Python from the extracted "
        "facts — the model is never asked to do arithmetic."
    )

    for field, (question, why) in JUDGMENT_LABELS.items():
        node = characterization.get(field)

        if not isinstance(node, dict):
            continue

        reasoning = asc_text(node.get("reasoning") or "")
        is_rule = reasoning.startswith("Rule (not LLM)")
        refs = asc_refs(node.get("kb_basis"))

        with st.container(border=True):
            head_left, head_right = st.columns([4, 1])
            head_left.markdown(f"**{question}**")
            head_right.markdown(
                badge(
                    "#898781" if is_rule else SERIES,
                    "Rule" if is_rule else "Judgment",
                ),
                unsafe_allow_html=True,
            )

            st.markdown(f"### {conclusion_text(node.get('conclusion'))}")
            st.caption(why)

            if reasoning:
                st.markdown(reasoning.replace("Rule (not LLM): ", ""))

            if refs:
                st.caption("Authority relied on — " + "; ".join(refs))


def render_schedule(result: dict[str, Any]) -> None:
    schedule = (result.get("agent5_output") or {}).get("schedule") or {}
    rows = schedule.get("rows") or []

    if not rows:
        st.info(
            schedule.get("note")
            or "No recognition schedule was produced."
        )
        return

    frame = pd.DataFrame(rows)

    st.dataframe(
        pd.DataFrame(
            {
                "Month": frame["month"],
                "Period": [
                    f"{fmt_date(start)} – {fmt_date(end)}" if start else "-"
                    for start, end in zip(
                        frame["period_start"],
                        frame["period_end"],
                    )
                ],
                "Billed": frame["billed"].map(money),
                "Revenue recognised": frame["revenue_recognised"].map(money),
                "Deferred balance": frame["closing_balance"].map(money),
            }
        ).set_index("Month"),
        width="stretch",
    )

    totals = st.columns(3)
    totals[0].metric("Total billed", money_short(frame["billed"].sum()))
    totals[1].metric(
        "Total recognised",
        money_short(frame["revenue_recognised"].sum()),
    )
    totals[2].metric(
        "Closing deferred",
        money_short(frame["closing_balance"].iloc[-1]),
    )

    # Two single-series panels rather than one dual-axis chart: monthly revenue and the
    # deferred balance are different magnitudes, and a second y-scale would misrepresent both.
    st.markdown("**Revenue recognised by month**")
    st.altair_chart(
        alt.Chart(frame)
        .mark_bar(
            color=SERIES,
            cornerRadiusTopLeft=4,
            cornerRadiusTopRight=4,
            size=18,
        )
        .encode(
            x=alt.X(
                "month:O",
                title="Month",
                axis=alt.Axis(labelColor=AXIS_INK, titleColor=AXIS_INK),
            ),
            y=alt.Y(
                "revenue_recognised:Q",
                title="Revenue ($)",
                axis=alt.Axis(
                    labelColor=AXIS_INK,
                    titleColor=AXIS_INK,
                    gridColor=GRID,
                ),
            ),
            tooltip=[
                alt.Tooltip("month:O", title="Month"),
                alt.Tooltip(
                    "revenue_recognised:Q",
                    title="Recognised",
                    format="$,.2f",
                ),
            ],
        )
        .properties(height=200),
        width="stretch",
    )

    st.markdown("**Deferred revenue run-off**")
    st.altair_chart(
        alt.Chart(frame)
        .mark_line(
            color=SERIES,
            strokeWidth=2,
            point=alt.OverlayMarkDef(color=SERIES, size=60),
        )
        .encode(
            x=alt.X(
                "month:O",
                title="Month",
                axis=alt.Axis(labelColor=AXIS_INK, titleColor=AXIS_INK),
            ),
            y=alt.Y(
                "closing_balance:Q",
                title="Deferred balance ($)",
                axis=alt.Axis(
                    labelColor=AXIS_INK,
                    titleColor=AXIS_INK,
                    gridColor=GRID,
                ),
            ),
            tooltip=[
                alt.Tooltip("month:O", title="Month"),
                alt.Tooltip(
                    "closing_balance:Q",
                    title="Deferred balance",
                    format="$,.2f",
                ),
            ],
        )
        .properties(height=200),
        width="stretch",
    )


def render_journal_entries(result: dict[str, Any]) -> None:
    entries = (result.get("agent5_output") or {}).get("journal_entries") or []

    if not entries:
        st.info("No journal entries were produced; see the schedule note.")
        return

    for entry in entries:
        st.markdown(f"**{entry.get('when') or 'Entry'}**")

        lines = []
        for line in entry.get("lines") or []:
            amount = (
                money(line["amount"])
                if line.get("amount") is not None
                else "As incurred"
            )
            if line.get("dr"):
                lines.append(("Dr", line["dr"], amount, ""))
            else:
                lines.append(("Cr", f"     {line.get('cr')}", "", amount))

        st.table(
            pd.DataFrame(
                lines,
                columns=["", "Account", "Debit", "Credit"],
            ).set_index("")
        )

        if entry.get("narrative"):
            st.caption(entry["narrative"])


def render_review(result: dict[str, Any]) -> None:
    agent4 = result.get("agent4_output") or {}
    agent5 = result.get("agent5_output") or {}

    points = agent5.get("review_points") or []
    immaterial = agent5.get("immaterial_matters") or []
    consistency = agent4.get("consistency_issues") or []
    revisions = agent4.get("revision_requests") or []
    assessed = agent5.get("items_assessed") or 0

    st.caption(
        f"The review checklist was applied in full. {len(points)} matter(s) require "
        f"follow-up, {len(immaterial)} were quantified and concluded immaterial, and "
        f"{max(assessed - len(points) - len(immaterial), 0)} were concluded not applicable."
    )

    if consistency or revisions:
        st.markdown("**Cross-layer checks**")
        st.caption(
            "Raised by the review agent, which is the only stage holding every upstream "
            "output at once — these are contradictions the treatment layer cannot see in "
            "its own result."
        )

        for issue in consistency:
            st.warning(
                f"{humanize(issue.get('type'))} — {issue.get('detail')}"
            )

        for request in revisions:
            st.warning(
                f"Revision requested on {humanize(request.get('field'))} — "
                f"{request.get('reason')}"
            )

    if not points:
        st.success(
            "No matter was concluded to require further procedures before recognition."
        )

    for point in points:
        color, mark, word = SEVERITY.get(
            str(point.get("severity")),
            ("#898781", "●", "Note"),
        )

        with st.container(border=True):
            st.markdown(
                f"<span style='color:{color};'>{mark}</span> **{word}** &nbsp; "
                f"{humanize(point.get('name'))}",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Finding.** {asc_text(point.get('finding'))}")
            st.markdown(f"**Question.** {asc_text(point.get('review_question'))}")
            st.markdown(
                f"**Recommended procedure.** {asc_text(point.get('action'))}"
            )

            meta = [f"Timing: {point.get('timing') or '-'}"]
            if point.get("exposure") is not None:
                meta.append(f"Exposure: {money(point.get('exposure'))}")
            if point.get("materiality_threshold") is not None:
                meta.append(
                    "Materiality threshold: "
                    f"{money(point.get('materiality_threshold'))}"
                )
            meta.append(f"Status: {humanize(point.get('disposition'))}")

            st.caption(" · ".join(meta))

    if immaterial:
        with st.expander(f"Matters concluded immaterial ({len(immaterial)})"):
            for matter in immaterial:
                st.markdown(
                    f"**{humanize(matter.get('name'))}** — "
                    f"{asc_text(matter.get('finding'))}"
                )
                if matter.get("exposure") is not None:
                    st.caption(
                        f"Exposure {money(matter.get('exposure'))} against a threshold "
                        f"of {money(matter.get('materiality_threshold'))}."
                    )


def memo_pdf(memo: str) -> bytes:
    """Typeset the memo as a PDF. The memo is fixed-width by design - its rules, columns and
    schedule alignment only hold in a monospaced face - so it is laid out in Courier at a size
    that fits the widest line rather than reflowed."""
    import fitz
    import textwrap

    margin = 48.0
    size = 8.0
    page_width, page_height = fitz.paper_size("letter")

    # Courier advances 0.6 em per character, which fixes how many fit across the text block.
    # The size is held constant and the rare over-long line is wrapped instead: shrinking to
    # the widest line would drive the whole memo down to an unreadable few points.
    columns = int((page_width - 2 * margin) / (0.6 * size))

    lines: list[str] = []
    for raw in memo.splitlines() or [""]:
        if len(raw) <= columns:
            lines.append(raw)
            continue
        indent = " " * (len(raw) - len(raw.lstrip()))
        lines.extend(
            textwrap.wrap(
                raw,
                width=columns,
                subsequent_indent=indent + "  ",
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [raw]
        )

    leading = size * 1.25
    per_page = max(int((page_height - 2 * margin) / leading), 1)

    document = fitz.open()
    for start in range(0, len(lines), per_page):
        page = document.new_page(width=page_width, height=page_height)
        for offset, line in enumerate(lines[start : start + per_page]):
            if not line.strip():
                continue
            page.insert_text(
                fitz.Point(margin, margin + size + offset * leading),
                line,
                fontname="cour",
                fontsize=size,
            )

    data = document.tobytes()
    document.close()
    return data


def render_memo(result: dict[str, Any]) -> None:
    memo = result.get("memo") or ""

    if not memo:
        st.info("No memo was produced for this contract.")
        return

    contract_id = (
        (result.get("agent5_output") or {}).get("contract_id")
        or "contract"
    )

    left, right = st.columns(2)
    left.download_button(
        "Download memo (.pdf)",
        data=memo_pdf(memo),
        file_name=f"{contract_id}_revenue_memo.pdf",
        mime="application/pdf",
        width="stretch",
    )
    right.download_button(
        "Download full analysis (.json)",
        data=json.dumps(result, indent=2, ensure_ascii=False),
        file_name=f"{contract_id}_analysis.json",
        mime="application/json",
        width="stretch",
    )

    # Fixed-width: the memo is laid out with column alignment that markdown would destroy.
    st.code(memo, language=None)


def render_authority(result: dict[str, Any]) -> None:
    basis = (result.get("agent5_output") or {}).get("basis") or {}

    rows = []
    for group, label in (
        ("authoritative", "ASC 606"),
        ("entity_policy", "Entity policy"),
        ("review_checklist", "Review checklist"),
    ):
        for item in basis.get(group) or []:
            ref = item.get("asc_ref")
            name = KB_LABEL.get(str(item.get("kb_id", "")).upper(), "")
            rows.append((label, f"ASC {ref}" if ref else "-", humanize(name)))

    if not rows:
        st.caption("No authority was recorded for this contract.")
        return

    st.table(
        pd.DataFrame(
            rows,
            columns=["Source", "Paragraph", "Topic"],
        ).set_index("Source")
    )


def render_agent_trace(result: dict[str, Any]) -> None:
    """Every raw agent payload, kept in one place so it stays out of the analysis views."""
    st.caption(
        "Raw agent output, for verification. Agent 1 extracts, Agent 2 retrieves policy, "
        "Agent 3 determines the treatment, Agent 4 reviews it against the checklist, "
        "Agent 5 writes the memo."
    )

    labels = {
        "agent1_output": "Agent 1 — extraction",
        "agent2_output": "Agent 2 — policy retrieval",
        "agent3_output": "Agent 3 — accounting treatment",
        "agent4_output": "Agent 4 — checklist review",
        "agent5_output": "Agent 5 — memo",
    }

    for key, label in labels.items():
        if key in result:
            with st.expander(label):
                st.json(result[key])


def render_analysis(result: dict[str, Any]) -> None:
    """Full analysis for one contract, in the order an accountant reads it."""
    render_headline(result)
    st.divider()

    sections = st.tabs(
        [
            "Memo",
            "Contract terms",
            "Accounting judgments",
            "Schedule & entries",
            "Review findings",
            "Authoritative Source",
            "Agent trace",
        ]
    )

    with sections[0]:
        render_memo(result)

    with sections[1]:
        render_contract_terms(result)

    with sections[2]:
        render_judgments(result)

    with sections[3]:
        render_schedule(result)
        st.divider()
        st.markdown("### Journal entries")
        render_journal_entries(result)

    with sections[4]:
        render_review(result)

    with sections[5]:
        render_authority(result)

    with sections[6]:
        render_agent_trace(result)

# Header
st.title("Rev-Lens")
st.subheader("ASC 606 SaaS Contract Analysis")

st.markdown(
    """
Revenue recognition for SaaS contracts under ASC 606, applied against the authoritative
guidance, the company's own revenue policy, and an audit review checklist.

Upload a contract to produce a position paper with the recognition schedule, the journal
entries, and the matters requiring follow-up.
"""
)

st.divider()

saved_paths = saved_result_paths()
source_options = ["Run a contract"] + (["History"] if saved_paths else [])
source = st.radio(
    "Source",
    source_options,
    horizontal=True,
    label_visibility="collapsed",
    key="analysis_source",
)

if source == "Run a contract":
    contract_file = st.file_uploader(
        "Upload a contract (PDF or text)",
        type=["pdf", "txt"],
        key="analysis_upload",
    )

    if contract_file and st.button("Run full analysis", type="primary"):
        with st.status("Running the pipeline…", expanded=True) as status_box:
            try:
                st.write("Saving the contract…")
                saved_path = save_uploaded_contract(contract_file)

                st.write("Loading the knowledge base and retrieval index…")
                pipeline = get_pipeline()

                st.write("Extracting, retrieving, determining treatment, reviewing, drafting…")
                analysis = pipeline.run_contract(str(saved_path))

                # Kept so this contract can be reopened after another one is analysed.
                st.write("Saving the analysis to history…")
                save_analysis(analysis)
                st.session_state["analysis_result"] = analysis

                status_box.update(label="Analysis complete", state="complete", expanded=False)
            except Exception as error:
                status_box.update(label="The pipeline could not complete", state="error")
                st.error(str(error))
else:
    st.caption(
        f"{len(saved_paths)} saved analysis(es). Every completed run is kept here, newest "
        "first, and opens without calling the model."
    )
    chosen = st.selectbox(
        "Saved analysis",
        saved_paths,
        format_func=analysis_label,
        key="analysis_saved_choice",
        label_visibility="collapsed",
    )
    if chosen and st.button("Open", type="primary"):
        st.session_state["analysis_result"] = load_json_file(str(chosen))

result = st.session_state.get("analysis_result")

if result:
    st.divider()
    render_analysis(result)
else:
    st.info(
        "Upload a contract and run the analysis, or reopen one from History. "
        "A live run calls the language model for the interpretive judgments."
    )
