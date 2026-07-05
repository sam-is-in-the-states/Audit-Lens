import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.llm.contract_label_extraction import (
    extract_pdf_raw_data,
    build_rule_based_candidate,
    validate_structured_output,
)


st.set_page_config(
    page_title="Audit-Lens",
    page_icon="📄",
    layout="wide",
)

st.title("Audit-Lens")
st.subheader("Contract PDF Upload + Revenue Recognition Review")

st.markdown(
    """
Upload a contract PDF to extract raw contract data, generate structured labels,
validate the output, and produce a review memo.
"""
)

uploaded_file = st.file_uploader(
    "Upload a contract PDF",
    type=["pdf"],
)

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Run Contract Review", type="primary"):
        progress = st.progress(0)
        status = st.empty()

        try:
            # Save uploaded PDF
            upload_dir = (
                Path(__file__).resolve().parents[1]
                / "llm"
                / "contracts"
                / "uploaded_contracts"
            )
            upload_dir.mkdir(parents=True, exist_ok=True)

            filename = uploaded_file.name
            saved_pdf = upload_dir / filename

            # Prevent duplicate uploads from overwriting existing files
            counter = 1
            while saved_pdf.exists():
                stem = Path(filename).stem
                suffix = Path(filename).suffix
                saved_pdf = upload_dir / f"{stem}_{counter}{suffix}"
                counter += 1

            with open(saved_pdf, "wb") as f:
                f.write(uploaded_file.getbuffer())

            pdf_path = str(saved_pdf)

            # Progress updates
            progress.progress(20)
            status.info("Step 1/5: PDF uploaded and saved.")

            # Extract raw PDF data
            raw = extract_pdf_raw_data(pdf_path)

            progress.progress(40)
            status.info("Step 2/5: Raw PDF data extracted.")

            # Generate structured contract labels
            candidate = build_rule_based_candidate(raw)

            progress.progress(60)
            status.info("Step 3/5: Contract labels extracted.")

            # Validate structured output with Pydantic schema
            validated, errors = validate_structured_output(candidate)

            progress.progress(80)
            status.info("Step 4/5: Structured output validated.")

            # Build final result object
            result = {
                "file_name": raw["file_name"],
                "saved_pdf_path": str(saved_pdf),
                "raw_text_preview": raw["full_text"][:1000],
                "section_names": list(raw["sections"].keys()),
                "tables": raw["tables"],
                "candidate_output": candidate,
                "validated_output": (
                    validated.model_dump(mode="json")
                    if validated
                    else None
                ),
                "validation_errors": errors,
            }

            progress.progress(100)
            status.success("Review complete!")

            # Results
            st.divider()
            st.caption(f"Saved uploaded contract to: `{saved_pdf}`")

            col1, col2 = st.columns([1, 1])

            # Structured JSON output
            with col1:
                st.header("📦 Structured JSON")

                if validated:
                    st.success("Validation status: VALID")
                    st.json(result["validated_output"])
                else:
                    st.error("Validation status: INVALID")
                    st.json(result["candidate_output"])

                with st.expander("View validation errors"):
                    st.json(errors)

                st.download_button(
                    label="Download JSON Result",
                    data=json.dumps(result, indent=2),
                    file_name=f"{Path(uploaded_file.name).stem}_review.json",
                    mime="application/json",
                )

            # Human-readable results memo
            with col2:
                st.header("📝 Results Memo")

                output = result["validated_output"] or result["candidate_output"]

                memo = f"""
### Contract Review Memo

**File Name:** {result["file_name"]}

**Contract ID:** {output.get("contract_id", "N/A")}

**Provider:** {output.get("provider_name", "N/A")}

**Customer:** {output.get("customer_name", "N/A")}

**Effective Date:** {output.get("effective_date", "N/A")}

**Subscription Term:** {output.get("subscription_term_months", "N/A")} months

**Start Date:** {output.get("start_date", "N/A")}

**End Date:** {output.get("end_date", "N/A")}

**Hosted Service:** {output.get("hosted_service", "N/A")}

**Authorized Users:** {output.get("authorized_users", "N/A")}

**Implementation Services:** {output.get("implementation_services", "N/A")}
"""
                st.markdown(memo)

                st.subheader("Fees")
                st.json(output.get("fees", []))

                st.subheader("Discount Terms")
                st.json(output.get("discount_terms", {}))

                st.subheader("Usage Fee")
                st.json(output.get("usage_fee", {}))

                st.subheader("Renewal Terms")
                st.json(output.get("renewal_terms", {}))

                st.subheader("Termination Terms")
                st.json(output.get("termination_terms", {}))

                st.subheader("Ambiguous Clauses")
                st.json(output.get("ambiguous_clauses", []))

        except Exception as e:
            progress.progress(0)
            status.error("Review failed.")
            st.exception(e)