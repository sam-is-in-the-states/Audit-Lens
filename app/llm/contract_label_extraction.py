"""
RevenueLens Task 1 + Task 2 Starter Implementation

Task 1: Extract raw data from contract PDFs.
Task 2: Validate structured contract outputs with Pydantic.

Install if needed:
    pip install pymupdf pdfplumber pydantic

Run:
    python revenue_lens_extraction_validation.py /path/to/contracts/*.pdf --out extracted_contracts.json
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import fitz  # PyMuPDF
import pdfplumber
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator



# Task 1: Raw PDF extraction
SECTION_PATTERN = re.compile(r"(?m)^\s*(\d+(?:\.\d+)?)\.\s+([A-Z][A-Z &/]+)\s*$")

def extract_pdf_raw_data(pdf_path: str | Path) -> dict[str, Any]:
    """Extract raw text, page text, simple tables, and section-level text from a PDF."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[dict[str, Any]] = []
    all_text_parts: list[str] = []

    with fitz.open(pdf_path) as doc:
        metadata = dict(doc.metadata or {})
        page_count = doc.page_count
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            all_text_parts.append(text)
            pages.append(
                {
                    "page_number": i,
                    "text": normalize_text(text),
                    "word_count": len(text.split()),
                }
            )

    tables: list[dict[str, Any]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            for table_idx, table in enumerate(page.extract_tables() or [], start=1):
                clean_rows = [[normalize_text(cell or "") for cell in row] for row in table]
                if clean_rows:
                    tables.append(
                        {
                            "page_number": page_idx,
                            "table_number": table_idx,
                            "rows": clean_rows,
                        }
                    )

    full_text = normalize_text("\n".join(all_text_parts))

    return {
        "file_name": pdf_path.name,
        "file_path": str(pdf_path),
        "page_count": page_count,
        "metadata": metadata,
        "full_text": full_text,
        "sections": split_into_sections(full_text),
        "tables": tables,
        "pages": pages,
    }


def normalize_text(text: str) -> str:
    """Normalize whitespace without destroying line breaks needed for sections."""
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_sections(full_text: str) -> dict[str, str]:
    """Split contract text into numbered sections such as '3. SERVICES'."""
    matches = list(SECTION_PATTERN.finditer(full_text))
    sections: dict[str, str] = {}

    for idx, match in enumerate(matches):
        section_number = match.group(1)
        section_title = match.group(2).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        key = f"{section_number}. {section_title}"
        sections[key] = full_text[start:end].strip()

    return sections


# Task 2: Structured output schema + validation
class MoneyAmount(BaseModel):
    amount: float = Field(ge=0)
    currency: str = Field(default="USD", pattern=r"^[A-Z]{3}$")


class FeeItem(BaseModel):
    name: str
    amount: MoneyAmount | None = None
    billing_frequency: Literal[
        "one_time", "annual", "quarterly", "monthly", "usage_based", "unknown"
    ] = "unknown"
    payment_timing: str | None = None
    notes: str | None = None


class DiscountTerms(BaseModel):
    has_discount: bool | None = None
    discount_type: Literal["percentage", "fixed_amount", "waiver", "unknown"] = "unknown"
    percentage: float | None = Field(default=None, ge=0, le=100)
    amount: MoneyAmount | None = None
    applies_to: str | None = None
    duration: str | None = None
    notes: str | None = None


class UsageFee(BaseModel):
    included_units: int | None = Field(default=None, ge=0)
    unit_name: str | None = None
    overage_rate: MoneyAmount | None = None
    reconciliation_frequency: Literal["monthly", "quarterly", "annual", "unknown"] = "unknown"


class OnboardingTerms(BaseModel):
    required: bool | None = None
    fee: MoneyAmount | None = None
    billing_frequency: Literal["one_time", "monthly", "annual", "unknown"] = "unknown"
    required_for_platform: bool | None = None
    timeline: str | None = None
    notes: str | None = None


class RenewalTerms(BaseModel):
    auto_renewal: bool | None = None
    renewal_term_months: int | None = Field(default=None, ge=1)
    notice_days: int | None = Field(default=None, ge=0)
    renewal_price: MoneyAmount | None = None
    renewal_pricing_basis: str | None = None


class TerminationTerms(BaseModel):
    termination_for_breach: bool | None = None
    notice_days: int | None = Field(default=None, ge=0)
    convenience_termination: bool | None = None
    missing_terms: list[str] = Field(default_factory=list)


class ExtractedContract(BaseModel):
    contract_id: str = Field(pattern=r"^RL-\d{4}-\d{4}$")
    provider_name: str
    customer_name: str
    effective_date: date
    subscription_term_months: int | None = Field(default=None, ge=1)
    start_date: date | None = None
    end_date: date | None = None

    services_summary: str
    hosted_service: bool
    authorized_users: int | None = Field(default=None, ge=0)
    implementation_services: bool = False
    implementation_required_for_platform: bool | None = None

    fees: list[FeeItem] = Field(default_factory=list)
    discount_terms: DiscountTerms = Field(default_factory=DiscountTerms)
    usage_fee: UsageFee | None = None
    onboarding_terms: OnboardingTerms = Field(default_factory=OnboardingTerms)
    renewal_terms: RenewalTerms = Field(default_factory=RenewalTerms)
    termination_terms: TerminationTerms = Field(default_factory=TerminationTerms)

    ambiguous_clauses: list[str] = Field(default_factory=list)
    source_file: str | None = None
    raw_text_hash: str | None = None


    @field_validator("effective_date", "start_date", "end_date", mode="before")
    @classmethod
    def parse_human_dates(cls, value: Any) -> Any:
        if value is None or isinstance(value, date):
            return value
        if isinstance(value, str):
            value = value.strip()
            for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
                try:
                    return datetime.strptime(value, fmt).date()
                except ValueError:
                    pass
        return value

    @field_validator("provider_name", "customer_name", "services_summary")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("field cannot be empty")
        return value.strip()

    @model_validator(mode="after")
    def validate_dates_and_term(self) -> "ExtractedContract":
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date cannot be before start_date")
        return self

    @model_validator(mode="after")
    def validate_fees_exist(self) -> "ExtractedContract":
        if not self.fees:
            raise ValueError("at least one fee item must be extracted")
        return self


def validate_structured_output(candidate: dict[str, Any]) -> tuple[ExtractedContract | None, list[dict[str, Any]]]:
    """Validate a candidate JSON/dict against the contract schema."""
    try:
        validated = ExtractedContract.model_validate(candidate)
        return validated, []
    except ValidationError as exc:
        return None, exc.errors(include_context=False)


# Optional: deterministic starter extraction baseline
# Replace this later with LLM output, then validate it.
MONTHS = {
    "twelve": 12,
    "12": 12,
    "twenty-four": 24,
    "24": 24,
}


def build_rule_based_candidate(raw: dict[str, Any]) -> dict[str, Any]:
    """A simple regex starter extractor for these synthetic contracts."""
    text = raw["full_text"]

    contract_id = search1(r"Order Form No\.\s*(RL-\d{4}-\d{4})", text)
    provider = search1(r"Provider:\s*([^,\n]+)", text) or "RevLens, Inc."
    customer = search1(r"Customer:\s*([^,\n]+(?:,\s*LLC|,\s*Inc\.|,\s*LLP|,\s*Corp\.)?)", text)
    effective_date = search1(r"([A-Z][a-z]+\s+\d{1,2},\s+\d{4})\s*\(the \"Effective Date\"\)", text)

    term_months = None
    term_match = re.search(r"(Twelve|twenty-four|\d+)\s*\(?\d*\)?\s+months", text, flags=re.I)
    if term_match:
        term_months = MONTHS.get(term_match.group(1).lower(), None)

    hosted_service = bool(re.search(r"hosted service|hosted, subscription basis|hosted subscription", text, re.I))
    authorized_users = search_int(r"up to ([\w-]+|\d+)\s*\(?\d*\)?\s+(?:named\s+)?authorized users", text)

    implementation_services = bool(re.search(r"Implementation Services|Implementation comprises", text, re.I))
    implementation_required = None
    if implementation_services:
        implementation_required = bool(re.search(r"required for the Platform to operate", text, re.I))

    fees = extract_fees(text, raw.get("tables", []))
    discount_terms = extract_discount_terms(text)
    usage_fee = extract_usage_fee(text)
    onboarding_terms = extract_onboarding_terms(text, fees)
    renewal_terms = extract_renewal_terms(text)
    termination_terms = extract_termination_terms(text)
    ambiguous = extract_ambiguous_clauses(text)

    return {
        "contract_id": contract_id,
        "provider_name": provider,
        "customer_name": customer,
        "effective_date": effective_date,
        "subscription_term_months": term_months,
        "start_date": effective_date,
        "end_date": search1(r"ending\s+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})", text),
        "services_summary": raw.get("sections", {}).get("3. SERVICES", "").split("\n")[0][:500] or "Hosted subscription service",
        "hosted_service": hosted_service,
        "authorized_users": authorized_users,
        "implementation_services": implementation_services,
        "implementation_required_for_platform": implementation_required,
        "fees": fees,
        "discount_terms": discount_terms,
        "usage_fee": usage_fee,
        "onboarding_terms": onboarding_terms,
        "renewal_terms": renewal_terms,
        "termination_terms": termination_terms,
        "ambiguous_clauses": ambiguous,
        "source_file": raw.get("file_name"),
    }


def extract_fees(text: str, tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fees: list[dict[str, Any]] = []

    # Prefer explicit table rows where available.
    for table in tables:
        for row in table.get("rows", []):
            joined = " ".join(row)
            if "$" in joined and "Total" not in joined:
                amount = money_from_text(joined)
                name = row[0] if row else "Fee"
                frequency = "annual" if "annual" in joined.lower() or "subscription" in joined.lower() else "one_time"
                if "implementation" in joined.lower():
                    frequency = "one_time"
                fees.append({"name": name, "amount": amount, "billing_frequency": frequency})

    # Fallback for simple Item/Amount blocks extracted as separate lines.
    if not fees:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for i in range(len(lines) - 1):
            name_line = lines[i]
            amount_line = lines[i + 1]
            if "$" in amount_line and re.search(r"Subscription|Implementation", name_line, re.I):
                amount = money_from_text(amount_line)
                frequency = "annual" if "subscription" in name_line.lower() else "one_time"
                fees.append({"name": name_line, "amount": amount, "billing_frequency": frequency})

    # Fallback regexes.
    if not fees:
        subscription_amount = money_from_text(search1(r"subscription fee is (\$[\d,]+)", text, flags=re.I) or "")
        if not subscription_amount:
            subscription_amount = money_from_text(search1(r"Subscription Fee\.\s*(\$[\d,]+)", text, flags=re.I) or "")
        if not subscription_amount:
            subscription_amount = money_from_text(search1(r"Base Subscription Fee\.\s*(\$[\d,]+)", text, flags=re.I) or "")
        if subscription_amount:
            billing_frequency = "annual"
            payment_timing = None
            if re.search(r"payable in advance|billed quarterly in advance", text, re.I):
                payment_timing = "in advance"
            fees.append(
                {
                    "name": "Platform Subscription",
                    "amount": subscription_amount,
                    "billing_frequency": billing_frequency,
                    "payment_timing": payment_timing,
                }
            )

    return fees


def extract_discount_terms(text: str) -> dict[str, Any]:
    """Extract discount terms such as percentage discounts, fixed discounts, and waived fees."""
    has_discount = bool(re.search(r"discount|reduction|reduced by|waiv(?:e|ed|er)|credit", text, re.I))
    if not has_discount:
        return {"has_discount": False}

    pct_raw = search1(r"(\d+(?:\.\d+)?)\s*%\s*(?:discount|reduction|off)", text, flags=re.I)
    amount = money_from_text(search1(r"(?:discount|credit|reduced by|reduction of)\s*(\$[\d,]+(?:\.\d+)?)", text, flags=re.I) or "")

    discount_type = "unknown"
    if pct_raw is not None:
        discount_type = "percentage"
    elif amount is not None:
        discount_type = "fixed_amount"
    elif re.search(r"waiv(?:e|ed|er)", text, re.I):
        discount_type = "waiver"

    applies_to = search1(r"(?:discount|reduction|credit|waiver).*?(?:applies to|against|for)\s+([^.;\n]+)", text, flags=re.I | re.S)
    duration = search1(r"(?:for|during)\s+the\s+([^.;\n]*(?:initial term|first year|first twelve|first 12|pilot period)[^.;\n]*)", text, flags=re.I)

    return {
        "has_discount": True,
        "discount_type": discount_type,
        "percentage": float(pct_raw) if pct_raw is not None else None,
        "amount": amount,
        "applies_to": applies_to,
        "duration": duration,
        "notes": first_sentence_matching(text, r"discount|reduction|reduced by|waiv(?:e|ed|er)|credit"),
    }


def extract_onboarding_terms(text: str, fees: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Extract onboarding / implementation setup terms."""
    required = bool(re.search(r"onboarding|required onboarding|implementation services|required for the Platform to operate", text, re.I))
    fee = None

    # Prefer a fee item already extracted from a table/fee section.
    for item in fees or []:
        if re.search(r"onboarding|implementation|setup", item.get("name", ""), re.I):
            fee = item.get("amount")
            break

    if fee is None:
        fee = money_from_text(search1(r"(?:onboarding|implementation|setup)\s+(?:fee|services)?[^$]{0,80}(\$[\d,]+(?:\.\d+)?)", text, flags=re.I | re.S) or "")

    return {
        "required": required if required else None,
        "fee": fee,
        "billing_frequency": "one_time" if fee else "unknown",
        "required_for_platform": bool(re.search(r"required for the Platform to operate", text, re.I)) if required else None,
        "timeline": search1(r"(?:onboarding|implementation)[^.;\n]{0,120}(?:within|during|over)\s+([^.;\n]+)", text, flags=re.I),
        "notes": first_sentence_matching(text, r"onboarding|implementation services|setup"),
    }


def extract_usage_fee(text: str) -> dict[str, Any] | None:
    if not re.search(r"Usage Fees|overage|processed account transactions", text, re.I):
        return None
    included_units = search_int(r"up to ([\w,]+|\d+)\s*\(?[\d,]*\)?\s+processed account transactions", text)
    overage = money_from_text(search1(r"billed at (\$[\d,.]+) per transaction", text, flags=re.I) or "")
    frequency = "quarterly" if re.search(r"quarterly basis|quarter-end", text, re.I) else "unknown"
    return {
        "included_units": included_units,
        "unit_name": "processed account transaction",
        "overage_rate": overage,
        "reconciliation_frequency": frequency,
    }


def extract_renewal_terms(text: str) -> dict[str, Any]:
    renewal_text = search1(r"RENEWAL\n(.*?)(?:\nTERMINATION|\nAgreed and accepted:|$)", text, flags=re.I | re.S) or ""
    return {
        "auto_renewal": bool(re.search(r"renews automatically|automatically renew", renewal_text, re.I)) if renewal_text else None,
        "renewal_term_months": 12 if re.search(r"twelve\s*\(12\)\s*month", renewal_text, re.I) else None,
        "notice_days": search_int(r"at least ([\w-]+|\d+)\s*\(?\d*\)?\s+days", renewal_text),
        "renewal_price": money_from_text(search1(r"(?:fixed price of|priced at|renewal price(?: is| of)?|renewal fee(?: is| of)?)\s*(\$[\d,]+(?:\.\d+)?)", renewal_text, flags=re.I) or ""),
        "renewal_pricing_basis": "then-current list pricing" if re.search(r"then-current list pricing", renewal_text, re.I) else None,
    }


def extract_termination_terms(text: str) -> dict[str, Any]:
    missing: list[str] = []
    if re.search(r"not specified notice requirements", text, re.I):
        missing.append("notice requirements")
    if re.search(r"early-termination charges", text, re.I):
        missing.append("early termination charges")
    if re.search(r"consequences of discontinuation", text, re.I):
        missing.append("consequences of discontinuation")

    return {
        "termination_for_breach": bool(re.search(r"uncured material breach", text, re.I)) if "TERMINATION" in text else None,
        "notice_days": search_int(r"uncured material breach on ([\w-]+|\d+)\s*\(?\d*\)?\s+days", text),
        "convenience_termination": bool(re.search(r"discontinue the arrangement at its discretion", text, re.I)),
        "missing_terms": missing,
    }


def extract_ambiguous_clauses(text: str) -> list[str]:
    clauses = []
    patterns = [
        r"Invoicing arrangements and the payment schedule will be established.*?agreed\.",
        r"Either party may discontinue the arrangement at its discretion\.",
        r"The parties have not specified notice requirements.*?discontinuation\.",
        r"formalize commercial terms following an initial evaluation period\.",
        r"Distinctness|not offered by RevLens on a standalone basis",
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, text, flags=re.I | re.S):
            clauses.append(normalize_text(m.group(0)))
    return clauses


def first_sentence_matching(text: str, pattern: str) -> str | None:
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence = normalize_text(sentence)
        if sentence and re.search(pattern, sentence, re.I):
            return sentence[:500]
    return None


def search1(pattern: str, text: str, flags: int = 0) -> str | None:
    match = re.search(pattern, text, flags)
    if not match:
        return None
    for group in match.groups():
        if group:
            return normalize_text(group)
    return normalize_text(match.group(1))


def search_int(pattern: str, text: str) -> int | None:
    value = search1(pattern, text, flags=re.I)
    if not value:
        return None
    return parse_int_words(value)


def parse_int_words(value: str) -> int | None:
    value = value.lower().replace(",", "").strip()
    if value.isdigit():
        return int(value)
    words = {
        "thirty": 30,
        "twenty-five": 25,
        "seventy": 70,
        "three thousand": 3000,
        "twelve": 12,
    }
    return words.get(value)


def money_from_text(value: str) -> dict[str, Any] | None:
    match = re.search(r"\$([\d,]+(?:\.\d+)?)", value or "")
    if not match:
        return None
    return {"amount": float(match.group(1).replace(",", "")), "currency": "USD"}



# CLI runner
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdfs", nargs="+", help="PDF files to process")
    parser.add_argument("--out", default="extracted_contracts.json")
    args = parser.parse_args()

    results = []
    for pdf in args.pdfs:
        raw = extract_pdf_raw_data(pdf)
        candidate = build_rule_based_candidate(raw)
        validated, errors = validate_structured_output(candidate)
        results.append(
            {
                "file_name": raw["file_name"],
                "raw_text_preview": raw["full_text"][:1000],
                "section_names": list(raw["sections"].keys()),
                "tables": raw["tables"],
                "candidate_output": candidate,
                "validated_output": validated.model_dump(mode="json") if validated else None,
                "validation_errors": errors,
            }
        )

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    for r in results:
        status = "VALID" if r["validated_output"] else "INVALID"
        print(f"{status}: {r['file_name']}")
        if r["validation_errors"]:
            print(json.dumps(r["validation_errors"], indent=2))


if __name__ == "__main__":
    main()