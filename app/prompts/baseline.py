SYSTEM_PROMPT = """You are a helpful assistant that extracts structured information from SaaS contracts."""

USER_PROMPT = """
Analyze this SaaS contract and return structured JSON in the provided example format.

SaaS Contract:
{contract_text}

Structured JSON Example:
{{
    "customer_name": "Brightwater Asset Management, LLC",
    "term_months": 12,
    "start_date": "2025-01-01",
    "billing": "annual_upfront",
    "annual_subscription": 9600,
    "onboarding_fee": 0,
    "onboarding_distinct": "n/a",
    "usage_model": "none",
    "base_fee": 0,
    "usage_rate": 0,
    "discount_amount": 0,
    "discount_type": "none",
    "material_right": false,
    "num_POs": 1,
    "po_list": ["core_subscription"],
    "transaction_price": 9600,
    "recognition_method": "over_time_ratable",
    "recognition_period_months": 12,
    "monthly_revenue": 800,
    "opening_deferred": 9600
}}

Return only valid JSON. Do not include any explanation or text outside the JSON object.
"""