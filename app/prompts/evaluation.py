"""
Prompts for Evaluation Agent 5 - Complex KB3 Validation Reasoning

Contains few-shot examples and system prompts for validating Agent 3 decisions
against KB3 risks using LLM reasoning when heuristic rules are insufficient.
"""

SYSTEM_PROMPT = """You are an expert ASC 606 revenue recognition evaluator. Your task is to assess whether Agent 3's treatment decisions properly address the revenue recognition risks outlined in the KB3 review checklist.

For each KB3 risk item presented:
1. Identify: What is the risk and what concern does it raise?
2. Assess: Did Agent 3 make a decision relevant to this risk?
3. Validate: Is Agent 3's conclusion supported by sound reasoning?
4. Conclude: PASS (risk properly mitigated), FAIL (risk not mitigated), or UNDETERMINED_ACCEPTABLE (caution appropriate given facts)

Output your evaluation in JSON format with these fields:
{
  "kb3_id": "KB3_X",
  "risk_identified": boolean,
  "agent3_addressed": boolean,
  "conclusion_valid": boolean,
  "status": "PASS|FAIL|UNDETERMINED_ACCEPTABLE",
  "confidence": 0-100,
  "reasoning": "explanation",
  "flags": ["flag1", "flag2"]
}

CRITICAL GUIDELINES:
- PASS only if both addressed AND conclusion is sound
- FAIL if risk is identified but Agent 3 made wrong decision
- UNDETERMINED_ACCEPTABLE only if appropriate caution given insufficient facts
- Flag any inconsistencies with related decisions
"""

USER_PROMPT_COMPLEX = """
Evaluate the following KB3 risks and Agent 3 response.

═══════════════════════════════════════════════════════════════════════
CONTEXT: Contract Facts (from Agent 1)
═══════════════════════════════════════════════════════════════════════
{facts_json}

═══════════════════════════════════════════════════════════════════════
AGENT 3 CHARACTERIZATION JUDGMENTS
═══════════════════════════════════════════════════════════════════════
{agent3_characterization}

═══════════════════════════════════════════════════════════════════════
AGENT 3 TREATMENT OUTCOMES
═══════════════════════════════════════════════════════════════════════
{agent3_treatment}

═══════════════════════════════════════════════════════════════════════
KB3 ITEMS TO EVALUATE
═══════════════════════════════════════════════════════════════════════
{kb3_items}

═══════════════════════════════════════════════════════════════════════
STEP 1: UNDERSTAND THE RISKS
═══════════════════════════════════════════════════════════════════════

For each KB3 item, the risk is the potential error:
- KB3_1 (Bundled Onboarding): May be wrongly separated or combined
- KB3_3 (Variable Consideration): Usage may be over-recognized or front-loaded
- KB3_7 (Discount Allocation): Bundle discount allocated to wrong obligations
- [etc.]

The trigger identifies contract characteristics that activate the risk.

═══════════════════════════════════════════════════════════════════════
STEP 2: EVALUATE AGENT 3'S RESPONSE
═══════════════════════════════════════════════════════════════════════

For each KB3 item triggered:

A) Did Agent 3 address it?
   - Made an explicit judgment (onboarding_distinct, usage_model, etc.)
   - Provided reasoning based on contract facts

B) Is the conclusion sound?
   - Reasoning cites relevant ASC 606 criteria (customization, dependency, etc.)
   - Not based on irrelevant factors (e.g., size alone)
   - Consistent with other treatment decisions

C) What's the confidence level?
   - 90%: Clear reasoning, well-supported
   - 70%: Reasonable but some uncertainty
   - 50%: Acceptable caution given facts
   - 30%: Missing reasoning or weak support
   - 10%: Contradicts best practices

═══════════════════════════════════════════════════════════════════════
STEP 3: CROSS-VALIDATE CONSISTENCY
═══════════════════════════════════════════════════════════════════════

Check that related decisions align:
- If onboarding_distinct=TRUE: po_list should include "onboarding" PO
- If material_right=TRUE: po_list should include "renewal_option" PO
- If usage_model≠"none": transaction_price must account for variable component
- If discount_type="residual": at least one PO must have observable SSP

═══════════════════════════════════════════════════════════════════════
FEW-SHOT EXAMPLES
═══════════════════════════════════════════════════════════════════════

EXAMPLE 1: KB3_1 PASS
Risk: Bundled onboarding may be wrongly separated or combined
Trigger: Contract shows onboarding fee + subscription fee

Contract Facts:
  - Onboarding Fee: $50,000
  - Subscription Fee: $100,000/year
  - Onboarding Description: "Includes custom integration and data migration; required before platform access"

Agent 3 Response:
  onboarding_distinct = TRUE
  Reasoning: "Integration is customized to customer environment; required before go-live; creates separate performance obligation per KB1_3"
  KB Basis: [KB1_3, KB2_2]

Evaluation:
  ✅ Risk identified: YES (separate onboarding fee)
  ✅ Agent 3 addressed: YES (explicit TRUE judgment)
  ✅ Conclusion sound: YES (reasoning cites customization + dependency)
  ✅ Consistency: po_list includes "onboarding"; recognition_method shows point-in-time
  Status: PASS (90% confidence)
  Flags: None

---

EXAMPLE 2: KB3_1 FAIL
Risk: Bundled onboarding may be wrongly separated or combined
Trigger: Contract shows onboarding fee + subscription

Contract Facts:
  - Onboarding Fee: $50,000
  - Subscription Fee: $100,000/year
  - Onboarding Description: (vague) "Implementation services"

Agent 3 Response:
  onboarding_distinct = TRUE
  Reasoning: "The onboarding fee is large"
  KB Basis: []

Evaluation:
  ✅ Risk identified: YES
  ✅ Agent 3 addressed: YES
  ❌ Conclusion sound: NO (reasoning cites size only, violates KB3_2)
  ❌ Consistency: po_list includes "onboarding" but sizing decision not justified
  Status: FAIL
  Flags: ["Reasoning cites size only, not nature of service", "KB3_2 violated"]
  Recommendation: Re-evaluate: Is onboarding truly customized per customer?

---

EXAMPLE 3: KB3_10 UNDETERMINED_ACCEPTABLE
Risk: Discounted renewal may hide separate performance obligation
Trigger: Contract grants renewal option but pricing not stated

Contract Facts:
  - Base Term: 3 years at $100,000/year
  - Renewal: "Right to renew at then-current market rates"
  - Renewal Price: (not specified)

Agent 3 Response:
  material_right = UNDETERMINED
  Reasoning: "Contract mentions renewal but price not specified; cannot assess if below market without pricing"
  KB Basis: []

Evaluation:
  ✅ Risk identified: YES (renewal option exists)
  ✅ Agent 3 addressed: YES (made explicit judgment)
  ✅ Conclusion sound: YES (UNDETERMINED appropriate given missing data)
  ✅ Consistency: po_list doesn't include "renewal_option" (correct for UNDETERMINED)
  Status: UNDETERMINED_ACCEPTABLE (70% confidence)
  Flags: ["ESCALATION: Renewal pricing must be obtained before period-end"]
  Recommendation: Contact customer for renewal pricing terms

---

EXAMPLE 4: KB3_3 UNDETERMINED_PROBLEMATIC
Risk: Usage/overage may be misrecognized
Trigger: Contract mentions transaction-based or usage-based fees

Contract Facts:
  - Base Fee: $100,000/year
  - Usage: "Additional charges for usage above 100,000 transactions/month"
  - Usage Rate: $0.01 per transaction above threshold

Agent 3 Response:
  usage_model = "none"
  Reasoning: (minimal)
  KB Basis: []

Evaluation:
  ✅ Risk identified: YES (usage fees exist)
  ✅ Agent 3 addressed: YES (made judgment)
  ❌ Conclusion sound: NO (usage fees exist but marked as "none")
  ❌ Consistency: transaction_price = $100k only; doesn't reflect overage structure
  Status: FAIL (or UNDETERMINED_PROBLEMATIC if data extraction unclear)
  Flags: ["Usage fees exist but usage_model incorrectly set to 'none'", "transaction_price doesn't account for variable component"]
  Recommendation: Correct usage_model to "mixed"; separate fixed ($100k) from variable

═══════════════════════════════════════════════════════════════════════
NOW EVALUATE THE FOLLOWING ITEMS:
═══════════════════════════════════════════════════════════════════════

For each KB3 item, output JSON evaluation following the format above.
Return an array of evaluations:

[
  {
    "kb3_id": "KB3_1",
    "status": "PASS|FAIL|UNDETERMINED_ACCEPTABLE",
    "confidence": 85,
    "reasoning": "...",
    "flags": [...]
  },
  ...
]
"""

# Simplified user prompt for deterministic validation (no LLM needed for most cases)
USER_PROMPT_SIMPLE = """
Validate Agent 3's treatment decisions against KB3 risks.

Contract: {source_file}
Agent 3 Confidence Score: {agent3_confidence}

KB3 Items Flagged by Agent 4:
{predicted_kb3_ids}

Agent 3 Decisions:
  - onboarding_distinct: {onboarding_distinct}
  - usage_model: {usage_model}
  - discount_type: {discount_type}
  - material_right: {material_right}
  - num_POs: {num_POs}
  - recognition_method: {recognition_method}

For each flagged KB3 item:
1. Did Agent 3 address the risk?
2. Is the conclusion correct?
3. Are there consistency issues?

Output: JSON array with status (PASS/FAIL/UNDETERMINED_ACCEPTABLE) and confidence for each.
"""


# Few-shot examples for specific KB3 validation tasks
FEW_SHOT_ONBOARDING = """
TASK: Validate onboarding distinctness (KB3_1, KB3_2)

CRITERIA:
- TRUE if: Customized to customer, required before access, creates separate obligation
- FALSE if: Standardized, optional, available elsewhere, no separate performance obligation
- UNDETERMINED if: Insufficient contract detail to apply distinctness criteria

EXAMPLE 1 (PASS - KB3_1):
Trigger: "$50k setup + $100k annual subscription"
Facts: "Custom integration required before go-live; no setup charge if customer provides own integration"
Agent 3: onboarding_distinct = TRUE
Validation: ✅ Customized, ✅ Required dependency, ✅ Separate if provided by third party
Result: PASS (90% confidence)

EXAMPLE 2 (FAIL - KB3_2):
Trigger: "$50k setup + $50k annual subscription" (large fee)
Facts: No detail on customization
Agent 3: onboarding_distinct = TRUE
Reasoning: "The setup fee is large"
Validation: ❌ Size cited but not nature; KB3_2 violated
Result: FAIL - Need: Is setup really customized per customer?

EXAMPLE 3 (UNDETERMINED_ACCEPTABLE - KB3_17):
Trigger: Onboarding mentioned but not described
Agent 3: onboarding_distinct = UNDETERMINED
Reasoning: "Contract states 'onboarding included' but lacks detail on customization"
Validation: ✅ Caution appropriate; need more contract specificity
Result: UNDETERMINED_ACCEPTABLE - Action: Obtain onboarding scope details
"""

FEW_SHOT_USAGE = """
TASK: Validate usage/variable consideration (KB3_3-6)

CRITERIA:
- Usage must be separated from fixed fees
- Variable amount should be constrained (not over-estimated)
- If volatile/unestimable: recognize as-incurred, not front-loaded
- Distinguish royalty (IP license) from ordinary metered usage

EXAMPLE 1 (PASS - KB3_3):
Trigger: "$100k fixed + $0.50/transaction"
Agent 3: usage_model = "mixed"
Reasoning: "Fixed = $100k subscription; variable = transaction-based, capped at probable amount"
Validation: ✅ Separated, ✅ Recognized appropriately, ✅ Constrained
Result: PASS (85% confidence)

EXAMPLE 2 (FAIL - KB3_5):
Trigger: "Pure transaction-based, highly volatile, no minimum"
Agent 3: usage_model = "pure_usage"
Treatment: transaction_price = estimated_annual_usage * rate (front-loaded estimate)
Validation: ❌ Over-recognition of uncertain amount; should be as-incurred
Result: FAIL - Action: Recognize usage as invoiced, not estimated

EXAMPLE 3 (PASS - KB3_6):
Trigger: "Customers pay per-use of proprietary analytics library"
Agent 3: usage_model = "pure_usage"
Reasoning: "IP license royalty; separate from subscription"
Validation: ✅ Correctly identified as license vs. metered SaaS
Result: PASS (80% confidence)
"""

FEW_SHOT_MATERIAL_RIGHT = """
TASK: Validate material right / renewal option (KB3_10, KB3_18)

CRITERIA:
- Material right if: Renewal price < price to comparable customers
- Requires separate PO allocation if material
- If price unknown: UNDETERMINED appropriate (need to obtain)

EXAMPLE 1 (PASS - KB3_10):
Facts: 3-yr contract, renewal at $50k/yr vs. market rate $80k/yr (37.5% discount)
Agent 3: material_right = TRUE
Treatment: PO includes "renewal_option"; transaction_price reduced by allocated value
Validation: ✅ Below-market identified, ✅ Separate PO, ✅ Allocated
Result: PASS (90% confidence)

EXAMPLE 2 (UNDETERMINED_ACCEPTABLE - KB3_18):
Facts: "Renewal right at then-current rates" (no price specified)
Agent 3: material_right = UNDETERMINED
Validation: ✅ Appropriate given missing data; cannot assess materiality
Result: UNDETERMINED_ACCEPTABLE - Action: Obtain renewal pricing
"""

FEW_SHOT_DISCOUNT = """
TASK: Validate discount allocation method (KB3_7, KB3_8)

CRITERIA:
- Default: Pro-rata allocation across all POs
- Directed: Only if regular separate sales at comparable discount
- Residual: Only if one PO has observable SSP

EXAMPLE 1 (PASS - KB3_7):
Facts: 3-component bundle at $300k vs. $450k list
Agent 3: discount_type = "pro_rata"
Reasoning: "No evidence of separate comparable pricing"
Validation: ✅ Default method applied correctly
Result: PASS (95% confidence)

EXAMPLE 2 (FAIL - KB3_8):
Facts: Bundle with discount; onboarding component assigned as residual
Agent 3: discount_type = "residual"
Validation: ❌ Residual used but onboarding has no observable SSP
Result: FAIL - Action: Use pro-rata or source observable SSP for onboarding
"""

FEW_SHOT_BILLING = """
TASK: Validate billing and revenue timing (KB3_11-14)

CRITERIA:
- Upfront billing → opening_deferred > 0 (contract liability)
- In-arrears → opening_deferred ≈ 0
- Mid-month start → first period prorated
- Multi-year upfront → financing component considered

EXAMPLE 1 (PASS - KB3_11):
Facts: $100k billed upfront; 12-month term
Agent 3: billing_used = "annual_upfront"
Schedule: opening_deferred = $100k; monthly_revenue = $8,333
Validation: ✅ Deferred matches upfront billing, ✅ Released evenly
Result: PASS (95% confidence)

EXAMPLE 2 (PASS - KB3_12):
Facts: Term starts June 15
Agent 3: schedule[0] = {period: "Jun 15-30", revenue: $4,167 (half-month)}
Validation: ✅ First period prorated correctly
Result: PASS (90% confidence)

EXAMPLE 3 (UNDETERMINED - KB3_13):
Facts: $500k upfront for 5-year term
Agent 3: No financing component documented
Validation: ⚠️ Financing component should be analyzed (even if concluded immaterial)
Result: UNDETERMINED - Action: Document financing component analysis
"""
