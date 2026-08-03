# Rev-Lens

Rev-Lens reviews SaaS contracts for revenue recognition under ASC 606. It tests whether a
multi-agent pipeline produces a more accurate and more auditable review than a single-prompt
LLM baseline.

A contract goes in as a PDF. What comes out is an ASC 606 position paper: the recognition
conclusion, the judgments it rests on with the paragraph relied on for each, a monthly revenue
schedule, the journal entries, and the matters an accountant has to follow up before close.

The division of labour is the point of the design. **Python computes every figure**; the language
model is asked only for the interpretive calls a rule cannot make — whether onboarding is a
distinct performance obligation, whether a renewal option creates a material right, how
usage-based consideration is structured. The model never does arithmetic and never writes a
number into the memo unchecked.

---

## Architecture

Five agents run in sequence. Each one's output is the next one's input.

| Agent | File | Role | LLM? |
|---|---|---|---|
| 1 | `app/llm/contract_label_extraction.py` | Parses the PDF and extracts the contract facts into a validated schema | No — regex + Pydantic |
| 2 | `app/llm/policy_retrieval.py` | Retrieves the applicable guidance from the three knowledge bases | TF-IDF retrieval + LLM rerank |
| 3 | `app/llm/treatment_agent.py` | Determines the accounting treatment: obligations, transaction price, recognition method, schedule | Judgments only; all arithmetic in Python |
| 4 | `app/llm/audit_agent.py` | Applies the review checklist to the contract text and cross-checks the layers above it | No — rule-based |
| 5 | `app/llm/memo_agent.py` | Produces the schedule, journal entries, materiality conclusions and the memo | Prose only, guarded against unsourced figures |

Agent 4 is the only stage holding every upstream output at once, so it is the only one that can
catch a contradiction between them. Agent 5's language model writes narrative and nothing else: a
whitelist rejects any figure that did not come from the deterministic layer, and the memo falls
back to a template if the guard trips.

**Knowledge bases** (`documents/knowledge_base/`)

- `KB1_asc606_guidance.jsonl` — ASC 606 guidance, keyed to paragraph references
- `KB2_revlens_policy.jsonl` — the entity's own revenue policy
- `KB3_review_checklist.jsonl` — the review checklist Agent 4 applies

---

## Requirements

- Python **3.12** (the code uses `X | Y` type syntax; 3.9 will not run it)
- An OpenAI API key

```bash
py -3.12 -m venv venv
venv\Scripts\Activate.ps1        # PowerShell; use source venv/bin/activate elsewhere
py -3.12 -m pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and fill it in:

```
OLLAMA_BASE_URL=https://api.openai.com
OLLAMA_MODEL=gpt-4.1-mini
OPENAI_API_KEY=sk-...
```

The `OLLAMA_*` names are historical. The project began on a local Ollama server running Qwen3
1.7B; that model proved too weak for the accounting judgments and the project moved to the
OpenAI API. `app/llm/client.py` speaks the OpenAI protocol and reads these three variables, so
pointing `OLLAMA_BASE_URL` at a local Ollama instance still works if you prefer to run a local
model — the variable names were left alone to avoid breaking teammates' local configuration.

---

## Running

### The application

```powershell
py -3.12 -m streamlit run app\UI\ui.py
```

Upload a contract, run the analysis, and read the memo. Every completed run is saved to
`analysis_history/`, so an earlier contract can be reopened without calling the model again.
The memo downloads as a PDF; the full analysis downloads as JSON.

### One contract from the command line

```powershell
py -3.12 -m app.pipeline.run_pipeline "app\llm\contracts\sub_onboarding\C11_CedarPoint.pdf" --out pipeline_result.json
```

Must be run from the repository root. `pipeline_result.json` in the root is a committed sample of
this output — the full five-agent trace for one contract, readable without an API key.

Programmatic use:

```python
from app.pipeline.run_pipeline import EndToEndPipeline

pipeline = EndToEndPipeline()                    # builds the retrieval index once
result = pipeline.run_contract("contract.pdf")

result["memo"]                # the finished memo
result["recognition_status"]  # the one-line conclusion
result["agent5_output"]       # schedule, journal entries, review points
```

---

## Data

50 synthetic SaaS contracts (`app/llm/contracts/`, C01–C50) spanning six kinds of arrangement —
plain subscriptions, bundled onboarding, usage-based pricing, discounts and material rights,
combinations, and deliberately ambiguous contracts where the correct answer is to refuse to
conclude. Labels are in `evaluation/GroundTruth.xlsx`; plain-text copies of the contracts are in
`evaluation/contract_txt/`.

---

## Evaluation

Each agent is scored against the ground truth by its own script. Run them from `evaluation/`:

```powershell
$env:PYTHONUTF8=1
py -3.12 evaluate_agent1.py           # extraction: field accuracy
py -3.12 evaluate_agent2.py           # retrieval: precision / recall / F1
py -3.12 evaluate_agent3.py           # treatment: per-field accuracy (add --run to call the model)
py -3.12 evaluate_agent4.py           # review: detection precision / recall / F1
py -3.12 evaluate_agent5.py           # memo: arithmetic integrity and the narrative guard
py -3.12 evaluate_baseline.py         # single-prompt baseline, for comparison
```

`evaluate_agent3.py` with no arguments re-scores the saved outputs without calling the model;
`--run` regenerates them.

The metric differs by agent because the task does. Agents 1 and 3 answer questions that have an
answer on every contract, so accuracy is meaningful and is reported both overall and on the
subset where the answer is not the common default. Agents 2 and 4 search for items that are rare
— roughly 1.4 checklist matters per contract out of 18 — where accuracy would be flattered by
saying nothing at all, so precision and recall are reported separately. Agent 4 is deliberately
tuned toward recall: a missed matter is a potential misstatement, a false alarm only costs
reviewer time.

Results are written to `evaluation/results/`. Reported figures are in the project report.

A limitation worth stating: the 50 contracts are synthetic and there is no held-out split for the
treatment layer, so its scores should be read as optimistic.

---

## Layout

```
app/
  llm/           the five agents, the LLM client, and the contract corpus
  pipeline/      run_pipeline.py - wires the agents end to end
  prompts/       prompt templates (method only; no accounting rules hardcoded)
  UI/            ui.py - the Streamlit application
  baseline/      the single-prompt baseline
  vector_db/     earlier ChromaDB experiment; the shipped retriever is TF-IDF
documents/
  knowledge_base/  KB1 guidance, KB2 policy, KB3 checklist
evaluation/
  evaluate_agent*.py, GroundTruth.xlsx, contract_txt/, results/
```
