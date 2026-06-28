import json

from app.baseline import services
from app.llm.client import get_llm_response
from app.prompts import baseline


def run_baseline_review(messages):
    contract_text = services.get_contract_details()
    messages = [
        {"role": "system", "content": baseline.SYSTEM_PROMPT},
        {"role": "user", "content": baseline.USER_PROMPT.format(contract_text=contract_text)}
    ]
    response = get_llm_response(messages)
    return json.loads(response)

if __name__ == "__main__":
    result = run_baseline_review([])
    print(json.dumps(result, indent=4))
