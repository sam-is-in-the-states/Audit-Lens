import json

from openai import OpenAI
from app.config.settings import settings

client = OpenAI(
    api_key=settings.OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

def run_review(messages, model, max_tokens=4000):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens
    )
    return json.loads(response.choices[0].message.content)