import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

def run_review(messages, max_tokens=4000):
    response = client.chat.completions.create(
        model=os.getenv("OLLAMA_MODEL"),
        messages=messages,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Can you help me with a question?"}
    ]
    result = run_review(messages)
    print(result)