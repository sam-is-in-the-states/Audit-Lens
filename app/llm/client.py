import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL")+"/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)

def get_llm_response(messages, max_tokens=4000, response_format="text", temperature=0.0):
    updated_messages = []
    for msg in messages:
        if msg["role"] == "system":
            updated_messages.append({
                "role": "system",
                "content": msg["content"] + "\n/no_think"
            })
        else:
            updated_messages.append(msg)
    
    response = client.chat.completions.create(
        model=os.getenv("OLLAMA_MODEL"),
        messages=updated_messages,
        max_tokens=max_tokens,
        response_format={"type": response_format},
        temperature=temperature,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Can you help me with a question?"}
    ]
    result = get_llm_response(messages)
    print(result)