from dotenv import load_dotenv
import os
load_dotenv()

class Settings:
    OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY")
    MODEL_NAME=os.getenv("MODEL_NAME","qwen/qwen3-8b")
    CHROMA_PATH=os.getenv("CHROMA_PATH")

settings=Settings()
