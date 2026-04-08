"""Configuration and shared dependencies."""
import uuid
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

session_id = str(uuid.uuid4())


def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.7) -> ChatOpenAI:
    """Get a configured ChatOpenAI instance."""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://www.dataexpert.io/api/v1/openai",
        streaming=True,
        default_headers={
            "X-Session-ID": session_id
        }
    )
