import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Read the .env file and copy its lines into os.environ.
# This MUST run before the class below is defined (explained later).
load_dotenv(Path(__file__).with_name(".env"))


@dataclass(frozen=True)
class Settings:
    """Central place for every configurable value in the app."""

    app_name: str = "finance-assistant"
    llm_model: str = os.getenv(
        "GROQ_MODEL", os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    )
    api_key: str = os.getenv("GROQ_API_KEY", "")
    llm_base_url: str = "https://api.groq.com"
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_dir: str = "./chroma_db"
    max_context_messages: int = 10

    def validate(self) -> None:
        """Fail early with a clear message if the API key is missing."""
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is miss     ing. Add it to your .env file.")


@lru_cache
def get_settings() -> Settings:
    """Create Settings once and reuse the same object everywhere."""
    return Settings()


if __name__ == "__main__":
    s = get_settings()
    print(s.app_name, s.llm_model, s.temperature)