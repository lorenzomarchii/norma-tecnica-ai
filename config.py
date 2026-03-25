from pathlib import Path

from pydantic_settings import BaseSettings

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def _get_streamlit_secret(key: str) -> str:
    """Try to read a secret from Streamlit Cloud secrets."""
    try:
        import streamlit as st
        return st.secrets.get(key, "")
    except Exception:
        return ""


class Settings(BaseSettings):
    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    # Embeddings (local model, no API needed)
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # Retrieval
    vector_top_k: int = 10
    bm25_top_k: int = 10
    final_top_k: int = 8
    max_cross_ref_expansion: int = 4

    # ChromaDB
    chroma_persist_dir: str = str(DATA_DIR / "chroma_db")

    # App
    max_upload_size_mb: int = 100
    app_title: str = "NormaTecnica AI"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Fallback to Streamlit secrets if .env is not available
        if not self.anthropic_api_key:
            self.anthropic_api_key = _get_streamlit_secret("ANTHROPIC_API_KEY")


settings = Settings()
