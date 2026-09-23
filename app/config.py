import os
from functools import lru_cache

class Settings:
    #Database
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://rag_user:rag_password@localhost:5432/rag_db",
    )

    #Embedding provider
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "openai")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "1536"))

    #LLM provider
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    llm_model: str == os.getenv("LLM_MODEL", "claude-sonnet-4-6")

    #Chunking
    chunk_size_tokens: int = int(os.getenv("CHUNK_SIZE_TOKENS", "400"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))

    #Retrieval
    top_k: int  int(os.getenv("TOP_K", "5"))

@lru_cache
def get_settings() -> Settings:
    return Settings()