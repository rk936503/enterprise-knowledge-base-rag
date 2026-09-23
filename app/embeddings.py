from openai import OpenAI
from sentence_transformers import SentenceTransformer

from app.config import get_settings

settings = get_settings()
_openai_client: OpenAI | None = None
_local_model: SentenceTransformer | None = None


def _get_openai_client() -> OpenAI:  #lazy initialization defers that untill the first actual embedding call.
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client     #We want exactly one client instance reused across all requests, not a new one per request.

def _get_local_model() -> SentenceTransformer:
    global _local_model
    if _local_model is None:
        #on first run downloads the model weights once(~90MB), then caches
        # them locally — every run after that loads instantly from disk.
        _local_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _local_model

def embed_text(text: str) -> list[float]:
    """Returns a single embedding vector for one piece of text."""
    return embed_batch([text])[0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Returns embedding vectors for a batch of texts in one API call/pass."""
    if settings.embedding_provider == "local":
        vectors = _get_local_model().encode(texts)
        return vectors.tolist()
    elif settings.embedding_provider == "openai":
        response = _get_openai_client().embeddings.create(
                model=settings.embedding_model,
                input=texts,
            )
        return [item.embedding for item in response.data]
    else:
        raise NotImplementedError(f"Embedding provider '{settings.embedding_provider}' not implemented.")