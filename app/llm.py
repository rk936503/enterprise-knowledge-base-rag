from anthropic import Anthropic

from app.config import get_settings

settings = get_settings()
_client: Anthropic | None = None

SYSTEM_PROMPT = """You are a knowledge base assistant. Answer the user's question using ONLY \
the provided context chunks. If the context does not contain enough information to answer, \
say so explicitly instead of guessing. Cite which chunk(s) you used by number."""


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=settings.anthropic_api_key)
    return _client


def generate_answer(question: str, context_chunks: list[str]) -> str:
    numbered_context = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks))
    user_message = f"Context:\n{numbered_context}\n\nQuestion: {question}"

    response = _get_client().messages.create(
        model=settings.llm_model,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text