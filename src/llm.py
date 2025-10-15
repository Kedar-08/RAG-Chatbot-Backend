from typing import List, Tuple
from src.config import settings
from google import genai

client = genai.Client(api_key=settings.GOOGLE_API_KEY)

SYSTEM = (
    "You are a helpful assistant that must answer ONLY using the provided context. "
    "If the answer is not completely supported by the context, reply exactly with: "
    "'Sorry, that topic is not present in the provided document.' "
    "When answering, cite chunk indices like [chunk {chunk_index}] for each claim."
)


def build_prompt(query: str, hits: List[Tuple[str, dict]]) -> str:
    # hits: list of (doc_text, metadata dict)
    context_blocks = []
    for doc, meta in hits:
        # Remove [chunk X] markers from the context
        context_blocks.append(doc)
    context = "\n\n".join(context_blocks)
    return (
        "You are a helpful assistant that must answer ONLY using the provided context. "
        "If the answer is not completely supported by the context, reply exactly with: "
        "'Sorry, that topic is not present in the provided document.'\n\n"
        f"# Context:\n{context}\n\n# User question:\n{query}\n\n# Answer:"
    )


def answer_with_context(query: str, hits: List[Tuple[str, dict]]) -> str:
    prompt = build_prompt(query, hits)
    resp = client.models.generate_content(
        model=settings.GEN_MODEL,
        contents=prompt,
        config={"temperature": 0.2}
    )
    # Remove any [chunk X] markers from the output, just in case
    import re
    answer = (resp.text or "").strip()
    answer = re.sub(r"\[chunk\s*\d+\]", "", answer)
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer
