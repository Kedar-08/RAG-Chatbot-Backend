import os
from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name, default)
    return v.strip() if isinstance(v, str) else v


class Settings:
    # Google / Gemini
    GOOGLE_API_KEY = _get("GOOGLE_API_KEY")
    EMBED_MODEL = _get("EMBED_MODEL", "text-embedding-004")
    GEN_MODEL = _get("GEN_MODEL", "gemini-2.5-flash") 

    # RAG knobs
    TOP_K = int(_get("TOP_K", "8"))
    SIM_THRESHOLD = float(_get("SIM_THRESHOLD", "0.5"))
    CHUNK_SIZE = int(_get("CHUNK_SIZE", "1200"))
    CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "200"))

    # Pinecone
    PINECONE_API_KEY = _get("PINECONE_API_KEY")
    PINECONE_INDEX = _get("PINECONE_INDEX", "rag-gemini-demo")
    PINECONE_CLOUD = _get("PINECONE_CLOUD", "aws")
    PINECONE_REGION = _get("PINECONE_REGION", "us-east-1")


settings = Settings()
