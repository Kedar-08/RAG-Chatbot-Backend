from typing import List
from app.config import settings
from google import genai
from pinecone import Pinecone, ServerlessSpec

# ---- Embeddings via Gemini ----


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str, output_dim: int = 768):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.output_dim = output_dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        # We explicitly request 768-dimensional embeddings.
        res = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config={"output_dimensionality": self.output_dim}
        )
        if hasattr(res, "embeddings"):
            return [e.values for e in res.embeddings]
        return [res.embedding.values]


EMBED_DIM = 768  # keep Pinecone index + Gemini output in sync
embedder = GeminiEmbedder(settings.GOOGLE_API_KEY,
                          settings.EMBED_MODEL, EMBED_DIM)

# ---- Pinecone setup (serverless) ----
PC_API_KEY = settings.PINECONE_API_KEY
PC_INDEX = settings.PINECONE_INDEX
PC_CLOUD = settings.PINECONE_CLOUD
PC_REGION = settings.PINECONE_REGION

pc = Pinecone(api_key=PC_API_KEY)

# Create index if it doesn't exist
existing = [i.name for i in pc.list_indexes()]
if PC_INDEX not in existing:
    pc.create_index(
        name=PC_INDEX,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=PC_CLOUD, region=PC_REGION)
    )

index = pc.Index(PC_INDEX)

BATCH = 100  # tune 64–200 based on your keys/limits


def batched(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]


def upsert_chunks(collection_name: str, chunks: list, source_id: str):
    """
    Store vectors in Pinecone. Supports both:
      - list of strings: ["chunk1", "chunk2", ...]
      - list of dicts: [{"text": ..., "page": ...}, ...]
    Each chunk is embedded (text only), and metadata includes source, chunk_index, and page (if present).
    """
    namespace = collection_name
    idx_offset = 0
    for chunk_batch in batched(chunks, BATCH):
        norm_batch = []
        for c in chunk_batch:
            if isinstance(c, dict):
                text = str(c.get("text", "")).strip()
                page = c.get("page")
            else:
                text = str(c).strip()
                page = None
            if text:
                norm_batch.append({"text": text, "page": page})
        if not norm_batch:
            continue
        texts = [c["text"] for c in norm_batch]
        vectors = embedder.embed(texts)
        items = []
        for i, vec in enumerate(vectors):
            text_val = norm_batch[i]["text"]
            if isinstance(text_val, dict):
                text_val = text_val.get("text", "")
            elif not isinstance(text_val, str):
                text_val = str(text_val)
            meta = {
                "text": text_val,
                "source": source_id,
                "chunk_index": idx_offset + i
            }
            if norm_batch[i]["page"] is not None:
                meta["page"] = norm_batch[i]["page"]
            items.append({
                "id": f"{source_id}_{idx_offset + i}",
                "values": vec,
                "metadata": meta
            })
        index.upsert(vectors=items, namespace=namespace)
        idx_offset += len(norm_batch)


def search(collection_name: str, query: str, top_k: int, filter: dict = None):
    """
    Vector search: embed the query, retrieve top_k nearest chunks.
    Optionally filter results using Pinecone's filter dict.
    We return a structure compatible with the rest of the app.
    """
    namespace = collection_name
    qv = embedder.embed([query])[0]
    query_args = {
        "vector": qv,
        "top_k": top_k,
        "include_metadata": True,
        "namespace": namespace
    }
    if filter is not None:
        query_args["filter"] = filter

    res = index.query(**query_args)

    docs, metas, dists = [], [], []
    for match in res.matches:
        md = match.metadata or {}
        docs.append(md.get("text", ""))
        meta = {
            "source": md.get("source", ""),
            "chunk_index": int(md.get("chunk_index", -1))
        }
        if "page" in md:
            meta["page"] = md["page"]
        metas.append(meta)
        dists.append(1.0 - float(match.score)
                     if match.score is not None else 1.0)

    return {
        "documents": [docs],
        "metadatas": [metas],
        "distances": [dists]
    }
