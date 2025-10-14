from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi import status
from pydantic import BaseModel
from pathlib import Path
from typing import List
from app.cors import setup_cors
from app import pdf, chunk
from app.vectors import upsert_chunks, search
from app.config import settings
from app.llm import answer_with_context
import json
import asyncio
from app.collections import router as collections_router

app = FastAPI(title="RAG x Gemini (Pinecone)", version="0.1.0")

setup_cors(app)

app.include_router(collections_router)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class QueryIn(BaseModel):
    collection: str
    question: str
    top_k: int | None = None
    sim_threshold: float | None = None
    suggest_search: bool = True
    filter: dict | None = None  # Pinecone metadata filter (see Pinecone docs)

    class Config:
        schema_extra = {
            "description": "The `filter` field follows Pinecone metadata filter syntax. See https://docs.pinecone.io/docs/metadata-filtering for details."
        }


class IngestTextIn(BaseModel):
    collection: str
    text: str
    source: str | None = "inline"


@app.post(
    "/ingest",
    summary="Ingest one or more PDF files into a collection",
    description="Upload one or more PDF files. Each file is chunked and indexed. Returns per-file status and total chunks.",
)
async def ingest(
    collection: str = Form(..., description="Collection (namespace) name"),
    files: List[UploadFile] = File(...,
                                   description="One or more PDF files (array of binary)"),
):
    MAX_FILES = 5
    MAX_SIZE = 50 * 1024 * 1024  # 50MB per file

    if not files:
        raise HTTPException(400, "No files uploaded.")

    if len(files) > MAX_FILES:
        raise HTTPException(
            400, f"Too many files: max {MAX_FILES} per request.")

    results = []
    total_chunks = 0

    for file in files:
        file_result = {"name": file.filename, "chunks": 0, "error": None}
        try:
            if not file.filename.lower().endswith(".pdf"):
                file_result["error"] = "Only PDF files are supported."
                results.append(file_result)
                continue

            content = await file.read()
            if not content or len(content) < 10:
                file_result["error"] = "Uploaded file appears empty."
                results.append(file_result)
                continue

            if len(content) > MAX_SIZE:
                file_result["error"] = f"File too large (max {MAX_SIZE // (1024*1024)}MB)."
                results.append(file_result)
                continue

            dest = UPLOAD_DIR / file.filename
            dest.write_bytes(content)

            # Try to extract text (string OR per-page list)
            try:
                raw = pdf.extract_text_from_pdf(str(dest))
            except Exception as e:
                file_result["error"] = f"Failed to read PDF: {type(e).__name__}: {e}"
                results.append(file_result)
                continue

            # Normalize to a list of (page, text) pairs
            pages: list[tuple[int | None, str]] = []
            if isinstance(raw, str):
                pages = [(None, raw)]
            elif isinstance(raw, list):
                for i, item in enumerate(raw):
                    if isinstance(item, dict):
                        t = item.get("text", "") or ""
                        p = item.get("page", i + 1)
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        p, t = item[0], item[1]
                    else:
                        t = str(item) if item is not None else ""
                        p = i + 1
                    pages.append((p, t))
            else:
                file_result["error"] = "Unexpected PDF extractor output type."
                results.append(file_result)
                continue

            total_len = sum(len((t or "").strip()) for _, t in pages)
            if total_len < 50:
                file_result["error"] = "No extractable text found in the PDF (maybe it's scanned?)."
                results.append(file_result)
                continue

            # Chunk per page; preserve page in metadata for each chunk
            chunks_with_meta: list[dict] = []
            for p, t in pages:
                t = (t or "").strip()
                if not t:
                    continue
                chunk_list = chunk.chunk_text(
                    t, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP, page=p
                )
                chunks_with_meta.extend(chunk_list)

            if not chunks_with_meta:
                file_result["error"] = "Chunking produced no chunks (unexpected)."
                results.append(file_result)
                continue

            source_id = file.filename
            try:
                upsert_chunks(collection, chunks_with_meta, source_id)
            except Exception as e:
                file_result["error"] = f"Vector upsert failed: {type(e).__name__}: {e}"
                results.append(file_result)
                continue

            file_result["chunks"] = len(chunks_with_meta)
            total_chunks += len(chunks_with_meta)
            print(
                f"[ingest] {file.filename}: {len(chunks_with_meta)} chunks ingested.")
        except Exception as e:
            file_result["error"] = f"Unhandled error: {type(e).__name__}: {e}"
        results.append(file_result)

    ok = any(r.get("chunks", 0) > 0 and not r.get("error") for r in results)
    resp = {
        "ok": ok,
        "collection": collection,
        "files": results,
        "total_chunks": total_chunks
    }
    if not ok:
        raise HTTPException(400, resp)
    return resp


def retrieve_context(q: QueryIn):
    """Shared retrieval logic for /query and /query_stream."""
    _top_k = q.top_k or settings.TOP_K
    _thr = q.sim_threshold or settings.SIM_THRESHOLD
    # --- input validation for filter ---
    if q.filter is not None and not isinstance(q.filter, dict):
        raise HTTPException(
            status_code=400,
            detail="`filter` must be a plain dict or null, following Pinecone metadata filter syntax."
        )
    res = search(q.collection, q.question, _top_k, filter=q.filter)
    docs = res["documents"][0] if res and res.get("documents") else []
    metas = res["metadatas"][0] if res and res.get("metadatas") else []
    dists = res.get("distances", [[]])[0] if "distances" in res else None
    pairs = []
    for i, doc in enumerate(docs):
        sim = 1.0 - dists[i] if dists and i < len(dists) else 1.0
        if sim >= _thr:
            pairs.append((doc, metas[i], sim))
    return pairs


def sse_event(event: str, data: dict):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/query")
async def query(q: QueryIn):
    pairs = retrieve_context(q)
    if not pairs:
        payload = {
            "answer": "Sorry, that topic is not present in the provided document.",
            "matches": [],
        }
        if q.suggest_search:
            import urllib.parse as u
            payload["suggested_search"] = f"https://www.google.com/search?q={u.quote(q.question)}"
        return JSONResponse(payload)

    # keep top 5
    top_hits = sorted(pairs, key=lambda x: x[2], reverse=True)[:5]

    # --- step 3: ask LLM (Gemini) ---
    try:
        answer = answer_with_context(
            q.question, [(d, m) for d, m, _ in top_hits])
    except Exception as e:
        # Helpful debug return: show what we would have sent to the model.
        preview = [
            {
                "chunk_index": m.get("chunk_index"),
                "source": m.get("source"),
                "text_preview": (d[:300] + "…") if len(d) > 300 else d
            }
            for d, m, _ in top_hits
        ]
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={
                "error": f"LLM call failed: {type(e).__name__}: {e}",
                "message": "Retrieval succeeded. The language model step failed.",
                "question": q.question,
                "top_hits_preview": preview
            }
        )

    # Collect, deduplicate, and sort citations (include page if present)
    citations = [
        {
            "chunk_index": m.get("chunk_index"),
            "source": m.get("source"),
            "page": m.get("page"),  # new
        }
        for _, m, _ in top_hits
    ]

    # Deduplicate and sort by (chunk_index, source); keep page if available
    unique = {}
    for c in citations:
        key = (c.get("chunk_index"), c.get("source"))
        # prefer keeping a version that has page info
        if key not in unique or (unique[key].get("page") is None and c.get("page") is not None):
            unique[key] = c

    sorted_citations = [unique[k] for k in sorted(unique)]

    return {"answer": answer, "citations": sorted_citations}


@app.post("/peek")
async def peek(q: QueryIn):
    pairs = retrieve_context(q)
    previews = []
    for d, m, sim in pairs:
        text = d.get("text", "") if isinstance(d, dict) else d
        previews.append({
            "similarity": round(sim, 3),
            "chunk_index": m.get("chunk_index"),
            "source": m.get("source"),
            "page": m.get("page"),
            "text_preview": text[:200] + ("..." if len(text) > 200 else "")
        })
    return {"top_hits": previews}


@app.post("/query_stream")
async def query_stream(q: QueryIn, request: Request):
    async def event_generator():
        try:
            pairs = retrieve_context(q)
            if not pairs:
                payload = {
                    "answer": "Sorry, that topic is not present in the provided document."
                }
                if q.suggest_search:
                    import urllib.parse as u
                    payload[
                        "suggested_search"] = f"https://www.google.com/search?q={u.quote(q.question)}"
                yield sse_event("not_found", payload)
                return

            # Prepare citations
            top_hits = sorted(pairs, key=lambda x: x[2], reverse=True)[:5]
            citations = [
                {
                    "chunk_index": m.get("chunk_index"),
                    "source": m.get("source"),
                    "page": m.get("page"),
                }
                for _, m, _ in top_hits
            ]
            unique = {}
            for c in citations:
                key = (c.get("chunk_index"), c.get("source"))
                if key not in unique or (unique[key].get("page") is None and c.get("page") is not None):
                    unique[key] = c
            sorted_citations = [unique[k] for k in sorted(unique)]
            yield sse_event("citations", {"citations": sorted_citations})

            # Stream model output
            try:
                # Try Gemini streaming if available
                # If not, fallback to chunked simulation
                model_streaming_supported = False
                if hasattr(answer_with_context, "stream"):
                    # If your answer_with_context supports streaming, use it
                    async for chunk in answer_with_context.stream(
                        q.question, [(d, m) for d, m, _ in top_hits]
                    ):
                        yield sse_event("token", {"text": chunk})
                        model_streaming_supported = True
                if not model_streaming_supported:
                    # Fallback: simulate streaming by chunking the answer
                    answer = answer_with_context(
                        q.question, [(d, m) for d, m, _ in top_hits]
                    )
                    chunk_size = 60
                    for i in range(0, len(answer), chunk_size):
                        await asyncio.sleep(0.01)
                        yield sse_event("token", {"text": answer[i:i+chunk_size]})
            except Exception as e:
                yield sse_event("error", {"message": f"LLM error: {type(e).__name__}: {e}"})
                return

            yield sse_event("done", {})
        except Exception as e:
            yield sse_event("error", {"message": f"Internal error: {type(e).__name__}: {e}"})
            return

    headers = {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    return StreamingResponse(event_generator(), headers=headers)
