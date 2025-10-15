from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.vectors import index
from typing import List

router = APIRouter(prefix="/collections", tags=["collections"])


class CollectionCreate(BaseModel):
    name: str


class CollectionListOut(BaseModel):
    collections: List[dict]


class CollectionDeleteOut(BaseModel):
    ok: bool


def get_namespaces():
    stats = index.describe_index_stats()
    return stats.get("namespaces", {})


@router.get("/", response_model=CollectionListOut)
def list_collections():
    namespaces = get_namespaces()
    collections = [
        {"name": name, "vector_count": ns.get("vector_count", 0)}
        for name, ns in namespaces.items()
    ]
    return {"collections": collections}


@router.post("/", response_model=CollectionCreate)
def create_collection(body: CollectionCreate):
    # Pinecone namespaces are logical; nothing to pre-create.
    namespaces = get_namespaces()
    if body.name in namespaces:
        print(f"[collections] Collection '{body.name}' already exists.")
        return body
    # No-op: just return as if created
    print(f"[collections] Created collection '{body.name}'.")
    return body


@router.delete("/{name}", response_model=CollectionDeleteOut)
def delete_collection(name: str):
    namespaces = get_namespaces()
    if name not in namespaces:
        raise HTTPException(status_code=404, detail="Collection not found")
    index.delete(namespace=name, delete_all=True)
    print(f"[collections] Deleted collection '{name}'.")
    return {"ok": True}


@router.get("/{name}/sources")
def collection_sources(name: str):
    namespaces = get_namespaces()
    if name not in namespaces:
        raise HTTPException(status_code=404, detail="Collection not found")
    # Per-source breakdown is not available without scanning all vectors.
    # For now, return not implemented.
    return {
        "detail": "Per-source breakdown not implemented. Only vector_count is available.",
        "vector_count": namespaces[name].get("vector_count", 0)
    }
