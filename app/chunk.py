import re
from typing import List, Dict, Optional

LIGATURES = {
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    "’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"
}


def normalize_unicode(t: str) -> str:
    for k, v in LIGATURES.items():
        t = t.replace(k, v)
    return t


def clean_text(t: str) -> str:
    t = normalize_unicode(t)
    t = re.sub(r"-\s*\n\s*", "", t)
    t = re.sub(r"\s*\n\s*", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def chunk_text(
    t: str, chunk_size: int = 1200, overlap: int = 200, page: Optional[int] = None
) -> List[Dict[str, Optional[str]]]:
    """
    Splits text into chunks, preserving page number metadata if provided.
    Returns a list of dicts: [{"text": ..., "page": ...}, ...]
    """
    t = clean_text(t)
    words = t.split(" ")
    chunks, cur = [], []
    cur_len = 0
    for w in words:
        cur.append(w)
        cur_len += len(w) + 1
        if cur_len >= chunk_size:
            block = " ".join(cur).strip()
            if block:
                chunks.append({"text": block, "page": page})
            tail = " ".join(cur)[-overlap:]
            cur = tail.split(" ") if overlap > 0 else []
            cur_len = sum(len(x)+1 for x in cur)
    if cur:
        chunks.append({"text": " ".join(cur).strip(), "page": page})
    return [c for c in chunks if c["text"]]
