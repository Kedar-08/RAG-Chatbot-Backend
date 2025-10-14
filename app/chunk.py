import re
from typing import List

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
    # join hyphenated line breaks: cam-\n el → camel
    t = re.sub(r"-\s*\n\s*", "", t)
    # collapse newlines to spaces
    t = re.sub(r"\s*\n\s*", " ", t)
    # collapse spaces
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def chunk_text(t: str, chunk_size=1200, overlap=200) -> List[str]:
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
                chunks.append(block)
            tail = " ".join(cur)[-overlap:]
            cur = tail.split(" ") if overlap > 0 else []
            cur_len = sum(len(x)+1 for x in cur)
    if cur:
        chunks.append(" ".join(cur).strip())
    return [c for c in chunks if c]


def chunk_text(t, chunk_size=1200, overlap=200, page=None) -> List[dict]:
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
