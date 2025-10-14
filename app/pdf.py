import fitz  # PyMuPDF

def extract_text_from_pdf(path: str):
    """
    Extracts text from each page of a PDF, preserving page numbers.
    Returns a list of dicts: [{"text": ..., "page": ...}, ...]
    Page numbers start from 1.
    """
    pages = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            page_text = page.get_text("text")
            pages.append({
                "text": page_text,
                "page": i + 1  # 1-based page numbers
            })
    return pages