import urllib.request
import pymupdf
from .config import PAPERS_DIR


def fetch_and_parse(state: dict) -> dict:
    paper = state["selected_paper"]
    paper_id = paper["id"].replace("/", "_")
    pdf_path = PAPERS_DIR / f"{paper_id}.pdf"

    if not pdf_path.exists():
        request = urllib.request.Request(
            paper["pdf_url"],
            headers={"User-Agent": "arxiv-digest-agent/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response, open(pdf_path, "wb") as output:
                output.write(response.read())
        except Exception as exc:
            raise RuntimeError(f"Could not download the paper PDF: {exc}") from exc

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Could not open the downloaded PDF: {exc}") from exc

    pages = []
    try:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(f"[Page {page_number}]\n{text}")
    finally:
        doc.close()

    parsed_text = "\n\n".join(pages)
    if len(parsed_text) < 1000:
        raise ValueError(
            "PDF produced too little text. It may be scanned, malformed, "
            "or inaccessible to text extraction."
        )

    return {
        "pdf_path": str(pdf_path),
        "parsed_text": parsed_text,
        "paper_metadata": paper,
    }


def chunk_text(text: str, chunk_size: int = 3500, overlap: int = 500) -> list[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be > 0 and overlap must be between 0 and chunk_size-1")

    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks
