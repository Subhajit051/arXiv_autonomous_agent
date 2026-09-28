from src.pdf_node import chunk_text

def test_chunking():
    chunks = chunk_text("a" * 8000, chunk_size=1000, overlap=100)
    assert len(chunks) > 1
    assert all(chunks)
