import chromadb
from .config import CHROMA_DIR
from .llm import embed_documents, embed_query
from .pdf_node import chunk_text


def _collection_name(paper_id: str) -> str:
    safe_id = paper_id.replace("/", "_").replace("-", "_").replace(".", "_")
    return "paper_" + safe_id


def build_vector_store(state: dict) -> dict:
    chunks = chunk_text(state["parsed_text"])
    paper_id = state["selected_paper"]["id"].replace("/", "_")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    name = _collection_name(paper_id)
    # Rebuild the collection for the selected paper so repeated runs are deterministic.
    # Chroma does not accept an empty where={} filter, so delete the whole
    # collection and recreate it when it already exists.
    try:
        client.delete_collection(name=name)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=name,
        configuration={"hnsw": {"space": "cosine"}},
    )

    embeddings = embed_documents(chunks)
    ids = [f"{paper_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {"paper_id": paper_id, "chunk_index": i}
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return {"chunks": chunks, "vector_store_name": name}


def retrieve_chunks(state: dict, question: str, k: int = 5) -> list[str]:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=state["vector_store_name"])
    count = collection.count()
    if count == 0:
        return []

    result = collection.query(
        query_embeddings=[embed_query(question)],
        n_results=min(k, count),
        include=["documents", "distances", "metadatas"],
    )

    docs = result["documents"][0]
    distances = result["distances"][0]
    metadatas = result["metadatas"][0]

    return [
        f"[Retrieved chunk {m['chunk_index']} | distance={distance:.4f}]\n{doc}"
        for doc, distance, m in zip(docs, distances, metadatas)
    ]
