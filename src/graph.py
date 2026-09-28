from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from .arxiv_node import retrieve_papers, select_paper
from .pdf_node import fetch_and_parse
from .vector_node import build_vector_store, retrieve_chunks
from .llm import generate


class AgentState(TypedDict, total=False):
    query: str
    query_type: str
    candidate_papers: list[dict]
    selected_paper: dict
    paper_metadata: dict
    pdf_path: str
    parsed_text: str
    chunks: list[str]
    vector_store_name: str
    briefing: str
    question: str
    retrieved_chunks: list[str]
    answer: str
    chat_history: list[dict]


def understand_query(state: AgentState) -> dict:
    import re
    is_id = bool(
        re.search(
            r"(?:arxiv\.org/(?:abs|pdf)/)?[0-9]{4}\.[0-9]{4,5}",
            state["query"],
            re.IGNORECASE,
        )
    )
    return {"query_type": "paper_id_or_url" if is_id else "topic"}


def create_briefing(state: AgentState) -> dict:
    p = state["selected_paper"]
    prompt = f"""You are a research-paper briefing assistant.
Use ONLY the supplied paper text and metadata. Do not use outside knowledge.
Do not invent results, limitations, authors, dates, or claims.

Create an executive briefing with exactly these headings:
# Title
# Authors
# arXiv ID
# Publish Date
# Link
# Why This Paper Matters
# Problem Statement
# Method / Approach
# Key Results / Claims
# Limitations
# Suggested Follow-up Questions

Paper metadata:
{p}

Extracted paper text:
{state['parsed_text'][:60000]}

If a result or limitation is not clearly supported by the extracted text, write:
Not clearly stated in the extracted text.
"""
    return {"briefing": generate(prompt), "paper_metadata": p}


def answer_question(state: AgentState) -> dict:
    chunks = retrieve_chunks(state, state["question"])
    if not chunks:
        answer = "I couldn't find enough information about that in the paper's retrieved sections."
    else:
        prompt = f"""Answer the user's question using ONLY the retrieved excerpts from the paper.

USER QUESTION:
{state['question']}

RETRIEVED EXCERPTS:
{'\n\n'.join(chunks)}

Rules:
- Do not use outside knowledge.
- If the excerpts do not contain enough evidence, say exactly:
  I couldn't find enough information about that in the paper's retrieved sections.
- Keep the answer concise but explanatory.
- Mention the retrieved chunk number for factual claims when practical.
"""
        answer = generate(prompt)

    history = state.get("chat_history", []) + [
        {"question": state["question"], "answer": answer}
    ]
    return {
        "retrieved_chunks": chunks,
        "answer": answer,
        "chat_history": history,
    }


def build_graph():
    workflow = StateGraph(AgentState)
    nodes = [
        ("understand_query", understand_query),
        ("retrieve_papers", retrieve_papers),
        ("select_paper", select_paper),
        ("fetch_and_parse", fetch_and_parse),
        ("build_vector_store", build_vector_store),
        ("create_briefing", create_briefing),
    ]
    for name, fn in nodes:
        workflow.add_node(name, fn)

    workflow.add_edge(START, "understand_query")
    workflow.add_edge("understand_query", "retrieve_papers")
    workflow.add_edge("retrieve_papers", "select_paper")
    workflow.add_edge("select_paper", "fetch_and_parse")
    workflow.add_edge("fetch_and_parse", "build_vector_store")
    workflow.add_edge("build_vector_store", "create_briefing")
    workflow.add_edge("create_briefing", END)
    return workflow.compile()


def build_qa_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("answer_question", answer_question)
    workflow.add_edge(START, "answer_question")
    workflow.add_edge("answer_question", END)
    return workflow.compile()
