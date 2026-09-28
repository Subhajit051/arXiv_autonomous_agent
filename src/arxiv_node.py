import re
import arxiv


def extract_arxiv_id(value: str) -> str | None:
    match = re.search(
        r"(?:arxiv\.org/(?:abs|pdf)/)?([0-9]{4}\.[0-9]{4,5})(?:v\d+)?",
        value.strip(),
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def _paper_to_dict(result) -> dict:
    paper_id = result.get_short_id()
    return {
        "id": paper_id,
        "title": result.title.strip(),
        "authors": [a.name for a in result.authors],
        "published": result.published.strftime("%Y-%m-%d"),
        "summary": result.summary.replace("\n", " ").strip(),
        "pdf_url": result.pdf_url,
        "abs_url": f"https://arxiv.org/abs/{paper_id}",
    }


def retrieve_papers(state: dict) -> dict:
    query = state["query"].strip()
    paper_id = extract_arxiv_id(query)
    client = arxiv.Client(page_size=5, delay_seconds=3, num_retries=3)

    if paper_id:
        search = arxiv.Search(id_list=[paper_id])
    else:
        # all: gives a simple, transparent baseline for topic search.
        search = arxiv.Search(
            query=f'all:"{query}"',
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance,
        )

    results = list(client.results(search))

    if not results and not paper_id:
        # Fallback for broad natural-language topics.
        search = arxiv.Search(
            query=query,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = list(client.results(search))

    if not results:
        raise ValueError(
            f"No arXiv papers found for '{query}'. Try a broader research topic "
            "or a valid arXiv ID such as 1706.03762."
        )

    return {"candidate_papers": [_paper_to_dict(r) for r in results]}


def select_paper(state: dict) -> dict:
    candidates = state["candidate_papers"]
    return {"selected_paper": candidates[0]}
