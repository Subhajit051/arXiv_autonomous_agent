from src.graph import build_graph, build_qa_graph


def main():
    print("=" * 70)
    print("Autonomous arXiv Paper Digest & QA Agent")
    print("=" * 70)
    query = input("\nEnter an arXiv paper ID/URL or research topic: ").strip()

    if not query:
        print("Please enter a paper ID, URL, or research topic.")
        return

    try:
        print("\n[1/6] Understanding query and finding the paper...")
        state = build_graph().invoke({"query": query})
    except Exception as exc:
        print(f"\nERROR: {exc}")
        print("\nTip: read the last line first; it usually tells you what needs fixing.")
        return

    paper = state["selected_paper"]
    print(f"Selected: {paper['title']} ({paper['id']})")
    print("Building the PDF text index and executive briefing...")

    print("\n" + "=" * 70)
    print("EXECUTIVE BRIEFING")
    print("=" * 70)
    print(state["briefing"])

    qa_graph = build_qa_graph()
    while True:
        question = input("\nAsk a question (or type 'exit'): ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        state = {
            **state,
            "question": question,
        }
        try:
            state = {**state, **qa_graph.invoke(state)}
            print("\nANSWER:\n" + state["answer"])
        except Exception as exc:
            print(f"\nQA ERROR: {exc}")


if __name__ == "__main__":
    main()
