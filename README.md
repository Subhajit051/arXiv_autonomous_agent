# Autonomous arXiv Paper Digest & QA Agent

A small, stateful research assistant that accepts either an **arXiv paper ID/URL** or a **natural-language research topic**, retrieves an arXiv paper through the official API, parses its PDF, chunks and embeds the text into a local ChromaDB collection, generates a structured executive briefing, and supports follow-up questions using retrieval-augmented generation (RAG).

## What this implements

- Explicit **LangGraph state graph**, rather than one monolithic prompt.
- Official **arXiv API** retrieval; no scraping.
- PDF extraction with **PyMuPDF**.
- Overlapping text chunking.
- **Gemini Embedding 001** for document/query embeddings.
- Local persistent **ChromaDB** vector store.
- Structured Gemini briefing with required sections and an explicit limitations section.
- QA loop that retrieves relevant paper chunks before generation.
- Grounding instruction: if retrieved excerpts do not contain enough evidence, the QA agent says so instead of using outside knowledge.
- Basic failure handling for empty input, no search results, PDF download/open failures, and PDFs that produce too little text.
- Limited retry plus model fallback for transient Gemini 5xx/429/timeout errors.

## Architecture

```text
                         +-------------------+
                         |      User CLI     |
                         +---------+---------+
                                   |
                                   v
                         +-------------------+
                         | Understand Query  |
                         +---------+---------+
                                   |
                                   v
                         +-------------------+
                         |   arXiv Retrieval |
                         +---------+---------+
                                   |
                                   v
                         +-------------------+
                         | Selection/Ranking |
                         +---------+---------+
                                   |
                                   v
                         +-------------------+
                         |  Fetch + Parse    |
                         |     PDF/PyMuPDF   |
                         +---------+---------+
                                   |
                                   v
                         +-------------------+
                         | Chunk + Embed     |
                         |     + ChromaDB    |
                         +---------+---------+
                                   |
                                   v
                         +-------------------+
                         | Executive Briefing|
                         +---------+---------+
                                   |
                                   v
                                  END

After briefing:

 Question
    |
    v
Query Embedding -> Chroma similarity search -> Retrieved chunks
                                                |
                                                v
                                        Gemini grounded answer
                                                |
                                                v
                                          chat_history
```

## Graph state

The shared LangGraph state can contain:

```text
query
query_type
candidate_papers
selected_paper
paper_metadata
pdf_path
parsed_text
chunks
vector_store_name
briefing
question
retrieved_chunks
answer
chat_history
```

This allows metadata, parsed content, the vector-store reference, and QA history to move between stages.

## Setup

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

If the Windows `pip` launcher is pointing to a different Python installation, use:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Configure Gemini

Copy `.env.example` to `.env` and add a Gemini API key from Google AI Studio.

Example:

```text
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
EMBEDDING_MODEL=gemini-embedding-001
```

**Never commit `.env` or an API key.** `.gitignore` already excludes `.env`.

The project is designed around free/open-source or free-tier components. Hosted Gemini availability and rate limits can vary; the application includes limited retry/fallback handling for transient generation failures.

### 4. Run

```bash
python main.py
```

Windows fallback if needed:

```powershell
.\.venv\Scripts\python.exe main.py
```

Then enter either:

```text
1706.03762
```

or a topic such as:

```text
KV-cache compression for LLMs
```

## Example run

A simple demonstration uses **Attention Is All You Need** (`1706.03762`).

### Input

```text
Enter an arXiv paper ID/URL or research topic: 1706.03762
```

### Briefing structure

The generated briefing contains:

```text
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
```

For this paper, the briefing discusses the Transformer architecture, its self-attention-based approach, the reported translation experiments, and limitations/future-work points supported by the extracted paper text.

### QA exchange 1 — grounded question

```text
Ask a question: What is the main idea of the Transformer architecture?
```

The QA stage first retrieves relevant chunks from ChromaDB and then asks Gemini to answer using those retrieved excerpts only.

### QA exchange 2 — grounded question

```text
Ask a question: What advantages does the Transformer have over recurrent neural networks?
```

Again, the answer is generated from retrieved paper chunks rather than from a general-knowledge prompt.

### QA exchange 3 — grounding/failure test

```text
Ask a question: What is the current price of Bitcoin?
```

This is intentionally outside the paper. The application is instructed to respond that it could not find enough information in the retrieved paper sections instead of answering from outside knowledge.

## Failure handling

- **Empty input:** rejected before the graph starts.
- **No arXiv results:** raises a clear error suggesting a broader topic or valid ID.
- **Topic search:** first uses a quoted `all:` query and then has a broader fallback query if no results are returned.
- **PDF download failure:** converted to a readable application error.
- **PDF parsing failure:** converted to a readable application error.
- **Very short extracted PDF text:** treated as likely scanned/malformed/inaccessible text.
- **Gemini transient failures:** limited exponential-backoff retry and fallback across configured stable text-generation models.
- **Repeated runs:** the paper's Chroma collection is rebuilt so stale chunks do not remain from a previous run.
- **Unsupported QA questions:** the generation prompt explicitly forbids outside knowledge and asks the model to state when retrieved evidence is insufficient.

## Design Decisions & Tradeoffs

### LangGraph
LangGraph was chosen because the assessment explicitly evaluates agent/state design. Separate nodes make the pipeline visible and make the shared state inspectable instead of hiding all behavior inside one prompt.

### arXiv API instead of scraping
The official arXiv API keeps retrieval within the assessment scope and provides structured metadata such as title, authors, dates, abstract, and PDF URL.

### PyMuPDF + simple overlapping chunks
PyMuPDF is lightweight for ordinary research PDFs. Fixed-size chunks with overlap keep the implementation understandable and provide enough context for basic semantic retrieval. A production system could use section-aware or token-aware chunking.

### ChromaDB locally
A persistent local vector database avoids external infrastructure and keeps the project reproducible. The tradeoff is that the store is local to one environment rather than shared between users.

### Gemini for generation and embeddings
Using one provider keeps the implementation small. The tradeoff is dependence on hosted-model availability and free-tier limits, so generation includes limited retry/fallback handling. Embeddings remain separate from text generation and use document/query retrieval task types.

### Transparent topic selection
For topic searches, arXiv relevance ordering is used as the baseline and the first returned candidate is selected. This is intentionally simple rather than introducing another ranking model that would add complexity without being required by the assessment.

### One paper per run
The implementation deliberately focuses on one selected paper at a time. This keeps the state and QA grounding clear while satisfying the required workflow. With more time, multi-paper comparison could be added.

## Testing

Run:

```bash
python -m pytest
```

The included unit test verifies that the chunking function creates multiple non-empty chunks.

For an end-to-end check, test:

1. A specific paper ID such as `1706.03762`.
2. A natural-language research topic.
3. A question whose answer is present in the paper.
4. An unrelated question to verify grounding.
5. An invalid/zero-result arXiv query to verify failure handling.

## Project structure

```text
arxiv_digest_agent/
├── main.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── llm.py
│   ├── arxiv_node.py
│   ├── pdf_node.py
│   ├── vector_node.py
│   └── graph.py
├── tests/
│   └── test_chunking.py
└── data/
    ├── papers/
    └── chroma/
```

## Scope

This is intentionally a focused assessment implementation. It does not include a production frontend, authentication, multi-user deployment, non-arXiv sources, or model fine-tuning. The CLI is sufficient for the assessment scope.
