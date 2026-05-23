# Car Financial RAG

A Retrieval-Augmented Generation (RAG) chatbot for automotive sector annual reports (BMW, Ford, Tesla).

## Architecture

```
PDFs  ──►  pdfplumber       ──►  ChromaDB (text-embedding-3-small)
           (table-aware                │
            extraction)               │  metadata filter: company + year
                                      ▼
Chat history ──► Question condensation (gpt-4o-mini)
                      │
                      ▼
              Retrieval (top-6 chunks)
                      │
                      ▼
              Answer generation (gpt-4o-mini)
                      │
                      ▼
              Rich CLI output
```

### Key design decisions

| Challenge | Approach |
|---|---|
| PDF tables | `pdfplumber` converts tables to Markdown; tables are never split across chunks |
| Follow-up questions | Two-stage: condense question with history → retrieve → generate (CoQA architecture) |
| Financial precision | Metadata filtering by company + year reduces hallucination |
| Conversation state | `RunnableWithMessageHistory` (LangChain LCEL) with per-session `ChatMessageHistory` |

### Papers

- **RAG**: Lewis et al. (2020) — [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- **FinQA** (financial table QA): Chen et al. (2021) — [arXiv:2109.00122](https://arxiv.org/abs/2109.00122)
- **CoQA** (conversational QA): Reddy et al. (2019) — [arXiv:1808.07042](https://arxiv.org/abs/1808.07042)
- **HyDE** (better retrieval): Gao et al. (2022) — [arXiv:2212.10496](https://arxiv.org/abs/2212.10496)
- **Self-RAG**: Asai et al. (2023) — [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)

## Available data

| Company | Years |
|---|---|
| BMW | 2021, 2022, 2023 |
| Ford | 2021, 2022, 2023 |
| Tesla | 2022, 2023 |

> Questions about years outside this range (e.g. 2017, 2020) will be answered with a clear "data not available" response.

## Local setup

```bash
cd car-financial-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Create .env
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# First run: ingest PDFs (takes ~2-5 min for 8 reports)
python main.py

# Force re-ingestion if you add new PDFs
python main.py --ingest
```

## Docker setup

```bash
# Set your key in the environment
export OPENAI_API_KEY=sk-...

# Build and run interactively
docker compose run --rm rag

# Force re-ingestion
docker compose run --rm rag python main.py --ingest
```

## Example queries

```
What was BMW's total revenue in 2023?
How much revenue did Tesla generate in 2023?
What were BMW's profit figures for 2020 and 2023?
Between Tesla and Ford, which company achieved higher profits in 2022?
Provide a summary of revenue figures for Tesla, BMW, and Ford over the past three years.
What were the growth trends for BMW's financial performance from 2021 to 2023?
```

Follow-up questions work naturally:
```
You:  What was Tesla's revenue in 2023?
Bot:  Tesla's total revenue in 2023 was $96.8 billion...
You:  What about BMW?          ← follow-up, no need to repeat "revenue in 2023"
Bot:  BMW's total revenue in 2023 was €142.4 billion...
```

## Known limitations & production improvements

- **Table extraction**: `pdfplumber` struggles with merged cells and image-based tables.
  Production alternative: Azure Document Intelligence or GPT-4o multimodal (render page as image).
- **Hybrid search**: Adding BM25 alongside vector search (ensemble retriever) improves recall
  for exact number/term matches — important for financial data.
- **Numerical reasoning**: The system retrieves text and lets the LLM reason over numbers.
  For complex multi-step calculations, a tool-use / code-interpreter approach (FinQA style) would
  be more reliable.
- **Data coverage**: Only 2021–2023 (BMW/Ford) and 2022–2023 (Tesla) are included.
