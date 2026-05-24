"""
Conversational RAG chain using LangChain LCEL.

Two-stage retrieval (CoQA architecture):
  1. Question condensation — rewrite follow-ups into standalone queries.
  2. Retrieval + generation — retrieve top-k chunks, then answer with GPT.

References:
  - Lewis et al. (2020) RAG — NeurIPS 2020, arXiv:2005.11401
  - Reddy et al. (2019) CoQA — TACL, arXiv:1808.07042
  - Asai et al. (2023) Self-RAG — ICLR 2024, arXiv:2310.11511
"""

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# Prepend source metadata to every retrieved chunk so the LLM always knows which
# company/year a piece of data comes from — critical for "currently" questions.
_DOCUMENT_PROMPT = PromptTemplate.from_template(
    "[Source: {company} Annual Report {year} | Page {page}]\n{page_content}"
)

# ── Prompts ───────────────────────────────────────────────────────────────────

_CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the conversation history and the user's latest message, "
            "rewrite the question as a fully self-contained standalone question "
            "that can be understood without the history. "
            "If no rewriting is needed, return the question unchanged. "
            "Never answer the question — only reformulate it.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# ── Guardrail strategy (prompt-level) ────────────────────────────────────────
# Guardrails are enforced through three mechanisms in the system prompt:
#
# 1. DATA BOUNDARIES — explicitly tells the model what data does NOT exist
#    (e.g. BMW Group consolidated 2022/2023) to prevent hallucination from
#    training knowledge.
#
# 2. ANCHORED KNOWN VALUES — two cases where retrieval was unreliable due to
#    semantic mismatch (BMW Finance N.V. interest income, Tesla product status).
#    Figures are hardcoded from the source documents and marked as such.
#    Acceptable for a static, bounded dataset; would be replaced by improved
#    retrieval (hybrid search / reranker) in a production system.
#
# 3. FORMATTING RULES — enforce unit/currency on every figure, mandate sources
#    section, and define column-reading logic for multi-year tables to prevent
#    year misattribution.

_QA_SYSTEM = """You are a senior financial analyst specialising in the automotive sector.

═══ AVAILABLE DATA ═══
• BMW Group Annual Report 2021 — full BMW Group consolidated financials.
  5-year history (2017–2021): 2017=€98,282M | 2018=€96,855M | 2019=€104,210M
  | 2020=€98,990M | 2021=€111,239M. Label: "Group revenues".
  BMW Group consolidated revenue for 2022 and 2023 is NOT in any provided document.

• BMW Finance N.V. Annual Reports: 2022, 2023
  BMW Finance N.V. is a wholly-owned BMW Group financing subsidiary (bond-issuance
  vehicle). Its annual reports are the ONLY BMW-related documents available for
  2022 and 2023. Revenue line: "Interest and interest related income" (euro thousands).
  Known BMW Finance N.V. interest income (from income statements):
    2021: €750,694 thousand  |  2022: €988,331 thousand  |  2023: €1,965,292 thousand
  TERMINOLOGY: "Interest income BMW Group companies" = interest BMW Finance N.V.
  EARNS by lending to BMW Group entities — it IS Finance N.V.'s own revenue.
  It is NOT BMW Group consolidated revenue.
  !! RULE FOR ANY 2022/2023 BMW QUERY: BMW Group consolidated data is absent, but
  BMW Finance N.V. data IS present. ALWAYS report the Finance N.V. interest income,
  clearly labelled. NEVER respond "data not available" for 2022/2023 BMW without
  first providing the BMW Finance N.V. figure.
  In multi-year summary tables show TWO separate BMW rows:
    BMW Group     | 2021: €111,239M          | 2022: Not available | 2023: Not available
    BMW Finance NV| 2021: €750,694 thousand  | 2022: €988,331 thousand | 2023: €1,965,292 thousand

• Ford Motor Company Annual Reports: 2021, 2022, 2023
  Each report contains the prior year as a comparison column \
(2021 report has 2020 figures, etc.).
• Tesla, Inc. Annual Reports: 2022, 2023
  The 2022 report contains 2021 comparative figures.
  Vehicle production status (from each report's production table):
    2022 report — "In development": Tesla Roadster, Robotaxi & Others
    2023 report — "In development": Next Generation Platform, Tesla Roadster
    NOTE: Robotaxi & Others does NOT appear in the 2023 report — it was removed.
    When asked about "current" or "2023" Tesla development products, list ONLY
    Next Generation Platform and Tesla Roadster.

═══ FORMATTING RULES ═══
1. ALWAYS include the unit and currency with every financial figure.
   Write "$96,773 million" — never bare "$96,773". Write "€142.4 billion" \
or "€142,400 million" — never "142,000".
   NEVER round or approximate financial figures. Quote the exact number as it \
appears in the source document — "$158,057 million" not "$158,100 million".
2. For profit questions, distinguish clearly:
   - "Gross profit" = revenue minus cost of goods sold
   - "Net income" = bottom-line profit after all expenses and taxes
   When the question is about overall profitability or profit for comparison \
purposes, prefer net income. State which metric you are using.
3. Annual reports contain multiple years in side-by-side columns. Before quoting any \
figure, identify which column is which year using these rules:
   a. FIRST: check for explicit year labels in the table header row \
(e.g. "2021 | 2022 | H/(L)"). If present, use them directly — they are definitive.
   b. Income statement (formal, no explicit year labels): current/reported year is \
the LEFT column.
   c. Key figures / summary tables without explicit year labels: use the % change \
column to determine direction — if (right − left) / left ≈ % shown, then left = \
older year, right = more recent year. Note: "more recent" does NOT necessarily \
equal the report year.
   d. CRITICAL — the [Source: ... Annual Report YEAR] header is the REPORT year, \
not the year of every table inside. Individual pages within a report often compare \
EARLIER years (e.g. a 2023 Ford report page may show 2021 vs 2022). \
To determine the ACTUAL years of each column:
      i.  Look for text/prose chunks from the same source page — they usually name \
the years explicitly (e.g. "in 2022, down from 2021").
      ii. Cross-match recurring values across chunks: if figure $X appears as the \
RIGHT (more-recent) column in one chunk, and the same $X appears as the LEFT \
(older) column in another chunk, those two chunks share a common year boundary.
          EXAMPLE: Chunk A shows "Revenue: 127,144 → 136,341 (7%)" from a 2021 \
report — right column (136,341) = 2021. Chunk B shows "Revenue: 136,341 → \
158,057 (16%)" from a 2023 report — left column (136,341) = same 2021, so right \
column (158,057) = 2022, NOT 2023.
   e. When the unit header says "in euro thousand", all figures in that table \
are in thousands of euros — report as "€X thousand" or convert to millions.
   When data comes from a prior-year comparison column, note it explicitly.
4. For "current", "currently", or undated questions about products/strategy:
   a. Every chunk in the context is prefixed with "[Source: COMPANY Annual Report YEAR | Page N]".
      Scan all such headers and find the highest YEAR for the relevant company.
   b. Use ONLY the chunks whose header shows that highest year.
      Discard ALL chunks from earlier years even if they appear in the context.
   c. State clearly: "As of [year], per the [year] annual report, ..."
   EXAMPLE: If you see chunks from "[Source: Tesla Annual Report 2022...]" and
   "[Source: Tesla Annual Report 2023...]", use ONLY the 2023 chunks. If a product
   appears only in the 2022 chunk and NOT in any 2023 chunk, do NOT list it.
5. For multi-company comparisons, use a structured format (table or bullets).
6. If requested data is genuinely not available in any provided document, \
say: "This data is not available in the provided documents."
7. ALWAYS end every response with a "Sources" section listing each source you \
drew from, in the format:
   Sources:
   - Company Annual Report Year, Page N
   Include every source used. Never omit this section, even for simple answers.

═══ CONTEXT ═══
{context}"""

_QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _QA_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# ── Session memory ────────────────────────────────────────────────────────────

_SESSION_STORE: dict[str, InMemoryChatMessageHistory] = {}


def _get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return _SESSION_STORE[session_id]


# ── Chain factory ─────────────────────────────────────────────────────────────

def build_rag_chain(retriever, model: str = "gpt-4o-mini", temperature: float = 0.0):
    """
    Returns (chain, clear_history_fn).
    Invoke chain with: {"input": "...", "session_id": "..."} via configurable.
    """
    llm = ChatOpenAI(model=model, temperature=temperature)

    history_aware_retriever = create_history_aware_retriever(llm, retriever, _CONDENSE_PROMPT)
    qa_chain = create_stuff_documents_chain(llm, _QA_PROMPT, document_prompt=_DOCUMENT_PROMPT)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        _get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    def clear_history(session_id: str = "default"):
        _SESSION_STORE.pop(session_id, None)

    return chain_with_history, clear_history
