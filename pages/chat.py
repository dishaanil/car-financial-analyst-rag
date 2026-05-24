"""
Chat page - RAG question answering interface.
"""

import os
import re
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "chroma_db"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SESSION_ID = "streamlit"

# ── Style: make expander label grey and small ─────────────────────────────────

st.markdown("""
<style>
details summary p {
    color: #888888 !important;
    font-size: 0.82em !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load chain (cached so it runs once per server process) ────────────────────

@st.cache_resource(show_spinner="Loading vector store and RAG chain...")
def load_chain():
    from src.vector_store import SmartFinancialRetriever, build_vectorstore
    from src.rag_chain import build_rag_chain

    vectorstore = build_vectorstore(CHROMA_DB_PATH)
    retriever = SmartFinancialRetriever(vectorstore, k=6).as_langchain_retriever()
    chain, clear_fn = build_rag_chain(retriever, model=OPENAI_MODEL)
    return chain, clear_fn

chain, clear_history = load_chain()

# ── Helpers ───────────────────────────────────────────────────────────────────

def split_sources(answer: str) -> tuple[str, str | None]:
    """Split LLM response into (main answer, sources block)."""
    match = re.split(r'\n+\*{0,2}Sources:\*{0,2}\n+', answer, maxsplit=1, flags=re.IGNORECASE)
    if len(match) == 2:
        return match[0].strip(), match[1].strip()
    return answer.strip(), None


def render_answer(answer: str):
    """Render answer with sources in a collapsed expander."""
    main, sources = split_sources(answer)
    st.markdown(main)
    if sources:
        with st.expander("Sources"):
            st.markdown(sources)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("**Coverage**")
    st.markdown(
        "- **BMW** — 2021, 2022, 2023\n"
        "- **Ford** — 2021, 2022, 2023\n"
        "- **Tesla** — 2022, 2023"
    )
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        clear_history(SESSION_ID)
        st.session_state.messages = []
        st.rerun()

# ── Chat UI ───────────────────────────────────────────────────────────────────

st.header("Financial Report Q&A", divider="gray")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_answer(msg["content"])
        else:
            st.markdown(msg["content"])

# New input
if prompt := st.chat_input("Ask about BMW, Ford, or Tesla financials..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chain.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": SESSION_ID}},
            )
            answer = result.get("answer", "No answer returned.")
        render_answer(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
