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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
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
    parts = re.split(r'\n+\*{0,2}Sources:\*{0,2}\n+', answer, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return answer.strip(), None


def extract_number(text: str) -> float | None:
    """
    Extract a numeric value normalised to millions.
    '$136,341 million' → 136341.0
    '€1,965,292 thousand' → 1965.292  (÷1000 to convert to millions)
    '€142.4 billion' → 142400.0       (×1000 to convert to millions)
    """
    if re.search(r'not available|n/a', text, re.IGNORECASE):
        return None
    t = text.lower()
    if 'billion' in t:
        multiplier = 1000        # billions → millions
    elif 'thousand' in t:
        multiplier = 0.001       # thousands → millions
    else:
        multiplier = 1           # already in millions
    nums = re.findall(r'[\d]+(?:,[\d]{3})*(?:\.\d+)?', text)
    if nums:
        return float(nums[0].replace(',', '')) * multiplier
    return None


def detect_currency(text: str) -> str:
    """Detect currency symbol from a value string."""
    if '€' in text:
        return '€'
    if '$' in text:
        return '$'
    return ''


def parse_table_for_chart(answer: str) -> dict | None:
    """
    Parse the first markdown table in the answer and extract chart data.

    - Multiple numeric columns (e.g. 2021/2022/2023 revenue across companies)
      → grouped bar chart: first column = row labels, each numeric column = one series.
    - Single numeric column with year labels → line chart.
    - Single numeric column otherwise → bar chart.
    Returns None if no table found or fewer than 2 plottable data points.
    """
    table_re = re.compile(
        r'(?m)^\|(.+)\|\s*\n\|[-| :]+\|\s*\n((?:\|.+\|\s*\n?)+)'
    )
    match = table_re.search(answer)
    if not match:
        return None

    # Parse headers
    headers = [h.strip() for h in match.group(1).split('|') if h.strip()]

    # Parse rows
    rows = []
    for line in match.group(2).strip().split('\n'):
        line = line.strip()
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        if cells:
            rows.append(cells)

    if not rows or not headers:
        return None

    # Identify which columns (index >= 1) contain at least one numeric value
    numeric_cols = [
        col_idx for col_idx in range(1, len(headers))
        if any(extract_number(row[col_idx]) is not None
               for row in rows if col_idx < len(row))
    ]

    if not numeric_cols:
        return None

    # ── Multi-column → grouped bar ────────────────────────────────────────────
    if len(numeric_cols) > 1:
        row_labels = [row[0] for row in rows if row]
        series = []
        for col_idx in numeric_cols:
            col_name = headers[col_idx] if col_idx < len(headers) else str(col_idx)
            values = [
                extract_number(row[col_idx]) if col_idx < len(row) else None
                for row in rows
            ]
            # Only include the series if it has at least one real value
            if any(v is not None for v in values):
                series.append({'name': col_name, 'values': values})

        if len(series) < 2 or len(row_labels) < 1:
            return None

        return {'type': 'grouped_bar', 'labels': row_labels, 'series': series}

    # ── Single numeric column → bar or line ───────────────────────────────────
    numeric_col = numeric_cols[0]
    labels, values, currencies = [], [], []
    for row in rows:
        if numeric_col >= len(row):
            continue
        val = extract_number(row[numeric_col])
        if val is None:
            continue
        labels.append(row[0])
        values.append(val)
        currencies.append(detect_currency(row[numeric_col]))

    if len(labels) < 2:
        return None

    chart_type = 'line' if all(re.match(r'^\d{4}$', l.strip()) for l in labels) else 'bar'
    unit = headers[numeric_col] if numeric_col < len(headers) else ''
    dominant_currency = max(set(currencies), key=currencies.count) if currencies else ''
    if dominant_currency and dominant_currency not in unit:
        unit = f"{dominant_currency}{unit}"

    return {'type': chart_type, 'title': unit, 'labels': labels, 'values': values, 'unit': unit}


def render_chart(chart_data: dict):
    """Render a Plotly chart from parsed table data."""
    try:
        import plotly.graph_objects as go

        COLOURS = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#B279A2']

        if chart_data['type'] == 'grouped_bar':
            fig = go.Figure()
            for i, s in enumerate(chart_data['series']):
                fig.add_trace(go.Bar(
                    name=s['name'],
                    x=chart_data['labels'],
                    y=s['values'],
                    marker_color=COLOURS[i % len(COLOURS)],
                ))
            fig.update_layout(barmode='group')

        elif chart_data['type'] == 'line':
            fig = go.Figure(go.Scatter(
                x=chart_data['labels'], y=chart_data['values'],
                mode='lines+markers',
                line=dict(color=COLOURS[0], width=2),
                marker=dict(size=7),
            ))

        elif chart_data['type'] == 'pie':
            fig = go.Figure(go.Pie(
                labels=chart_data['labels'], values=chart_data['values'], hole=0.3
            ))

        else:  # single bar
            fig = go.Figure(go.Bar(
                x=chart_data['labels'], y=chart_data['values'],
                marker_color=COLOURS[0],
            ))

        fig.update_layout(
            title=chart_data.get('title', ''),
            yaxis_title=chart_data.get('unit', ''),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FAFAFA'),
            legend=dict(orientation='h', y=1.1),
            margin=dict(t=60, b=40, l=40, r=20),
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception:
        pass  # silently skip if anything goes wrong


def render_answer(answer: str):
    """Render answer text, optional chart from table, and collapsible sources."""
    main, sources = split_sources(answer)
    st.markdown(main)
    chart_data = parse_table_for_chart(main)
    if chart_data:
        render_chart(chart_data)
    if sources:
        with st.expander("Sources"):
            st.markdown(sources)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("**Coverage**")
    st.markdown(
        "- **BMW** - 2021, 2022, 2023\n"
        "- **Ford** - 2021, 2022, 2023\n"
        "- **Tesla** - 2022, 2023"
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
