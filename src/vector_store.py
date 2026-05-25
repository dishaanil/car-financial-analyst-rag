"""
ChromaDB vector store management with smart metadata filtering.

Key improvements:
- Prior-year comparative fallback: "Ford 2020" also searches Ford 2021,
  because annual reports always include the prior year as a comparison column.
- Per-company-per-year retrieval for multi-company/multi-year summaries,
  guaranteeing every combination gets representation in the context window.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

COMPANIES = ["BMW", "Ford", "Tesla"]
COLLECTION_NAME = "financial_reports"

# BMW 2021 = full BMW Group Annual Report (5-year history 2017-2021).
# BMW 2022 & 2023 = BMW Finance N.V. subsidiary reports.
AVAILABLE_YEARS: Dict[str, Set[str]] = {
    "BMW":   {"2021", "2022", "2023"},
    "Ford":  {"2021", "2022", "2023"},
    "Tesla": {"2022", "2023"},
}

_PROFIT_TERMS = {"profit", "loss", "income", "earnings", "ebitda", "margin"}
_REVENUE_TERMS = {"revenue", "sales", "turnover", "growth", "summary"}


def _is_profit_query(query: str) -> bool:
    return any(t in query.lower() for t in _PROFIT_TERMS)


def _is_revenue_query(query: str) -> bool:
    return any(t in query.lower() for t in _REVENUE_TERMS)

# Which report years are actually in our ChromaDB for each company.
# BMW reports are BMW Finance N.V., not BMW Group consolidated.
AVAILABLE_YEARS: Dict[str, Set[str]] = {
    "BMW":   {"2021", "2022", "2023"},
    "Ford":  {"2021", "2022", "2023"},
    "Tesla": {"2022", "2023"},
}


def _build_filter(companies: List[str], years: List[str]) -> Optional[dict]:
    company_filter = {"company": {"$in": companies}} if companies else None
    year_filter    = {"year":    {"$in": years}}    if years    else None
    if company_filter and year_filter:
        return {"$and": [company_filter, year_filter]}
    return company_filter or year_filter or None


_ALL_COMPANY_PHRASES = (
    "all three", "all companies", "all three companies",
    "each company", "each of the", "all of the companies",
    "three companies", "every company",
)


def extract_query_entities(query: str):
    """
    Simple NER: detect known company names and 4-digit years in the query.

    Special case: phrases like "all three companies" or "each company" are
    treated as naming all companies so the multi-company retrieval path fires.
    """
    q_lower = query.lower()
    companies = [c for c in COMPANIES if c.lower() in q_lower]
    # "all three" / "all companies" / "each company" → include every company
    if not companies and any(p in q_lower for p in _ALL_COMPANY_PHRASES):
        companies = list(COMPANIES)
    years = [str(y) for y in range(2010, 2030) if str(y) in query]
    return companies, years


def _expand_years_with_comparatives(companies: List[str], requested_years: List[str]) -> List[str]:
    """
    Expand requested years so that data is reachable via annual-report comparatives.

    Two expansion rules:
    1. Y+1 rule: if year Y is not available but Y+1 is, include Y+1 (annual
       reports always carry the prior year as a side-by-side column).
       Examples: Ford 2020 → add 2021;  Tesla 2021 → add 2022.

    2. Full-fallback rule: if after rule 1 none of the resolved years intersect
       with any available year for the company, include ALL available years.
       This catches multi-year historical summary tables, e.g. the BMW Group
       2021 report contains a 5-year table with 2017 data.
       Example: BMW 2017 → 2018 not available → fallback → add {2021,2022,2023}.
    """
    resolved: Set[str] = set(requested_years)
    scope = companies if companies else list(AVAILABLE_YEARS.keys())
    all_avail: Set[str] = {y for c in scope for y in AVAILABLE_YEARS.get(c, set())}

    for year in requested_years:
        next_year = str(int(year) + 1)
        for company in scope:
            if next_year in AVAILABLE_YEARS.get(company, set()):
                resolved.add(next_year)
                break

    # Rule 2: none of our resolved years are actually indexed → full fallback
    if not resolved & all_avail:
        resolved |= all_avail

    return sorted(resolved)


def build_vectorstore(persist_dir: str, embedding_model: str = "text-embedding-3-small") -> Chroma:
    embeddings = OpenAIEmbeddings(model=embedding_model)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


def ingest_documents(
    documents: List[Document],
    persist_dir: str,
    embedding_model: str = "text-embedding-3-small",
) -> Chroma:
    """Embed and persist documents. Always recreates the collection."""
    embeddings = OpenAIEmbeddings(model=embedding_model)
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )


def is_already_ingested(persist_dir: str) -> bool:
    if not Path(persist_dir).exists():
        return False
    try:
        return build_vectorstore(persist_dir)._collection.count() > 0
    except Exception:
        return False


class SmartFinancialRetriever:
    """
    Entity-aware ChromaDB retriever with three retrieval strategies:

    1. Single company + year(s) → filter by company + expanded years, k chunks.
    2. Multi-company             → per-company-per-year retrieval (2 chunks each)
                                   so every combination is represented.
    3. No entities detected     → full corpus similarity search.
    """

    def __init__(self, vectorstore: Chroma, k: int = 6):
        self.vectorstore = vectorstore
        self.k = k

    def get_relevant_documents(self, query: str) -> List[Document]:
        companies, years = extract_query_entities(query)
        expanded_years = _expand_years_with_comparatives(companies, years)

        # ── Strategy 1: Multi-company → retrieve per company × year ──────────
        if len(companies) > 1:
            return self._multi_company_retrieve(query, companies, expanded_years)

        # ── Strategy 2: Single company (with optional year filter) ────────────
        filter_dict = _build_filter(companies, expanded_years)
        if filter_dict:
            docs = self.vectorstore.similarity_search(query, k=self.k, filter=filter_dict)
            seen = {d.page_content for d in docs}

            # Secondary searches target consolidated financial statements so that
            # segment-level tables don't crowd out consolidated income-statement rows.
            # Use original requested years (not expanded) in sub-query TEXT so the
            # embedding stays anchored to the user's intended year, while the filter
            # (which uses expanded_years) still searches the right report(s).
            query_years = years if years else expanded_years

            if _is_profit_query(query):
                sub = f"{' '.join(companies)} {' '.join(query_years)} net income consolidated statements of operations"
                for d in self.vectorstore.similarity_search(sub, k=3, filter=filter_dict):
                    if d.page_content not in seen:
                        seen.add(d.page_content)
                        docs.append(d)
            elif _is_revenue_query(query):
                # Sub-query 1: income-statement style - catches "total revenues" / "interest income"
                sub1 = (
                    f"{' '.join(companies)} {' '.join(query_years)} "
                    "total revenues group revenues interest income consolidated"
                )
                for d in self.vectorstore.similarity_search(sub1, k=3, filter=filter_dict):
                    if d.page_content not in seen:
                        seen.add(d.page_content)
                        docs.append(d)
                # Sub-query 2: key-metrics style - catches summary tables ("Revenue ($M) ... Cash Flows")
                sub2 = (
                    f"{' '.join(companies)} {' '.join(query_years)} "
                    "revenue net income millions cash flows operating activities compared prior year"
                )
                for d in self.vectorstore.similarity_search(sub2, k=2, filter=filter_dict):
                    if d.page_content not in seen:
                        seen.add(d.page_content)
                        docs.append(d)

                # Sub-query 3: targeted search for the most recent requested year only -
                # ensures the latest year is not crowded out by earlier years' chunks.
                if query_years:
                    latest_year = max(query_years)
                    latest_filter = _build_filter(companies, [latest_year])
                    latest_sub = (
                        f"{' '.join(companies)} {latest_year} "
                        "total revenues income statement annual report"
                    )
                    for d in self.vectorstore.similarity_search(latest_sub, k=2, filter=latest_filter):
                        if d.page_content not in seen:
                            seen.add(d.page_content)
                            docs.append(d)
            return docs

        # ── Strategy 3: No entities - full corpus search ──────────────────────
        return self.vectorstore.similarity_search(query, k=self.k)

    def _multi_company_retrieve(
        self, query: str, companies: List[str], years: List[str]
    ) -> List[Document]:
        """
        For each (company, year) pair: run TWO searches and take the union.
          1. The original user query  - broad semantic match.
          2. A revenue-targeted query - ensures income-statement chunks surface.
        This prevents high-traffic pages (MD&A narrative) from crowding out the
        summary financial tables that contain the actual top-line revenue figures.
        """
        docs: List[Document] = []
        seen_ids: set = set()

        for company in companies:
            avail = AVAILABLE_YEARS.get(company, set())
            target_years = sorted(avail & set(years)) if years else sorted(avail)

            for year in target_years:
                f = _build_filter([company], [year])

                # Query 1: original user query - broad semantic match
                for d in self.vectorstore.similarity_search(query, k=2, filter=f):
                    if d.page_content not in seen_ids:
                        seen_ids.add(d.page_content)
                        docs.append(d)

                # Query 2: income-statement targeted - catches formal financial statement tables
                # BMW Finance N.V. uses "interest income" as revenue; BMW Group uses "revenues"
                targeted = f"{company} {year} total revenues group revenues interest income net income"
                for d in self.vectorstore.similarity_search(targeted, k=2, filter=f):
                    if d.page_content not in seen_ids:
                        seen_ids.add(d.page_content)
                        docs.append(d)

                # Query 3: key-metrics text - retrieves the prose chunk that contains
                # explicit year column labels (e.g. "2020 2021 H/(L)"), letting the LLM
                # resolve which column maps to which year in adjacent table chunks.
                header_q = f"{company} {year} GAAP financial measures company key metrics revenue cash flows compared"
                for d in self.vectorstore.similarity_search(header_q, k=2, filter=f):
                    if d.page_content not in seen_ids:
                        seen_ids.add(d.page_content)
                        docs.append(d)

        return docs

    def as_langchain_retriever(self):
        """Return a LangChain BaseRetriever wrapper."""
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        from langchain_core.retrievers import BaseRetriever

        outer = self

        class _Retriever(BaseRetriever):
            def _get_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> List[Document]:
                return outer.get_relevant_documents(query)

        return _Retriever()
