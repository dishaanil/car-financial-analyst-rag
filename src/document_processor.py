"""
PDF document processor with pdfplumber for accurate table extraction.

Key insight from FinQA (Chen et al., 2021): financial tables need to be
kept intact as semantic units — naive text splitting destroys table structure
and makes numerical reasoning impossible.
"""

import re
from pathlib import Path
from typing import List, Tuple

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

COMPANIES = ["BMW", "Ford", "Tesla"]

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _extract_company_and_year(pdf_path: Path) -> Tuple[str, str]:
    """Infer company and year from file path (name or parent directory)."""
    parts = [pdf_path.stem, pdf_path.parent.name]
    text = " ".join(parts)

    company = next((c for c in COMPANIES if c.lower() in text.lower()), "Unknown")

    match = re.search(r"20\d{2}", text)
    year = match.group() if match else "Unknown"

    return company, year


def _table_to_markdown(table: List[List]) -> str:
    """
    Convert a pdfplumber table (list of rows, each row a list of cells)
    to a GitHub-flavoured markdown table string.
    """
    if not table or not any(table):
        return ""

    cleaned_rows = []
    for row in table:
        cells = [str(cell).strip().replace("\n", " ") if cell is not None else "" for cell in row]
        cleaned_rows.append(cells)

    if not cleaned_rows:
        return ""

    col_count = max(len(r) for r in cleaned_rows)

    lines = []
    for i, row in enumerate(cleaned_rows):
        padded = row + [""] * (col_count - len(row))
        lines.append("| " + " | ".join(padded) + " |")
        if i == 0:
            lines.append("| " + " | ".join(["---"] * col_count) + " |")

    return "\n".join(lines)


def process_pdf(pdf_path: Path) -> List[Document]:
    """
    Extract both prose text and tables from a PDF with full metadata.

    Strategy:
      - Tables  → single Document per table (never split; see FinQA paper).
      - Prose   → RecursiveCharacterTextSplitter chunks of ~1200 chars.
    Each Document carries company, year, page, and chunk_type metadata for
    smart ChromaDB filtering at query time.
    """
    company, year = _extract_company_and_year(pdf_path)
    base_meta = {"company": company, "year": year, "source": pdf_path.name}

    documents: List[Document] = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        for page_num, page in enumerate(pdf.pages, 1):
            page_meta = {**base_meta, "page": page_num, "total_pages": total_pages}

            # ── Tables ────────────────────────────────────────────────────────
            tables = page.extract_tables() or []
            for table in tables:
                if not table or len(table) < 2:
                    continue
                md = _table_to_markdown(table)
                if not md.strip():
                    continue
                header = f"[{company} Annual Report {year} | Page {page_num} | TABLE]\n\n"
                documents.append(
                    Document(
                        page_content=header + md,
                        metadata={**page_meta, "chunk_type": "table"},
                    )
                )

            # ── Prose text ────────────────────────────────────────────────────
            raw_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            raw_text = re.sub(r"\n{3,}", "\n\n", raw_text).strip()
            if not raw_text:
                continue

            header = f"[{company} Annual Report {year} | Page {page_num}/{total_pages}]\n\n"
            full_text = header + raw_text

            # Split long pages; short pages become a single chunk.
            chunks = TEXT_SPLITTER.split_text(full_text)
            for chunk in chunks:
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={**page_meta, "chunk_type": "text"},
                    )
                )

    return documents


def load_all_pdfs(data_dir: Path) -> List[Document]:
    """Walk data_dir recursively, processing only PDFs from known companies."""
    pdf_paths = sorted(data_dir.rglob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found under {data_dir}")

    # Skip PDFs that can't be associated with a known company
    known = [p for p in pdf_paths if _extract_company_and_year(p)[0] != "Unknown"]
    skipped = len(pdf_paths) - len(known)
    if skipped:
        print(f"[document_processor] Skipping {skipped} PDFs with no recognisable company name.")

    all_docs: List[Document] = []
    for path in known:
        all_docs.extend(process_pdf(path))

    return all_docs
