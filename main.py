"""
Car Financial RAG - CLI entry point.

Usage:
    python main.py            # interactive chat (ingests on first run)
    python main.py --ingest   # force re-ingestion of all PDFs
    python main.py --help
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path

from dotenv import load_dotenv
from rich import print as rprint
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.rule import Rule

load_dotenv()

console = Console()

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR.parent / "Data")))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "chroma_db"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SESSION_ID = "default"

BANNER = """
[bold cyan]Car Financial RAG[/bold cyan]
[dim]Automotive Sector Annual Report Analyst[/dim]
[dim]BMW · Ford · Tesla  |  2021–2023[/dim]

[italic]Type your question, or:[/italic]
  [bold]/clear[/bold]  - reset conversation history
  [bold]/quit[/bold]   - exit
"""


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest(force: bool = False):
    from src.document_processor import load_all_pdfs
    from src.vector_store import ingest_documents, is_already_ingested

    if not force and is_already_ingested(CHROMA_DB_PATH):
        console.print("[dim]Vector store already populated - skipping ingestion.[/dim]")
        console.print("[dim]Run with --ingest to force a rebuild.[/dim]\n")
        return

    console.print(Rule("[bold]Ingesting PDFs[/bold]"))

    pdf_paths = sorted(DATA_DIR.rglob("*.pdf"))
    if not pdf_paths:
        console.print(f"[red]No PDFs found under {DATA_DIR}[/red]")
        sys.exit(1)

    all_docs = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing PDFs…", total=len(pdf_paths))
        for path in pdf_paths:
            progress.update(task, description=f"[cyan]{path.name}[/cyan]")
            from src.document_processor import process_pdf
            docs = process_pdf(path)
            all_docs.extend(docs)
            progress.advance(task)

    console.print(f"[green]Extracted {len(all_docs):,} chunks from {len(pdf_paths)} PDFs.[/green]")

    with console.status("[bold]Embedding & storing in ChromaDB…[/bold]"):
        ingest_documents(all_docs, CHROMA_DB_PATH)

    console.print("[green bold]Ingestion complete.[/green bold]\n")


# ── Chat loop ─────────────────────────────────────────────────────────────────

def chat_loop(chain, clear_history_fn):
    console.print(Panel(BANNER, border_style="cyan"))

    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            console.print("[dim]Goodbye.[/dim]")
            break

        if user_input.lower() == "/clear":
            clear_history_fn(SESSION_ID)
            console.print("[dim]Conversation history cleared.[/dim]\n")
            continue

        with console.status("[bold cyan]Thinking…[/bold cyan]", spinner="dots"):
            try:
                result = chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": SESSION_ID}},
                )
                answer = result.get("answer", "No answer returned.")
            except Exception as exc:
                answer = f"[red]Error:[/red] {exc}"

        console.print()
        console.print(Rule("[bold blue]Assistant[/bold blue]", style="blue"))
        console.print(Markdown(answer))
        console.print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Car Financial RAG CLI")
    parser.add_argument("--ingest", action="store_true", help="Force re-ingestion of all PDFs")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]OPENAI_API_KEY not set. Create a .env file (see .env.example).[/red]")
        sys.exit(1)

    ingest(force=args.ingest)

    from src.vector_store import SmartFinancialRetriever, build_vectorstore
    from src.rag_chain import build_rag_chain

    vectorstore = build_vectorstore(CHROMA_DB_PATH)
    smart_retriever = SmartFinancialRetriever(vectorstore, k=6)
    retriever = smart_retriever.as_langchain_retriever()

    chain, clear_history_fn = build_rag_chain(retriever, model=OPENAI_MODEL)

    chat_loop(chain, clear_history_fn)


if __name__ == "__main__":
    main()
