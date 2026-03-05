"""
main.py
-------
Entry point for the AI GitHub Repo Explainer CLI.

Flow:
  1. User enters a GitHub repository URL.
  2. The repo is cloned, files are read, split, and embedded into FAISS.
  3. A RetrievalQA chain is built on top of the FAISS store.
  4. The user enters an interactive question-answering loop.
  5. Type 'exit' or 'quit' to leave the session.

Usage:
  python main.py [--openai] [--model gpt-4]

Options:
  --openai    Use OpenAI GPT instead of the free HuggingFace model.
              Requires OPENAI_API_KEY environment variable to be set.
  --model     OpenAI model name (default: gpt-3.5-turbo). Ignored without --openai.
"""

import argparse
import sys
import os

# ── Windows UTF-8 Fix ─────────────────────────────────────────────────────────
# Force UTF-8 output on Windows so emoji in print statements render correctly.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


from repo_loader import clone_repository, load_files_from_repo, cleanup_repo
from embeddings import build_documents, split_documents, get_embedding_model
from vector_store import (
    create_vector_store,
    load_vector_store,
    vector_store_exists,
    DEFAULT_INDEX_DIR,
)
from qa_chain import build_qa_chain, ask_question
from utils import (
    print_banner,
    print_separator,
    print_sources,
    get_openai_key,
    require_openai_key,
    validate_github_url,
    normalize_github_url,
    prompt_yes_no,
)


# ── CLI Argument Parsing ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI GitHub Repo Explainer – Ask questions about any GitHub repo."
    )
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Use OpenAI GPT instead of the free HuggingFace model.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="OpenAI model name (default: gpt-3.5-turbo). Only used with --openai.",
    )
    return parser.parse_args()


# ── Repository Ingestion ──────────────────────────────────────────────────────

def ingest_repository(github_url: str, embedding_model) -> None:
    """
    Clone → read files → split → embed → save FAISS index.
    This is the expensive path; run once per repo.
    """
    repo_path = None
    try:
        # Step 1: Clone
        repo_path = clone_repository(github_url)

        # Step 2: Load files
        file_contents = load_files_from_repo(repo_path)
        if not file_contents:
            print("❌  No supported files found in the repository. Exiting.")
            sys.exit(1)

        # Step 3: Build LangChain Documents
        documents = build_documents(file_contents)

        # Step 4: Split into chunks
        chunks = split_documents(documents)

        # Step 5 & 6: Embed + store in FAISS
        create_vector_store(chunks, embedding_model)

    finally:
        # Always clean up the cloned repo to save disk space
        if repo_path:
            cleanup_repo(repo_path)


# ── Interactive Q&A Loop ──────────────────────────────────────────────────────

def interactive_qa_loop(chain) -> None:
    """Run the user through an interactive question-answering session."""
    print_separator()
    print("💬  Interactive Q&A Session")
    print("    Type your question and press Enter.")
    print("    Type 'exit' or 'quit' to end the session.")
    print_separator()
    print()

    while True:
        try:
            question = input("❓  Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋  Session ended. Goodbye!")
            break

        if question.lower() in ("exit", "quit", "q", "bye"):
            print("\n👋  Goodbye! Come back anytime to explore more repos.")
            break

        if not question:
            print("    (Please type a question or 'exit' to quit)\n")
            continue

        print("\n🔍  Searching repository …\n")
        result = ask_question(chain, question)

        print("💡  Answer:")
        print_separator(char="·")
        print(result["answer"])
        print_separator(char="·")
        print_sources(result["sources"])
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    print_banner()

    # ── Resolve API key ──────────────────────────────────────────────────────
    openai_api_key = ""
    if args.openai:
        try:
            openai_api_key = require_openai_key()
        except EnvironmentError as exc:
            print(exc)
            sys.exit(1)

    # ── Build the embedding model (same model MUST be used for create AND load) ─
    embedding_model = get_embedding_model(
        use_openai=args.openai,
        openai_api_key=openai_api_key,
    )

    # ── Decide whether to ingest a new repo or reuse existing index ──────────
    if vector_store_exists(DEFAULT_INDEX_DIR):
        print(f"📦  Found existing FAISS index at '{DEFAULT_INDEX_DIR}'.")
        reuse = prompt_yes_no("Would you like to reuse the existing index?", default=True)
        if not reuse:
            # User wants a fresh index for a new repo
            ingest_new = True
        else:
            ingest_new = False
    else:
        ingest_new = True

    if ingest_new:
        # ── Get GitHub URL from user ─────────────────────────────────────────
        while True:
            print("\n🌐  Enter the GitHub repository URL you want to explore.")
            print("    Example: https://github.com/openai/openai-python\n")
            github_url = input("🔗  GitHub URL: ").strip()
            github_url = normalize_github_url(github_url)

            if not validate_github_url(github_url):
                print(
                    f"❌  '{github_url}' doesn't look like a valid GitHub URL.\n"
                    "    Format: https://github.com/<owner>/<repo>\n"
                )
                continue
            break

        # ── Ingest the repository ────────────────────────────────────────────
        print(f"\n🚀  Ingesting repository: {github_url}\n")
        print_separator()
        ingest_repository(github_url, embedding_model)

    # ── Load the FAISS index ─────────────────────────────────────────────────
    vector_store = load_vector_store(embedding_model, DEFAULT_INDEX_DIR)

    # ── Build the QA chain ───────────────────────────────────────────────────
    chain, _ = build_qa_chain(
        vector_store=vector_store,
        use_openai=args.openai,
        openai_api_key=openai_api_key,
        model_name=args.model,
    )

    # ── Start the Q&A session ────────────────────────────────────────────────
    interactive_qa_loop(chain)


if __name__ == "__main__":
    main()
