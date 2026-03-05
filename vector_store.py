"""
vector_store.py
---------------
Responsible for:
1. Creating a FAISS vector store from document chunks + embeddings.
2. Persisting the FAISS index to disk so you don't re-embed on every run.
3. Loading a previously saved FAISS index from disk.
"""

import os
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# Default directory where the FAISS index will be saved
DEFAULT_INDEX_DIR = "./faiss_index"


def create_vector_store(
    chunks: List[Document],
    embedding_model,
    index_dir: str = DEFAULT_INDEX_DIR,
) -> FAISS:
    """
    Embed all document chunks and store them in a FAISS vector database.
    The index is then saved to disk for later reuse.

    FAISS (Facebook AI Similarity Search) keeps everything in RAM and performs
    extremely fast nearest-neighbour lookups – perfect for a local tool like this.

    Args:
        chunks:          Chunked LangChain Documents from embeddings.split_documents().
        embedding_model: A LangChain Embeddings instance.
        index_dir:       Directory path where the FAISS index will be saved.

    Returns:
        A LangChain FAISS vector store ready for similarity search.
    """
    if not chunks:
        raise ValueError("Cannot create a vector store from an empty chunk list.")

    print(f"⚙️   Embedding {len(chunks)} chunks and building FAISS index …")
    print("    (This may take a minute for large repositories)\n")

    # FAISS.from_documents embeds every chunk and builds the index in one call
    vector_store = FAISS.from_documents(chunks, embedding_model)

    # Persist to disk so future runs can skip re-embedding
    os.makedirs(index_dir, exist_ok=True)
    vector_store.save_local(index_dir)
    print(f"💾  FAISS index saved to: {os.path.abspath(index_dir)}\n")

    return vector_store


def load_vector_store(
    embedding_model,
    index_dir: str = DEFAULT_INDEX_DIR,
) -> FAISS:
    """
    Load a previously saved FAISS index from disk.

    Args:
        embedding_model: Must be the SAME embedding model used when creating the index.
        index_dir:       Directory where the FAISS index was saved.

    Returns:
        A LangChain FAISS vector store.

    Raises:
        FileNotFoundError: If no saved index is found at index_dir.
    """
    if not os.path.exists(index_dir):
        raise FileNotFoundError(
            f"No FAISS index found at '{index_dir}'. "
            "Run the tool with a GitHub URL first to create one."
        )

    print(f"📂  Loading existing FAISS index from: {os.path.abspath(index_dir)}\n")
    # allow_dangerous_deserialization=True is required by newer LangChain versions
    vector_store = FAISS.load_local(
        index_dir,
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    return vector_store


def vector_store_exists(index_dir: str = DEFAULT_INDEX_DIR) -> bool:
    """
    Check whether a saved FAISS index already exists at index_dir.

    Args:
        index_dir: Directory to check.

    Returns:
        True if the index directory exists and contains index files, else False.
    """
    index_file = os.path.join(index_dir, "index.faiss")
    return os.path.isfile(index_file)
