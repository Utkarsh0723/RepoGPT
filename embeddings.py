"""
embeddings.py
-------------
Responsible for:
1. Converting (path, content) pairs into LangChain Document objects.
2. Splitting large documents into smaller chunks using RecursiveCharacterTextSplitter.
3. Building an embedding model (HuggingFace by default, OpenAI optional).
"""

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# Chunking Configuration
# ---------------------------------------------------------------------------

# Each chunk is at most CHUNK_SIZE characters long.
# CHUNK_OVERLAP characters are shared between consecutive chunks so that
# context is not completely lost at boundaries.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def build_documents(file_contents: List[Tuple[str, str]]) -> List[Document]:
    """
    Wrap raw (path, content) pairs as LangChain Document objects.

    The relative file path is stored in the document metadata so we can
    display where a retrieved chunk came from.

    Args:
        file_contents: List of (relative_path, raw_text) tuples.

    Returns:
        List of LangChain Document objects.
    """
    docs: List[Document] = []
    for relative_path, content in file_contents:
        doc = Document(
            page_content=content,
            metadata={"source": relative_path},
        )
        docs.append(doc)
    print(f"📝  Created {len(docs)} Document object(s).")
    return docs


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller, overlapping chunks.

    LangChain's RecursiveCharacterTextSplitter tries to split on
    paragraph breaks → newlines → spaces, keeping chunks as semantically
    whole as possible.

    Args:
        documents: Full-file LangChain Documents.

    Returns:
        List of chunked LangChain Documents (more items, smaller text).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,  # stores character start position in metadata
    )
    chunks = splitter.split_documents(documents)
    print(f"🔪  Split into {len(chunks)} chunk(s) (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).\n")
    return chunks


def get_embedding_model(use_openai: bool = False, openai_api_key: str = ""):
    """
    Return an embedding model instance.

    Two options:
    - HuggingFace (default, FREE, runs locally):
        Uses 'all-MiniLM-L6-v2' – small, fast, good quality.
    - OpenAI (requires an API key):
        Uses 'text-embedding-ada-002' – higher quality but costs money.

    Args:
        use_openai:    Set True to use OpenAI embeddings.
        openai_api_key: Your OpenAI API key (only needed when use_openai=True).

    Returns:
        A LangChain Embeddings object.
    """
    if use_openai:
        if not openai_api_key:
            raise ValueError(
                "OpenAI API key is required when use_openai=True. "
                "Set it via the OPENAI_API_KEY environment variable or pass it directly."
            )
        from langchain_openai import OpenAIEmbeddings
        print("🔑  Using OpenAI embeddings (text-embedding-ada-002).\n")
        return OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Default: free HuggingFace model (downloads once, cached locally)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"🤗  Using HuggingFace embeddings: {model_name}")
    print("    (Model downloads automatically on first run – ~90 MB)\n")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},     # use "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True},
    )
