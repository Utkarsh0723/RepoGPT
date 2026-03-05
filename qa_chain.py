"""
qa_chain.py
-----------
Responsible for:
1. Building a RAG (Retrieval-Augmented Generation) chain using LCEL.
2. Wiring together the LLM, vector store retriever, and prompt template.
3. Answering user questions by retrieving relevant chunks then generating a response.

Supports three LLM backends:
  - Groq (FREE, recommended) — uses Llama-3.3-70B via Groq cloud
  - OpenAI (paid) — GPT-3.5/4
  - HuggingFace (FREE, local) — flan-t5-small (low quality, fallback only)
"""

from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are an expert software engineer and code reviewer.
You have been given relevant excerpts from a GitHub repository's source code and documentation.
Use ONLY the provided context to answer the question accurately.

Important rules:
- Give a clear, concise, human-readable answer
- DO NOT just repeat the raw code — summarize and explain it
- Reference specific files or functions when relevant
- If the answer is not found in the context, say: "I couldn't find relevant information in this repository for that question."

Context (extracted from the repository):
-----------------------------------------
{context}
-----------------------------------------

Question: {question}

Answer:"""


def _format_docs(docs: List[Document]) -> str:
    """Concatenate retrieved documents into a single context string."""
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[File: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_qa_chain(
    vector_store: FAISS,
    llm_provider: str = "groq",
    api_key: str = "",
    model_name: str = "",
    num_retrieved_chunks: int = 4,
):
    """
    Build a RAG chain using LCEL (LangChain Expression Language).

    Args:
        vector_store:          Populated FAISS vector store.
        llm_provider:          One of "groq", "openai", or "huggingface".
        api_key:               API key (required for groq and openai).
        model_name:            Model name override. If empty, uses sensible defaults.
        num_retrieved_chunks:  How many chunks to retrieve per question (k).

    Returns:
        A tuple of (lcel_chain, retriever).
    """
    # ── 1. Build the LLM ────────────────────────────────────────────────────
    if llm_provider == "groq":
        if not api_key:
            raise ValueError(
                "Groq API key required. Get a FREE key at https://console.groq.com/keys"
            )
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model_name=model_name or "llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=api_key,
            max_tokens=1024,
        )
        print(f"🚀  LLM: Groq ({model_name or 'llama-3.3-70b-versatile'})\n")

    elif llm_provider == "openai":
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env variable."
            )
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model_name=model_name or "gpt-3.5-turbo",
            temperature=0,
            openai_api_key=api_key,
        )
        print(f"🤖  LLM: OpenAI ({model_name or 'gpt-3.5-turbo'})\n")

    else:  # huggingface (local fallback)
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
        from langchain_huggingface import HuggingFacePipeline

        hf_model = model_name or "google/flan-t5-small"
        print(f"🤗  LLM: HuggingFace pipeline ({hf_model}) [local]\n")

        tokenizer = AutoTokenizer.from_pretrained(hf_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(hf_model)
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
        )
        llm = HuggingFacePipeline(pipeline=pipe)

    # ── 2. Build the Retriever ───────────────────────────────────────────────
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": num_retrieved_chunks},
    )

    # ── 3. Build the Prompt ──────────────────────────────────────────────────
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    # ── 4. Build the LCEL Chain ──────────────────────────────────────────────
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: _format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    print("✅  QA chain is ready. You can now ask questions!\n")
    return rag_chain_with_source, retriever


def ask_question(chain, question: str) -> dict:
    """
    Pass a user question through the RAG chain and return the result.

    Args:
        chain:    A built LCEL RAG chain.
        question: The user's natural-language question.

    Returns:
        Dict with keys:
          - 'answer'  : The LLM's answer string.
          - 'sources' : List of source file paths used to form the answer.
    """
    if not question.strip():
        return {"answer": "Please enter a non-empty question.", "sources": []}

    result = chain.invoke(question)

    answer = result.get("answer", "No answer generated.")
    source_docs = result.get("context", [])
    sources = list({doc.metadata.get("source", "unknown") for doc in source_docs})

    return {"answer": answer, "sources": sources}
