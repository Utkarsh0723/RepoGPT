"""
app.py
------
Streamlit Web UI for the AI GitHub Repo Explainer.

Run with:
    streamlit run app.py
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

# ── Load environment variables from .env ──────────────────────────────────────
load_dotenv()

# ── Windows UTF-8 Fix ─────────────────────────────────────────────────────────
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
from utils import validate_github_url, normalize_github_url

# ── API Key (loaded from .env automatically) ──────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


# ──────────────────────────────────────────────────────────────────────────────
# Cached model loaders — only run ONCE, then reused across reruns
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def cached_embedding_model():
    return get_embedding_model(use_openai=False)


@st.cache_resource(show_spinner="Loading LLM & building QA chain...")
def cached_qa_chain(_vector_store, model_name: str = "llama-3.3-70b-versatile"):
    chain, retriever = build_qa_chain(
        _vector_store,
        llm_provider="groq",
        api_key=GROQ_API_KEY,
        model_name=model_name,
    )
    return chain


# ──────────────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Repo Explainer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2.5rem 2rem; border-radius: 16px; text-align: center;
        margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.05);
    }
    .hero h1 { color: #fff; font-size: 2.4rem; font-weight: 700; margin-bottom: 0.3rem; }
    .hero p { color: #a0aec0; font-size: 1.05rem; margin: 0; }

    .stat-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.06); border-radius: 12px;
        padding: 1.2rem 1.4rem; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stat-card .stat-value {
        font-size: 1.8rem; font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .stat-card .stat-label {
        color: #718096; font-size: 0.82rem; text-transform: uppercase;
        letter-spacing: 1px; margin-top: 0.25rem;
    }

    .user-msg {
        background: linear-gradient(135deg, #667eea, #764ba2); color: #fff;
        padding: 0.9rem 1.3rem; border-radius: 18px 18px 4px 18px;
        margin: 0.6rem 0; max-width: 80%; margin-left: auto; font-size: 0.95rem;
        box-shadow: 0 3px 12px rgba(102,126,234,0.25);
    }
    .bot-msg {
        background: linear-gradient(145deg, #1e1e30, #252540); color: #e2e8f0;
        padding: 1rem 1.3rem; border-radius: 18px 18px 18px 4px; margin: 0.6rem 0;
        max-width: 90%; font-size: 0.95rem; line-height: 1.65;
        border: 1px solid rgba(255,255,255,0.05);
        box-shadow: 0 3px 12px rgba(0,0,0,0.2);
        white-space: pre-wrap;
    }
    .source-tag {
        display: inline-block; background: rgba(102,126,234,0.15); color: #667eea;
        padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.78rem;
        margin: 0.2rem 0.25rem 0.2rem 0; border: 1px solid rgba(102,126,234,0.2);
    }

    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0c29, #1a1a2e); }
    section[data-testid="stSidebar"] .stMarkdown h2 { color: #e2e8f0; }

    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important; border: none !important; border-radius: 8px !important;
        padding: 0.55rem 1.6rem !important; font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(102,126,234,0.45) !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stats" not in st.session_state:
    st.session_state.stats = {}
if "indexed" not in st.session_state:
    st.session_state.indexed = False


# ──────────────────────────────────────────────────────────────────────────────
# Hero Banner
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔍 AI Repo Explainer</h1>
    <p>Paste a GitHub URL → Ask anything about the codebase</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    model_name = st.selectbox(
        "🤖 AI Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
        help="llama-3.3-70b gives the best answers. Powered by Groq.",
    )

    st.markdown("---")
    st.markdown("## 🌐 Repository")

    repo_url = st.text_input(
        "GitHub URL",
        placeholder="https://github.com/owner/repo",
    )

    col1, col2 = st.columns(2)
    with col1:
        index_btn = st.button("🚀 Index Repo", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Clear Chat", use_container_width=True)

    if clear_btn:
        st.session_state.messages = []
        st.rerun()

    # Reuse existing index
    if vector_store_exists():
        st.success("📦 Existing FAISS index found.")
        if st.button("♻️ Reuse Existing Index", use_container_width=True):
            try:
                emb = cached_embedding_model()
                vs = load_vector_store(emb)
                chain = cached_qa_chain(vs, model_name=model_name)
                st.session_state.chain = chain
                st.session_state.indexed = True
                st.success("✅ Ready! Ask a question.")
            except Exception as exc:
                st.error(f"❌ Error: {exc}")

    st.markdown("---")
    st.markdown(
        "<p style='color:#4a5568;font-size:0.75rem;text-align:center;'>"
        "Powered by Groq · LangChain · FAISS<br/>Python + Streamlit</p>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Index button handler
# ──────────────────────────────────────────────────────────────────────────────
if index_btn:
    if not repo_url.strip():
        st.error("Please enter a GitHub URL in the sidebar.")
    else:
        url = normalize_github_url(repo_url)
        if not validate_github_url(url):
            st.error(f"❌ `{url}` doesn't look like a valid GitHub URL.")
        else:
            st.session_state.messages = []
            repo_path = None

            try:
                with st.status("🚀 Indexing repository...", expanded=True) as status:
                    st.write("📥 Cloning repository...")
                    repo_path = clone_repository(url)
                    st.write("✅ Clone complete!")

                    st.write("📄 Reading code files...")
                    file_contents = load_files_from_repo(repo_path)
                    if not file_contents:
                        st.error("No supported files found.")
                        st.stop()
                    st.write(f"✅ Found **{len(file_contents)}** files")

                    st.write("📝 Splitting into chunks...")
                    documents = build_documents(file_contents)
                    chunks = split_documents(documents)
                    st.write(f"✅ Created **{len(chunks)}** chunks")

                    st.write("⚙️ Building FAISS index...")
                    emb = cached_embedding_model()
                    vs = create_vector_store(chunks, emb)
                    st.write("💾 Index saved!")

                    st.write("🤖 Building QA chain...")
                    chain = cached_qa_chain(vs, model_name=model_name)
                    st.session_state.chain = chain
                    st.session_state.indexed = True
                    st.session_state.stats = {
                        "files": len(file_contents),
                        "chunks": len(chunks),
                        "repo": url.split("/")[-1],
                    }
                    status.update(label="✅ Indexing complete!", state="complete")

            except Exception as exc:
                st.error(f"❌ Error: {exc}")
            finally:
                if repo_path:
                    cleanup_repo(repo_path)


# ──────────────────────────────────────────────────────────────────────────────
# Stats row
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.stats:
    s = st.session_state.stats
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="stat-card"><div class="stat-value">{s.get("repo","—")}</div>'
            f'<div class="stat-label">Repository</div></div>', unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="stat-card"><div class="stat-value">{s.get("files",0)}</div>'
            f'<div class="stat-label">Files Indexed</div></div>', unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="stat-card"><div class="stat-value">{s.get("chunks",0)}</div>'
            f'<div class="stat-label">Chunks Created</div></div>', unsafe_allow_html=True,
        )
    st.markdown("<br/>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Chat interface
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.indexed:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-msg">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            sources_html = ""
            if msg.get("sources"):
                tags = "".join(
                    f'<span class="source-tag">📄 {s}</span>' for s in msg["sources"]
                )
                sources_html = f"<div style='margin-top:0.6rem;'>{tags}</div>"
            st.markdown(
                f'<div class="bot-msg">{msg["content"]}{sources_html}</div>',
                unsafe_allow_html=True,
            )

    # Suggested questions
    if not st.session_state.messages:
        st.markdown("#### 💡 Try asking:")
        suggestions = [
            "What does this project do?",
            "Explain the main function",
            "What libraries are used?",
            "Explain the folder structure",
        ]
        cols = st.columns(len(suggestions))
        for i, q in enumerate(suggestions):
            with cols[i]:
                if st.button(q, key=f"sug_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": q})
                    with st.spinner("🔍 Searching..."):
                        result = ask_question(st.session_state.chain, q)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    })
                    st.rerun()

    # Chat input
    user_input = st.chat_input("Ask a question about the repository...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("🔍 Searching repository..."):
            result = ask_question(st.session_state.chain, user_input)
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })
        st.rerun()

else:
    st.markdown("""
    <div style="text-align:center; padding:3rem 1rem; color:#718096;">
        <p style="font-size:3rem; margin-bottom:0.5rem;">📂</p>
        <h3 style="color:#e2e8f0; font-weight:600;">No repository indexed yet</h3>
        <p style="font-size:1rem; max-width:500px; margin:0 auto;">
            Paste a public GitHub URL in the sidebar and click
            <strong>🚀 Index Repo</strong> to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)
