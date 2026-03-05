# 🔍 AI GitHub Repo Explainer

An AI-powered CLI tool that **clones any public GitHub repository** and lets you ask natural-language questions about the codebase using **LangChain + FAISS + HuggingFace/OpenAI**.

---

## ✨ Features

| Feature | Detail |
|---|---|
| Clone any public GitHub repo | Uses GitPython (shallow clone – fast) |
| Smart file filtering | Only reads `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.md`, `.json`, and more |
| Large-file protection | Skips files > 200 KB automatically |
| Semantic search | FAISS vector store for fast nearest-neighbour retrieval |
| Two LLM options | **Free**: HuggingFace `flan-t5-large` · **Paid**: OpenAI GPT-3.5/4 |
| Persistent index | FAISS index saved to disk – reuse without re-embedding |
| Source attribution | Every answer shows which files were used as context |

---

## 📁 Project Structure

```
repo-explainer/
│
├── main.py          # CLI entry point – orchestrates the full pipeline
├── repo_loader.py   # Clone repo & read code files
├── embeddings.py    # Build LangChain Documents, split into chunks, embedding model factory
├── vector_store.py  # Create / save / load FAISS vector index
├── qa_chain.py      # RetrievalQA chain with custom code-aware prompt
├── utils.py         # Shared helpers (banner, URL validation, env keys, prompts)
├── requirements.txt # All Python dependencies with pinned versions
└── README.md        # This file
```

---

## 🚀 How to Run Locally

### 1 · Prerequisites

- Python 3.10 or 3.11 (recommended)
- Git installed and on your PATH
- ~2 GB free disk space (for HuggingFace model cache)

---

### 2 · Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

---

### 3 · Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run downloads the HuggingFace models (~400 MB total). Subsequent runs are instant.

---

### 4 · (Optional) Set Your OpenAI API Key

Only required if you want to use GPT instead of the free HuggingFace model.

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."

# Windows Command Prompt
set OPENAI_API_KEY=sk-...

# macOS / Linux
export OPENAI_API_KEY=sk-...
```

Or create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-...
```

---

### 5 · Run the Tool

#### Option A – Free (HuggingFace, no API key needed)
```bash
python main.py
```

#### Option B – OpenAI GPT-3.5-Turbo
```bash
python main.py --openai
```

#### Option C – OpenAI GPT-4
```bash
python main.py --openai --model gpt-4
```

---

### 6 · Example Session

```
╔══════════════════════════════════════════════════════╗
║        🔍  AI GitHub Repo Explainer  🔍              ║
║  Ask anything about any public GitHub repository!    ║
╚══════════════════════════════════════════════════════╝

🌐  Enter the GitHub repository URL you want to explore.
    Example: https://github.com/openai/openai-python

🔗  GitHub URL: https://github.com/tiangolo/fastapi

📥  Cloning: https://github.com/tiangolo/fastapi
📂  Destination: C:\Users\...\AppData\Local\Temp\repo_explainer_abc123
✅  Clone successful!

📄  Loaded 87 file(s) from the repository.
📝  Created 87 Document object(s).
🔪  Split into 412 chunk(s) (chunk_size=1000, overlap=150).
⚙️   Embedding 412 chunks and building FAISS index …
💾  FAISS index saved to: .\faiss_index
✅  QA chain is ready. You can now ask questions!

──────────────────────────────────────────────────────────────
💬  Interactive Q&A Session
──────────────────────────────────────────────────────────────

❓  Your question: What does this project do?

🔍  Searching repository …

💡  Answer:
····························································
FastAPI is a modern, fast web framework for building APIs
with Python based on standard Python type hints. It offers
automatic interactive API documentation, data validation,
and very high performance comparable to NodeJS and Go.
····························································

📎  Sources used:
    • README.md
    • fastapi/applications.py

❓  Your question: What libraries are used?
...
```

---

## 🛠️ Configuration

| Setting | Where | Default |
|---|---|---|
| Supported file extensions | `repo_loader.py` → `SUPPORTED_EXTENSIONS` | `.py .js .ts .java .cpp .md ...` |
| Max file size | `repo_loader.py` → `MAX_FILE_SIZE_BYTES` | 200 KB |
| Chunk size | `embeddings.py` → `CHUNK_SIZE` | 1 000 chars |
| Chunk overlap | `embeddings.py` → `CHUNK_OVERLAP` | 150 chars |
| Embedding model (HF) | `embeddings.py` → `get_embedding_model()` | `all-MiniLM-L6-v2` |
| LLM (HF) | `qa_chain.py` → `build_qa_chain()` | `flan-t5-large` |
| Retrieved chunks (k) | `qa_chain.py` → `num_retrieved_chunks` | 5 |
| FAISS index directory | `vector_store.py` → `DEFAULT_INDEX_DIR` | `./faiss_index` |

---

## ❓ Example Questions to Ask

```
What does this project do?
Explain the main function.
What libraries or frameworks are used?
How does authentication work?
Where is the database configured?
What does the README say about installation?
Explain the folder structure.
How are API routes defined?
```

---

## 📖 How It Works (Pipeline)

```
GitHub URL
    │
    ▼
[repo_loader.py]  ──► Clone repo (GitPython, shallow)
    │                  Read .py/.js/.ts/... files (skip large/binary)
    ▼
[embeddings.py]   ──► Wrap as LangChain Documents
    │                  Split into 1 000-char chunks (RecursiveCharacterTextSplitter)
    │                  Load embedding model (HuggingFace or OpenAI)
    ▼
[vector_store.py] ──► Embed chunks → FAISS index (saved to ./faiss_index/)
    ▼
[qa_chain.py]     ──► RetrievalQA chain
    │                  On each question:
    │                    1. Embed question
    │                    2. Retrieve top-5 similar chunks from FAISS
    │                    3. Feed chunks + question into LLM with custom prompt
    │                    4. Return answer + source files
    ▼
[main.py]         ──► CLI loop – reads questions, prints answers
```

---

## ⚠️ Limitations

- Works only with **public** GitHub repositories.
- HuggingFace `flan-t5-large` is good for short answers; for complex code questions, use `--openai`.
- Very large repositories (thousands of files) will take longer to embed on first run.

---

## 📜 License

MIT – free to use, modify, and distribute.
