# 🔍 RepoGPT: AI GitHub Repo Explainer

**RepoGPT** is a powerful AI-driven tool that allows you to explore and understand any public GitHub repository through natural language. Built with **LangChain**, **FAISS**, and **Groq**, it provides a stunning Streamlit web interface for interactive codebase analysis.

---

## ✨ Features

- **🚀 Live Chat:** Ask questions about the code and get instant, intelligent answers.
- **⚡ Groq Speed:** Powered by Llama-3.3-70B on Groq for lightning-fast, high-quality responses.
- **📂 Smart Indexing:** Clones repos, chunks files, and builds a FAISS vector store automatically.
- **♻️ Index Persistence:** Saves your indexed repositories to disk—load them instantly next time.
- **📄 Source Attribution:** See exactly which files the AI used to answer your question.
- **🎨 Premium UI:** Beautiful dark-themed dashboard with real-time indexing stats.

---

## 🚀 Quick Start

### 1. Clone the Project
```bash
git clone https://github.com/Utkarsh0723/RepoGPT.git
cd RepoGPT
```

### 2. Setup Environment
```bash
# Create virtual environment
python -m venv venv
# Activate (Windows)
.\venv\Scripts\activate
# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Add API Key
Create a `.env` file in the root directory:
```text
GROQ_API_KEY=gsk_your_free_key_here
```
> Get your **FREE** API key at [console.groq.com](https://console.groq.com/keys).

### 4. Run the App
```bash
# Recommended for Windows terminal emoji support:
$env:PYTHONIOENCODING="utf-8"
streamlit run app.py
```

---

## 🛠️ Tech Stack

- **[Streamlit](https://streamlit.io/):** For the interactive web dashboard.
- **[LangChain](https://www.langchain.com/):** For RAG (Retrieval-Augmented Generation) orchestration.
- **[Groq](https://groq.com/):** For high-speed LLM inference (Llama-3.3-70B).
- **[FAISS](https://github.com/facebookresearch/faiss):** For efficient vector similarity search.
- **[Sentence-Transformers](https://huggingface.co/sentence-transformers):** For local embeddings (`all-MiniLM-L6-v2`).
- **[GitPython](https://gitpython.readthedocs.io/):** For automated repository cloning.

---

## 📂 Project Structure

- `app.py`: Main Streamlit application and UI logic.
- `qa_chain.py`: RAG pipeline configuration (Groq/OpenAI/HuggingFace).
- `embeddings.py`: Document processing, chunking, and embedding.
- `vector_store.py`: FAISS index creation and management.
- `repo_loader.py`: GitHub cloning and file loading logic.
- `utils.py`: Shared helper functions and validation.

---


