# 📄 DocuBot 🤖

_A Document Question-Answering Bot using FastAPI, Streamlit, LangChain, HuggingFace & Groq_

---

## 🚀 Overview

DocuBot is an intelligent Q&A assistant that allows you to upload **PDF/Word documents** and ask natural language questions about their content.  
It uses:

- **FastAPI** → Backend for document processing & retrieval-based QA
- **Streamlit** → Simple frontend UI
- **LangChain + Chroma** → For text splitting, embeddings & vector search
- **HuggingFace Transformers** → Free sentence embeddings (`all-MiniLM-L6-v2`)
- **Groq LLM** → Fast inference with `llama-3.1-8b-instant`

---

## 🛠 Features

- Upload `.pdf` and `.docx` documents
- Automatic text extraction & chunking
- Vector embeddings for semantic retrieval
- Ask natural questions about the content
- Displays **answers + retrieved sources**
- Fallback to full-document answer if retrieval fails
