# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os

app = FastAPI()

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    os.makedirs("temp_files", exist_ok=True)
    file_location = f"temp_files/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    return {"file_name": file.filename, "file_location": file_location}


# ------- LangChain & deps -------
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

# --------- Loaders ----------
def load_document(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type (use .pdf or .docx)")
    return loader.load()

# --------- Chunking ----------
def split_document(documents):
    # Bigger chunks give the LLM more context to answer
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# --------- Embeddings / Vectorstore ----------
def create_vectorstore(docs):
    # Requires: pip install langchain-huggingface sentence-transformers
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # In-memory Chroma (avoid Windows file locks)
    return Chroma.from_documents(docs, embedding=embeddings)

# --------- QA chain with custom prompt ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant. Use ONLY the context to answer the question.\n"
        "If the answer is not in the context, say: 'I could not find the answer in the document.'\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ),
)

def create_qa_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    llm = ChatGroq(model="llama-3.1-8b-instant")  # uses env var
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )

# --------- API route ----------
@app.post("/ask/")
async def ask_question(file: UploadFile = File(...), question: str = ""):
    try:
        if not question or not question.strip():
            return JSONResponse({"error": "Question is empty"}, status_code=400)

        os.makedirs("temp_files", exist_ok=True)
        file_location = f"temp_files/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Load → Split → Embed → Vectorstore
        documents = load_document(file_location)
        docs = split_document(documents)
        vectordb = create_vectorstore(docs)
        qa_chain = create_qa_chain(vectordb)

        # Run retrieval QA
        result = qa_chain.invoke({"query": question})
        answer = result["result"]
        source_docs = result.get("source_documents", [])
        sources_preview = [d.page_content[:300] for d in source_docs]

        # Debug: how many chunks did we retrieve?
        print(f"Retrieved {len(source_docs)} source docs")

        # Fallback: if nothing was retrieved or LLM gave generic fallback, answer from whole doc
        generic_markers = [
            "no context provided",
            "you haven't provided any context",
            "i could not find the answer",
        ]
        if len(source_docs) == 0 or any(m in answer.lower() for m in generic_markers):
            print("Fallback to full-document answer")
            # Stuff first N chunks directly
            context = "\n\n".join(d.page_content for d in docs[:8])  # keep prompt size manageable
            llm = ChatGroq(model="llama-3.1-8b-instant")
            fallback_prompt = QA_PROMPT.format(context=context, question=question)
            # minimalist call via LC's LLM .invoke
            fallback = llm.invoke(fallback_prompt)
            answer = fallback.content if hasattr(fallback, "content") else str(fallback)
            sources_preview = [d.page_content[:300] for d in docs[:3]]

        return {"answer": answer, "sources": sources_preview}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
