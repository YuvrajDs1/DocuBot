from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import tempfile
import uuid
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

TEMP_DIR = Path(tempfile.gettempdir()) / "docubot_files"
TEMP_DIR.mkdir(exist_ok=True)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstores = {}  # in-memory store {file_id: Chroma}


def load_document(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type (use .pdf or .docx)")
    return loader.load()


def split_document(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)


def create_vectorstore(docs, file_id: str):
    vectordb = Chroma.from_documents(docs, embedding=embeddings)
    vectorstores[file_id] = vectordb
    return vectordb


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
    llm = ChatGroq(model="llama-3.1-8b-instant")
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )



@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Upload and index a document, return file_id for later queries"""
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{file_id}_{file.filename}"
    file_location = TEMP_DIR / unique_filename

    try:
        with open(file_location, "wb") as f:
            content = await file.read()
            f.write(content)

        # Process once
        documents = load_document(str(file_location))
        docs = split_document(documents)
        create_vectorstore(docs, file_id)

        return {"file_id": file_id, "file_name": file.filename}

    except Exception as e:
        return JSONResponse({"error": f"Failed to upload file: {str(e)}"}, status_code=500)


@app.post("/ask/")
async def ask_question(file_id: str = Form(...), question: str = Form(...)):
    """Ask a question about an already uploaded document"""
    try:
        if file_id not in vectorstores:
            raise HTTPException(status_code=404, detail="File not found or not uploaded")

        vectordb = vectorstores[file_id]
        qa_chain = create_qa_chain(vectordb)

        result = qa_chain.invoke({"query": question})
        answer = result["result"]
        source_docs = result.get("source_documents", [])
        sources_preview = [d.page_content[:300] for d in source_docs]

        return {"answer": answer, "sources": sources_preview}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.delete("/cleanup/")
async def cleanup_all_files():
    """Clear all temporary files and memory vectorstores"""
    try:
        import shutil
        vectorstores.clear()
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
            TEMP_DIR.mkdir(exist_ok=True)
        return {"message": "All temporary files and vectorstores cleaned up successfully"}
    except Exception as e:
        return JSONResponse({"error": f"Cleanup failed: {str(e)}"}, status_code=500)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "temp_dir": str(TEMP_DIR), "active_files": len(vectorstores)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
