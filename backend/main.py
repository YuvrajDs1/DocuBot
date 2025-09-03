from fastapi import FastAPI, UploadFile, File, Form
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

# Use system temp directory instead of relative path
TEMP_DIR = Path(tempfile.gettempdir()) / "docubot_files"
TEMP_DIR.mkdir(exist_ok=True)

def cleanup_file(file_path: str):
    """Safely delete a temporary file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up: {file_path}")
    except Exception as e:
        print(f"Error cleaning up {file_path}: {e}")

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Generate unique filename to avoid conflicts
    unique_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{unique_id}_{file.filename}"
    
    file_location = TEMP_DIR / unique_filename
    
    try:
        with open(file_location, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return {
            "file_name": file.filename,
            "file_location": str(file_location),
            "unique_id": unique_id
        }
    except Exception as e:
        return JSONResponse({"error": f"Failed to upload file: {str(e)}"}, status_code=500)

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

def create_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(docs, embedding=embeddings)

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
    llm = ChatGroq(model="llama-3.1-8b-instant") 
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )

@app.post("/ask/")
async def ask_question(file: UploadFile = File(...), question: str = Form()):
    temp_file_path = None
    
    try:
        # Validate question
        if not question or not question.strip():
            raise HTTPException(status_code=400, detail="Question is required and cannot be empty")

        # Create unique temporary file path
        unique_id = str(uuid.uuid4())
        filename = file.filename or "unknown_file"
        file_extension = os.path.splitext(filename)[1]
        unique_filename = f"{unique_id}_{filename}"
        temp_file_path = TEMP_DIR / unique_filename

        # Save uploaded file
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        print(f"Processing file: {temp_file_path}")
        
        # Load and process document
        documents = load_document(str(temp_file_path))
        docs = split_document(documents)
        vectordb = create_vectorstore(docs)
        qa_chain = create_qa_chain(vectordb)

        # Get answer from QA chain
        result = qa_chain.invoke({"query": question})
        answer = result["result"]
        source_docs = result.get("source_documents", [])
        sources_preview = [d.page_content[:300] for d in source_docs]

        print(f"Retrieved {len(source_docs)} source docs")

        # Fallback logic if no good answer found
        generic_markers = [
            "no context provided",
            "you haven't provided any context",
            "i could not find the answer",
        ]
        
        if len(source_docs) == 0 or any(m in answer.lower() for m in generic_markers):
            print("Fallback to full-document answer")
            # Use first N chunks directly as context
            context = "\n\n".join(d.page_content for d in docs[:8])  
            llm = ChatGroq(model="llama-3.1-8b-instant")
            fallback_prompt = QA_PROMPT.format(context=context, question=question)

            fallback = llm.invoke(fallback_prompt)
            answer = fallback.content if hasattr(fallback, "content") else str(fallback)
            sources_preview = [d.page_content[:300] for d in docs[:3]]

        return {"answer": answer, "sources": sources_preview}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
    
    finally:
        # Always cleanup temp file
        if temp_file_path and os.path.exists(temp_file_path):
            cleanup_file(str(temp_file_path))

@app.delete("/cleanup/")
async def cleanup_all_files():
    """Clean up all temporary files"""
    try:
        import shutil
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
            TEMP_DIR.mkdir(exist_ok=True)
            return {"message": "All temporary files cleaned up successfully"}
        else:
            return {"message": "No temporary files to clean up"}
    except Exception as e:
        return JSONResponse({"error": f"Cleanup failed: {str(e)}"}, status_code=500)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "temp_dir": str(TEMP_DIR)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)