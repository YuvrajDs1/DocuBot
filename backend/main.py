# FastAPI setup
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Union, List

app = FastAPI()

@app.post('/upload/')
async def upload_file(file: UploadFile = File(...)):
    file_location = f'temp_files/{file.filename}'
    with open(file_location, 'wb') as f:
        f.write(await file.read())
    return {'file_name': file.filename, 'file_location': file_location}

# Langchain core
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

def load_document(file_path: str):
    if file_path.endswith('.pdf'):
        loader = UnstructuredPDFLoader(file_path)

    elif file_path.endswith('.docx'):
        loader = UnstructuredWordDocumentLoader(file_path)

    else:
        raise ValueError('Unsupported file type')
    document = loader.load()
    return document

def split_document(document):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(document)
    return docs

def create_vectorstore(docs):
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    vector_store = Chroma.from_documents(docs,embedding=embeddings)
    return vector_store

def create_qa_chain(vector_store):
    llm = OllamaLLM(model='gemma:2b')
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        chain_type='refine'
    )
    return qa

@app.post('/ask/')
async def ask_question(file: UploadFile = File(...), question: str=""):
    file_location = f'temp_files/{file.filename}'
    with open(file_location, 'wb') as f:
        f.write(await file.read())
    document = load_document(file_location)
    split = split_document(document)
    vectordb = create_vectorstore(split)
    qa_chain = create_qa_chain(vectordb)

    answer = qa_chain.run(question)
    return JSONResponse({'answer': answer})