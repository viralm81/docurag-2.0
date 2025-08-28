import os
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # updated import

CHROMA_DB_DIR = "chroma_db"

# Updated embedding model (no more deprecation warning)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize vectorstore
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_model)

def add_pdf_to_index(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    vectorstore.add_documents(chunks)
    vectorstore.persist()

def clear_doc_index():
    global vectorstore
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_model)

def list_indexed_files():
    return vectorstore._collection.count()

def answer_with_docs(query: str, k: int = 3):
    docs = vectorstore.similarity_search(query, k=k)
    if not docs:
        return {"answer": "No relevant documents found.", "sources": []}
    return {
        "answer": "Based on retrieved documents:",
        "sources": [
            {
                "file": os.path.basename(d.metadata.get("source", "")),
                "page": d.metadata.get("page", "?"),
                "content": d.page_content
            }
            for d in docs
        ]
    }
