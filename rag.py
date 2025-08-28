import os
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma as ChromaStore
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document

# Path for local persistent storage
CHROMA_DB_DIR = "chroma_db"

# Embeddings
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vectorstore
vectorstore = ChromaStore(
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embedding_model
)

def add_pdf_to_index(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    vectorstore.add_documents(chunks)
    vectorstore.persist()

def clear_doc_index():
    if os.path.exists(CHROMA_DB_DIR):
        import shutil
        shutil.rmtree(CHROMA_DB_DIR)
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)

def list_indexed_files():
    # just show stored collection stats
    return vectorstore._collection.count()

def answer_with_docs(query: str, k: int = 3):
    docs = vectorstore.similarity_search(query, k=k)
    if not docs:
        return "No relevant documents found."
    answer = "\n\n".join([f"Source: {d.metadata}\n{d.page_content}" for d in docs])
    return answer
