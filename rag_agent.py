import os
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_chromadb import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# =====================
# Setup
# =====================
load_dotenv()

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Persistent storage for Chroma
CHROMA_DIR = "./chroma_store"
os.makedirs(CHROMA_DIR, exist_ok=True)

vector_store = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model.encode
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# =====================
# Document Handling
# =====================

def add_pdf_to_index(uploaded_file):
    """Load PDF, split, embed, and store in Chroma"""
    pdf_path = f"./temp_{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    docs = text_splitter.split_documents(pages)

    # Convert to langchain Document objects
    lc_docs = [Document(page_content=d.page_content, metadata={"source": uploaded_file.name}) for d in docs]

    vector_store.add_documents(lc_docs)
    vector_store.persist()

    os.remove(pdf_path)


def list_indexed_files():
    """Return list of all sources (filenames) in Chroma"""
    results = vector_store.get()
    if "metadatas" in results:
        return list({meta["source"] for meta in results["metadatas"] if "source" in meta})
    return []


def clear_doc_index():
    """Clear ChromaDB index"""
    global vector_store
    if os.path.exists(CHROMA_DIR):
        for f in os.listdir(CHROMA_DIR):
            os.remove(os.path.join(CHROMA_DIR, f))
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model.encode
    )


def clear_memory_index():
    """Alias to clear doc index"""
    clear_doc_index()


# =====================
# Querying
# =====================

def search_docs(query, top_k=3):
    """Search top_k documents in Chroma"""
    results = vector_store.similarity_search(query, k=top_k)
    return results


def answer_with_memory_and_docs(query):
    """Simple retrieval-based answer"""
    docs = search_docs(query, top_k=3)
    if not docs:
        return "No relevant documents found."

    response = f"Query: {query}\n\nRelevant Info:\n"
    for i, doc in enumerate(docs, 1):
        response += f"{i}. {doc.page_content}\n"
    return response


# =====================
# Tools
# =====================

def tool_current_time():
    """Return current system time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def tool_combine_docs(docs):
    """Combine docs into a single string"""
    return "\n---\n".join(docs)
