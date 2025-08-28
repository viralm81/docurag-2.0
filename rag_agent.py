import os
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Try both faiss and faiss_cpu (Render usually uses faiss-cpu wheel)
try:
    import faiss
except ModuleNotFoundError:
    import faiss_cpu as faiss

from sentence_transformers import SentenceTransformer

# =====================
# Setup
# =====================
load_dotenv()

# Sentence Transformer embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS index (cosine similarity)
dimension = embedding_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dimension)

# In-memory storage for documents and vectors
doc_store = []        # List of texts
file_store = []       # List of filenames
id_to_text = {}       # Mapping ID → text


# =====================
# Utility Functions
# =====================

def embed_text(texts):
    """Convert text list to embeddings using SentenceTransformer"""
    embeddings = embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype("float32")


def add_documents_to_index(file_name, docs):
    """
    Add list of documents to FAISS index and store mapping.
    file_name: source file
    docs: list of text chunks
    """
    global doc_store, file_store, id_to_text

    embeddings = embed_text(docs)
    index.add(embeddings)

    start_id = len(doc_store)
    for i, text in enumerate(docs):
        idx = start_id + i
        id_to_text[idx] = text
        doc_store.append(text)
        file_store.append(file_name)


def list_indexed_files():
    """Return list of indexed file names"""
    return list(set(file_store))


def clear_doc_index():
    """Clear document index but keep memory"""
    global doc_store, file_store, id_to_text, index
    dimension = embedding_model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dimension)
    doc_store.clear()
    file_store.clear()
    id_to_text.clear()


def clear_memory_index():
    """Alias for full reset (same as clear_doc_index here)"""
    clear_doc_index()


def search_docs(query, top_k=3):
    """Return top_k most similar docs for a query"""
    if index.ntotal == 0:
        return []

    q_emb = embed_text([query])
    distances, indices = index.search(q_emb, top_k)

    results = []
    for idx in indices[0]:
        if idx in id_to_text:
            results.append(id_to_text[idx])
    return results


def answer_with_memory_and_docs(query):
    """Return simple combined answer from docs + memory"""
    top_docs = search_docs(query, top_k=3)
    if not top_docs:
        return "No relevant documents found."

    response = f"Query: {query}\n\nRelevant Info:\n"
    for i, doc in enumerate(top_docs, 1):
        response += f"{i}. {doc}\n"
    return response


# =====================
# Tools (example)
# =====================

def tool_current_time():
    """Return current system time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def tool_combine_docs(docs):
    """Combine list of docs into single text"""
    return "\n---\n".join(docs)
