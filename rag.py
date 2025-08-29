import os
import textwrap
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceHub
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# HuggingFace hosted LLM
hf_token = os.getenv("HF_API_KEY")
llm = HuggingFaceHub(
    repo_id="google/flan-t5-small",
    model_kwargs={"temperature": 0.1},
    huggingfacehub_api_token=hf_token
)

# In-memory storage for PDF chunks
pdf_chunks_store = []

# ------------------- Functions -------------------
def add_pdf_to_index(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Store in memory
    pdf_chunks_store.extend(chunks)

def clear_doc_index():
    global pdf_chunks_store
    pdf_chunks_store = []

def list_indexed_files():
    return len(pdf_chunks_store)

def answer_with_docs(query: str, k: int = 3):
    if not pdf_chunks_store:
        return "No documents uploaded.", ""

    # Simple similarity search: for demo, take first k chunks
    relevant_docs = pdf_chunks_store[:k]

    # Run LLM on retrieved docs
    try:
        answer = llm.generate([{
            "input_documents": relevant_docs,
            "question": query
        }]).generations[0][0].text

        sources = "\n\n".join([
            f"Source: {d.metadata}\n{textwrap.shorten(d.page_content, width=200)}"
            for d in relevant_docs
        ])
        return answer, sources
    except Exception as e:
        return f"Error generating answer: {str(e)}", ""
