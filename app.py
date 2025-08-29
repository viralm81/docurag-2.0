import os
import streamlit as st
import textwrap
from langchain.llms import HuggingFaceHub
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ------------------- Streamlit page config -------------------
st.set_page_config(page_title="📄 DocuRAG HF Hosted LLM", layout="wide")
st.title("📄 DocuRAG - PDF Q&A (Render Free Plan)")

# ------------------- HuggingFace Hosted LLM -------------------
hf_token = os.getenv("HF_API_KEY")
llm = None
if not hf_token:
    st.error("❌ HF_API_KEY is not set! Cannot run LLM.")
else:
    try:
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-small",
            model_kwargs={"temperature": 0.1},
            huggingfacehub_api_token=hf_token,
            task="text2text-generation"
        )
        st.info("✅ HuggingFace LLM is ready.")
    except Exception as e:
        st.error(f"❌ Failed to initialize HuggingFace LLM: {e}")

# ------------------- In-memory storage for PDF chunks -------------------
pdf_chunks_store = []

# ------------------- Functions -------------------
def add_pdf_to_index(pdf_path: str):
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        pdf_chunks_store.extend(chunks)
    except Exception as e:
        st.error(f"Error adding PDF '{pdf_path}': {e}")

def clear_doc_index():
    global pdf_chunks_store
    pdf_chunks_store = []

def list_indexed_files():
    return len(pdf_chunks_store)

def answer_with_docs(query: str, k: int = 3):
    if not pdf_chunks_store:
        return "No documents uploaded.", ""
    if not llm:
        return "LLM not initialized. Check HF_API_KEY.", ""

    relevant_docs = pdf_chunks_store[:k]
    try:
        prompt = "Answer the question based on the following documents:\n\n"
        for d in relevant_docs:
            prompt += f"{d.page_content}\n\n"
        prompt += f"Question: {query}\nAnswer:"

        answer = llm(prompt)
        sources = "\n\n".join([
            f"Source: {d.metadata}\n{textwrap.shorten(d.page_content, width=200)}"
            for d in relevant_docs
        ])
        return answer, sources
    except Exception as e:
        return f"Error generating answer: {e}", ""

# ------------------- PDF Upload -------------------
uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        add_pdf_to_index(file_path)
        st.success(f"Indexed: {file_path}")

# ------------------- Ask Questions -------------------
st.subheader("Ask a question about your PDFs")
query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if query.strip():
        answer, sources = answer_with_docs(query)
        st.write("### ✅ Answer")
        st.write(answer)
        st.write("### 📚 Sources")
        st.write(sources)
    else:
        st.warning("Please enter a question.")

# ------------------- Admin Sidebar -------------------
st.sidebar.subheader("Admin Tools")
if st.sidebar.button("Clear Index"):
    clear_doc_index()
    st.sidebar.success("Document index cleared!")

if st.sidebar.button("List Indexed Files"):
    count = list_indexed_files()
    st.sidebar.info(f"Total chunks stored: {count}")
