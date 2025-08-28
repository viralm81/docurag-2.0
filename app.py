import streamlit as st
import pandas as pd
import docx2txt
import fitz  # PyMuPDF for PDFs
from rag_agent import (
    add_documents_to_index,
    list_indexed_files,
    clear_doc_index,
    clear_memory_index,
    answer_with_memory_and_docs,
    tool_current_time,
    tool_combine_docs,
)

st.set_page_config(page_title="DocuRAG", layout="wide")

st.title("📄 DocuRAG – Document Q&A with Memory")

# Sidebar
st.sidebar.header("Controls")

if st.sidebar.button("🧹 Clear Document Index"):
    msg = clear_doc_index()
    st.sidebar.success(msg)

if st.sidebar.button("🧹 Clear Memory Index"):
    msg = clear_memory_index()
    st.sidebar.success(msg)

if st.sidebar.button("🕒 Current Time"):
    st.sidebar.info(tool_current_time())

# File uploader
uploaded_files = st.file_uploader(
    "Upload documents", 
    type=["txt", "pdf", "docx", "csv"], 
    accept_multiple_files=True
)

def extract_text_from_file(file):
    """Extract text from uploaded file based on type"""
    if file.type == "text/plain":
        return file.read().decode("utf-8")

    elif file.type == "application/pdf":
        text = ""
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        for page in pdf:
            text += page.get_text()
        return text

    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(file)

    elif file.type == "text/csv":
        df = pd.read_csv(file)
        return df.to_string()

    return None

if uploaded_files:
    docs = []
    for file in uploaded_files:
        extracted_text = extract_text_from_file(file)
        if extracted_text:
            docs.append(extracted_text)

    if docs:
        msg = add_documents_to_index(docs)
        st.success(msg)

# Show indexed docs
if st.checkbox("Show Indexed Files"):
    ids = list_indexed_files()
    if ids:
        st.write("Indexed document IDs:", ids)
    else:
        st.info("No documents indexed yet.")

# Query box
st.subheader("Ask a Question")
query = st.text_input("Type your query:")

if query:
    answer = answer_with_memory_and_docs(query)
    st.write("### Answer")
    st.write(answer)
