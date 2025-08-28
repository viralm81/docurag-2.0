import streamlit as st
from rag_agent import (
    add_pdf_to_index,
    list_indexed_files,
    clear_doc_index,
    clear_memory_index,
    answer_with_memory_and_docs,
    tool_current_time,
    tool_combine_docs,
)

st.set_page_config(page_title="DocuRAG", layout="wide")

st.title("📄 DocuRAG - PDF & Memory QA Agent")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    with st.spinner("Indexing PDF..."):
        add_pdf_to_index(uploaded_file)
    st.success(f"Added {uploaded_file.name} to index.")

# Show indexed files
if st.button("📂 Show Indexed Files"):
    files = list_indexed_files()
    if files:
        st.write("Indexed files:", files)
    else:
        st.write("No files indexed yet.")

# Clear indexes
col1, col2 = st.columns(2)
with col1:
    if st.button("❌ Clear Docs Index"):
        clear_doc_index()
        st.success("Document index cleared.")

with col2:
    if st.button("🧹 Clear Memory"):
        clear_memory_index()
        st.success("Memory cleared.")

# Query
query = st.text_input("🔍 Ask a question:")
if query:
    with st.spinner("Searching and answering..."):
        answer =
