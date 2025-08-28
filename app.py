import streamlit as st
from rag import add_pdf_to_index, clear_doc_index, list_indexed_files, answer_with_docs

st.set_page_config(page_title="DocuRAG with Chroma", layout="wide")

st.title("📄 DocuRAG - PDF Q&A with ChromaDB")

# Upload PDFs
uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        add_pdf_to_index(uploaded_file.name)
        st.success(f"Indexed: {uploaded_file.name}")

st.subheader("Ask a question about your PDFs")
query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if query.strip():
        answer = answer_with_docs(query)
        st.write("### Answer")
        st.write(answer)
    else:
        st.warning("Please enter a question.")

st.sidebar.subheader("Admin Tools")
if st.sidebar.button("Clear Index"):
    clear_doc_index()
    st.sidebar.success("Document index cleared!")

if st.sidebar.button("List Indexed Files"):
    count = list_indexed_files()
    st.sidebar.info(f"Total chunks stored: {count}")
