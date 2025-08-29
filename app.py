import streamlit as st
from rag import add_pdf_to_index, clear_doc_index, list_indexed_files, answer_with_docs
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

st.set_page_config(page_title="📄 DocuRAG HF Hosted LLM", layout="wide")
st.title("📄 DocuRAG - PDF Q&A (Render Free Plan)")

# Upload PDFs
uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        add_pdf_to_index(file_path)
        st.success(f"Indexed: {file_path}")

# Ask Questions
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

# Admin Sidebar
st.sidebar.subheader("Admin Tools")
if st.sidebar.button("Clear Index"):
    clear_doc_index()
    st.sidebar.success("Document index cleared!")

if st.sidebar.button("List Indexed Files"):
    count = list_indexed_files()
    st.sidebar.info(f"Total chunks stored: {count}")
