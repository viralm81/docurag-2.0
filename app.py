import os
import tempfile
import streamlit as st
from rag_agent import add_documents_to_index, list_indexed_files, clear_doc_index, clear_memory_index, answer_with_memory_and_docs, tool_current_time, tool_combine_docs
from tools import MCPAdapter, ddg_web_search
import json

st.set_page_config(page_title="DocuRAG 2.0 (Groq)", layout="wide")
st.title("📄 DocuRAG 2.0 — Groq-backed, FAISS persistent")

# Sidebar controls
st.sidebar.header("Index Controls")
if st.sidebar.button("Clear Document Index"):
    clear_doc_index()
    st.sidebar.success("Document index cleared.")

if st.sidebar.button("Clear Long-term Memory"):
    clear_memory_index()
    st.sidebar.success("Long-term memory cleared.")

files_listed = list_indexed_files()
st.sidebar.subheader("Indexed files")
if files_listed:
    for f in files_listed:
        st.sidebar.write("- " + f)
else:
    st.sidebar.info("No files indexed yet")

# Upload area
st.subheader("Upload documents (PDF / DOCX / TXT)")
uploaded = st.file_uploader("Upload", accept_multiple_files=True, type=['pdf','docx','txt'])
if uploaded:
    tmpdir = tempfile.mkdtemp()
    paths = []
    for f in uploaded:
        p = os.path.join(tmpdir, f.name)
        with open(p, "wb") as out:
            out.write(f.read())
        paths.append(p)
    chunks, files_count = add_documents_to_index(paths)
    st.success(f"Indexed {files_count} file(s) — {chunks} chunks added.")

st.divider()

# Chat / QA
st.subheader("Chat with your documents")
if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Ask a question about your documents (you can also ask combine/compare)...")
if query:
    st.session_state.history.append({"role":"user","text":query})
    with st.spinner("Thinking..."):
        out = answer_with_memory_and_docs(query)
    st.session_state.history.append({"role":"assistant","text":out["answer"]})

for m in st.session_state.history:
    if m["role"] == "user":
        st.chat_message("user").write(m["text"])
    else:
        st.chat_message("assistant").write(m["text"])

st.divider()

# Utilities: combine docs
st.subheader("Utilities")
if st.button("Combine top docs (quick)"):
    with st.spinner("Combining..."):
        res = tool_combine_docs("summarize important content", k=5)
    st.markdown("**Combined Summary**")
    st.write(res)

with st.expander("Tools (DuckDuckGo / MCP)"):
    q2 = st.text_input("DuckDuckGo search query")
    if st.button("Search web"):
        res = ddg_web_search(q2)
        st.write(res)
    st.markdown("---")
    st.write("MCP Adapter (placeholder)")
    mcp = MCPAdapter(enabled=(os.getenv("MCP_ENABLED","").lower() in ("1","true","yes")))
    tool = st.text_input("Tool name")
    args = st.text_area("JSON args (optional)", value="{}")
    if st.button("Call MCP"):
        try:
            parsed = json.loads(args)
        except Exception as e:
            parsed = {"raw": args}
        st.write(mcp.call_tool(tool, **parsed))
