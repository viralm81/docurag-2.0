import os
import shutil
import textwrap
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceHub
from langchain.chains import RetrievalQA

# ------------------- Paths & Embeddings -------------------
CHROMA_DB_DIR = "chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize vectorstore (disk-based)
vectorstore = Chroma(
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embedding_model
)

# ------------------- HuggingFace Hosted LLM -------------------
hf_token = os.getenv("HF_API_KEY")
llm = HuggingFaceHub(
    repo_id="google/flan-t5-small",  # hosted model (small)
    model_kwargs={"temperature":0.1},
    huggingfacehub_api_token=hf_token
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ------------------- Functions -------------------
def add_pdf_to_index(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    vectorstore.add_documents(chunks)
    vectorstore.persist()

def clear_doc_index():
    global vectorstore, retriever, qa_chain
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k":3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def list_indexed_files():
    return vectorstore._collection.count()

def answer_with_docs(query: str):
    try:
        docs = retriever.get_relevant_documents(query)
        answer = qa_chain.llm_chain.run({"input_documents": docs, "question": query})
        
        # Format sources
        sources = "\n\n".join(
            [f"Source: {d.metadata}\n{textwrap.shorten(d.page_content, width=200)}" for d in docs]
        )
        return answer, sources
    except Exception as e:
        return f"Error generating answer: {str(e)}", ""
