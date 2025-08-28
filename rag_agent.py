import os
import json
import pickle
import faiss
import requests
import time
from typing import List, Dict, Tuple, Optional

from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import docx2txt

# Paths for persistence
DOC_INDEX_DIR = "vector_index_docs"
DOC_META_PATH = os.path.join(DOC_INDEX_DIR, "meta.pkl")
DOC_FAISS_PATH = os.path.join(DOC_INDEX_DIR, "index.faiss")
MEM_INDEX_DIR = "vector_index_memory"
MEM_FAISS_PATH = os.path.join(MEM_INDEX_DIR, "index.faiss")
MEM_META_PATH = os.path.join(MEM_INDEX_DIR, "meta.pkl")

# Embedding model (local, free)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
EMBED_DIM = _embedder.get_sentence_embedding_dimension()

# Short-term memory (per-process; Streamlit session keeps chat in state)
SHORT_TERM_MEMORY_LIMIT = int(os.getenv("SHORT_TERM_MEMORY_LIMIT", "6"))


# -------------------------------
# Utilities for FAISS + metadata
# -------------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_meta(meta: dict, path: str):
    with open(path, "wb") as f:
        pickle.dump(meta, f)

def load_meta(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}

def create_faiss_index(dim: int):
    idx = faiss.IndexFlatL2(dim)
    return idx

def save_faiss_index(index: faiss.IndexFlatL2, path: str):
    ensure_dir(os.path.dirname(path))
    faiss.write_index(index, path)

def load_faiss_index(path: str) -> Optional[faiss.IndexFlatL2]:
    if os.path.exists(path):
        return faiss.read_index(path)
    return None

# -------------------------------
# Document ingestion
# -------------------------------
def _extract_text_from_pdf(fp: str) -> str:
    text_parts = []
    reader = PdfReader(fp)
    for p in reader.pages:
        text_parts.append(p.extract_text() or "")
    return "\n".join(text_parts)

def _extract_text_from_docx(fp: str) -> str:
    return docx2txt.process(fp)

def _extract_text_from_txt(fp: str) -> str:
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _load_file_text(fp: str) -> str:
    lower = fp.lower()
    if lower.endswith(".pdf"):
        return _extract_text_from_pdf(fp)
    if lower.endswith(".docx"):
        return _extract_text_from_docx(fp)
    if lower.endswith(".txt"):
        return _extract_text_from_txt(fp)
    raise ValueError("Unsupported file type: " + fp)

def chunk_text(text: str, chunk_size=1000, overlap=200) -> List[str]:
    chunks = []
    i = 0
    L = len(text)
    while i < L:
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def init_doc_index():
    ensure_dir(DOC_INDEX_DIR)
    idx = load_faiss_index(DOC_FAISS_PATH)
    meta = load_meta(DOC_META_PATH)
    if idx is None:
        idx = create_faiss_index(EMBED_DIM)
        save_faiss_index(idx, DOC_FAISS_PATH)
        save_meta({"docs": []}, DOC_META_PATH)
    return idx, meta

def add_documents_to_index(filepaths: List[str]) -> Tuple[int,int]:
    """
    Ingest multiple files. Returns (total_chunks_added, total_files_processed)
    """
    idx, meta = init_doc_index()
    starting_count = len(meta.get("docs", []))
    total_chunks = 0
    for fp in filepaths:
        text = _load_file_text(fp)
        chunks = chunk_text(text)
        embeddings = _embedder.encode(chunks, show_progress_bar=False)
        # Append embeddings to faiss
        current_n = idx.ntotal
        idx.add(embeddings.astype('float32'))
        # store metadata entries
        for i, chunk in enumerate(chunks):
            meta["docs"].append({
                "source_file": os.path.basename(fp),
                "content": chunk,
                "added_at": time.time()
            })
        total_chunks += len(chunks)
    save_faiss_index(idx, DOC_FAISS_PATH)
    save_meta(meta, DOC_META_PATH)
    return total_chunks, len(filepaths)

def list_indexed_files() -> List[str]:
    meta = load_meta(DOC_META_PATH)
    files = sorted({d.get("source_file","unknown") for d in meta.get("docs", [])})
    return files

def clear_doc_index():
    if os.path.exists(DOC_INDEX_DIR):
        for f in os.listdir(DOC_INDEX_DIR):
            os.remove(os.path.join(DOC_INDEX_DIR, f))

# -------------------------------
# Long-term memory index (summaries)
# -------------------------------
def init_mem_index():
    ensure_dir(MEM_INDEX_DIR)
    idx = load_faiss_index(MEM_FAISS_PATH)
    meta = load_meta(MEM_META_PATH)
    if idx is None:
        idx = create_faiss_index(EMBED_DIM)
        save_faiss_index(idx, MEM_FAISS_PATH)
        save_meta({"mems": []}, MEM_META_PATH)
    return idx, meta

def add_memory_summary(summary_text: str) -> None:
    idx, meta = init_mem_index()
    emb = _embedder.encode([summary_text]).astype('float32')
    idx.add(emb)
    meta["mems"].append({
        "summary": summary_text,
        "ts": time.time()
    })
    save_faiss_index(idx, MEM_FAISS_PATH)
    save_meta(meta, MEM_META_PATH)

def retrieve_memory(query: str, k: int = 3) -> List[str]:
    idx = load_faiss_index(MEM_FAISS_PATH)
    meta = load_meta(MEM_META_PATH)
    if idx is None or len(meta.get("mems", [])) == 0:
        return []
    q_emb = _embedder.encode([query]).astype('float32')
    D, I = idx.search(q_emb, k)
    results = []
    for i in I[0]:
        if i < len(meta["mems"]):
            results.append(meta["mems"][i]["summary"])
    return results

def clear_memory_index():
    if os.path.exists(MEM_INDEX_DIR):
        for f in os.listdir(MEM_INDEX_DIR):
            os.remove(os.path.join(MEM_INDEX_DIR, f))

# -------------------------------
# Retrieval from docs
# -------------------------------
def retrieve_docs(query: str, k: int = 4) -> List[Dict]:
    idx = load_faiss_index(DOC_FAISS_PATH)
    meta = load_meta(DOC_META_PATH)
    if idx is None or len(meta.get("docs", [])) == 0:
        return []
    q_emb = _embedder.encode([query]).astype('float32')
    D, I = idx.search(q_emb, k)
    results = []
    for i in I[0]:
        if i < len(meta["docs"]):
            results.append(meta["docs"][i])
    return results

# -------------------------------
# LLM calls: GROQ (primary) / OPENAI (fallback)
# Improved: support GROQ payload type env and better response parsing
# -------------------------------
def call_groq(prompt: str, max_tokens: int = 512) -> str:
    """
    Generic call to Groq inference endpoint.
    Environment variables:
      GROQ_API_URL - full inference URL
      GROQ_API_KEY - api key
      GROQ_PAYLOAD_TYPE - one of: 'prompt' (default), 'input', 'messages'
    This function tries to be flexible and accept common Groq shapes.
    """
    GROQ_API_URL = os.getenv("GROQ_API_URL")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    PAYLOAD_TYPE = os.getenv("GROQ_PAYLOAD_TYPE", "prompt")  # prompt | input | messages
    if not GROQ_API_URL or not GROQ_API_KEY:
        return "[GROQ not configured. Set GROQ_API_URL and GROQ_API_KEY to use Groq.]"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    # Build payload depending on expected shape
    if PAYLOAD_TYPE == "messages":
        payload = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    elif PAYLOAD_TYPE == "input":
        payload = {"input": prompt, "max_tokens": max_tokens}
    else:
        payload = {"prompt": prompt, "max_tokens": max_tokens}

    try:
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # Try several common response shapes
        if isinstance(data, dict):
            # direct text
            for key in ("text", "generated_text", "output", "completion"):
                if key in data and isinstance(data[key], str):
                    return data[key]
            # 'choices' like OpenAI
            if "choices" in data and len(data["choices"]) > 0:
                c0 = data["choices"][0]
                if isinstance(c0, dict):
                    return c0.get("text") or c0.get("message", {}).get("content", "") or str(c0)
            # 'outputs' list
            if "outputs" in data and isinstance(data["outputs"], list) and len(data["outputs"])>0:
                out = data["outputs"][0]
                if isinstance(out, dict) and "text" in out:
                    return out["text"]
                return str(out)
        # fallback: return truncated json
        return json.dumps(data)[:2000]
    except Exception as e:
        return f"[Groq request failed: {e}]"

# Optional OpenAI fallback
def call_openai(prompt: str, max_tokens: int = 512) -> str:
    import openai
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        return "[OpenAI not configured.]"
    openai.api_key = OPENAI_API_KEY
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
    )
    return resp.choices[0].message.content

def call_llm(prompt: str, max_tokens: int = 512) -> str:
    """
    Use Groq if configured, else fallback to OpenAI if configured, else return stub.
    """
    if os.getenv("GROQ_API_URL") and os.getenv("GROQ_API_KEY"):
        return call_groq(prompt, max_tokens=max_tokens)
    if os.getenv("OPENAI_API_KEY"):
        return call_openai(prompt, max_tokens=max_tokens)
    return "[LLM not configured (set GROQ_API_URL+GROQ_API_KEY or OPENAI_API_KEY).]"

# -------------------------------
# Prompt assembly + high-level QA
# -------------------------------
BASE_PROMPT_TEMPLATE = """
You are a helpful assistant that answers user questions using the provided document snippets and remembered facts.
Be concise. Cite source_file names when you use a snippet.

Long-term memory (summaries / facts):
{mem_facts}

Document snippets (top {k}):
{doc_snippets}

User question:
{question}

Answer:
"""

def answer_with_memory_and_docs(question: str, k_docs: int = 4) -> Dict:
    docs = retrieve_docs(question, k=k_docs)
    doc_snips = []
    for d in docs:
        snip = d.get("content","").strip()
        src = d.get("source_file","unknown")
        doc_snips.append(f"[{src}] {snip[:800]}")
    mems = retrieve_memory(question, k=3)
    prompt = BASE_PROMPT_TEMPLATE.format(
        mem_facts = "\n".join(mems) if mems else "(none)",
        doc_snippets = "\n\n".join(doc_snips) if doc_snips else "(none)",
        question = question,
        k = k_docs
    )
    answer = call_llm(prompt, max_tokens=512)
    # store summary as long-term memory (short summary)
    add_memory_summary = os.getenv("AUTO_SAVE_MEM", "true").lower() in ("1","true","yes")
    if add_memory_summary:
        # create a short summary of Q/A and store
        summary_prompt = f"Summarize the following Q and A into a single short fact (<=80 words):\nQ: {question}\nA: {answer}"
        summary = call_llm(summary_prompt, max_tokens=150)
        add_memory_summary(summary)
    return {"answer": answer, "sources": list({d.get("source_file","unknown") for d in docs})}

# -------------------------------
# Small tool functions (MCP placeholders)
# -------------------------------
def tool_current_time(_) -> str:
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def tool_combine_docs(query: str, k: int = 5) -> str:
    docs = retrieve_docs(query, k=k)
    if not docs:
        return "No documents indexed."
    combined = "\n\n---\n\n".join([f"[{d['source_file']}]\n{d['content'][:2000]}" for d in docs])
    prompt = f"Please produce a concise combined summary of the following text. Use bullet points and highlight overlaps/conflicts/gaps:\n\n{combined}"
    return call_llm(prompt, max_tokens=600)
