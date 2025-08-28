# DocuRAG 2.0 — Groq-backed RAG with persistent FAISS

This project is a self-contained Document-RAG app using:
- FAISS for local persistent vectors (document index + long-term memory)
- Sentence-transformers embeddings (all-MiniLM)
- Groq (hosted inference) for the LLM (or OpenAI fallback)
- Streamlit UI for chat + upload
- Simple MCP adapter placeholder + DuckDuckGo tool

## Quick start (local)

1. Install Python 3.11 and Docker if required.
2. Create a virtualenv and install:
   ```
   pip install -r requirements.txt
   ```
3. Set environment variables:
   - `GROQ_API_URL` : Groq model endpoint (example: `https://api.groq.com/v1/models/<model>/generate`)
   - `GROQ_API_KEY` : Your Groq API key
   - (optional) `OPENAI_API_KEY` : fallback if Groq not provided
4. Run app:
   ```
   streamlit run app.py
   ```

## Deploy on Render (recommended)

1. Push repo to GitHub.
2. Create a Web Service on Render:
   - Connect your GitHub repo.
   - Use Docker (recommended): Render will use the `Dockerfile`.
   - Attach a **Persistent Disk** and mount it at the project root (so `vector_index_docs/` persists). Configure Render to use that mount (see Render docs).
3. Add Environment Variables on Render dashboard:
   - `GROQ_API_URL`, `GROQ_API_KEY`
   - Optional: `OPENAI_API_KEY`
   - Optional: `MCP_ENABLED=true` if you supply MCP client later
4. Deploy. Monitor logs and ensure FAISS index files appear under the persistent disk.

## Notes
- **Groq**: set your correct model endpoint / payload expectations. If Groq expects `input` instead of `prompt` you may need to modify `GROQ_PAYLOAD_TYPE` or the payload in `call_groq()`.
- **Persistence**: Render requires attaching a Persistent Disk for data to survive redeploys — otherwise, your FAISS indices are ephemeral.
- **Free LLM tiers**: Groq, TogetherAI, etc. often have free tiers. Confirm quotas with provider.
