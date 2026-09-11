# ⚡ InsightStream RAG

A high-performance **Retrieval-Augmented Generation (RAG)** application designed for accelerated career intelligence and document consultation. By leveraging asynchronous orchestration, **InsightStream RAG** achieves superior throughput compared to traditional sequential RAG pipelines.

## 🚀 Key Engineering Highlights

*   **⚡ Parallel Async Orchestration:** Uses `asyncio.gather` (behind a semaphore that caps fan-out to respect API rate limits) to fire several independent analysis chains at once. These are separate calls rather than one combined prompt because each question needs *different chunks* — "payment terms" and "termination conditions" live in different parts of a contract, so one shared retrieval would serve both badly.
*   **🎯 Adaptive RAG Strategy:** A single **Document Type** control drives both halves of the pipeline: indexing (`Chunk Size`, `Overlap`, `Retrieval Depth (k)`) and analysis (which question set the Strategic Analysis tab runs). A contract wants contract-shaped chunking *and* contract-shaped questions — they aren't independent choices.
*   **⚖️ Retrieval Where It Earns Its Place:** Long documents (Legal/Book, Technical/Code) fan out into several independent questions, each with its own retrieval, run concurrently. Short documents don't: the Resume path deliberately skips retrieval and parallelism entirely and sends the complete document in one structured call, because chunking a two-page CV only discards information — and "which skills are missing?" is unanswerable from partial text.
*   **🔍 OCR Fallback:** Automatically detects scanned/image-only PDFs and PDFs with a corrupted text layer (e.g. a broken embedded-font encoding — a real failure mode found in testing with Hebrew-generated PDFs), and retries extraction via Tesseract OCR instead of failing.
*   **🛡️ Data Isolation & Privacy:** Each browser session gets its own ChromaDB collection, keyed to a session ID rather than the filename, so two users never share indexed data. All indexed data is in-memory only (nothing is written to disk), and the sidebar's **"🔥 Incinerate Document Data"** button deletes the active collection and resets the session on demand.
*   **📊 Live Timing Telemetry:** The sidebar reports wall-clock time for your own chat queries and Strategic Analysis runs as you use the app. Note this is live telemetry, not a controlled sequential-vs-parallel benchmark — see "Performance Benchmarks" below.

## 🛠️ Tech Stack

*   **Orchestration:** [LangChain](https://langchain.com) (Async API)
*   **LLM:** OpenAI GPT-4o-mini
*   **Vector DB:** [ChromaDB](https://trychroma.com)
*   **Frontend:** [Streamlit](https://streamlit.io)

## ⚙️ Performance Benchmarks

The Strategic Analysis tab has a **"Also run sequentially (benchmark)"** toggle.
Enabling it re-runs the *identical* question set a second time, one call at a
time, and reports both timings plus the speedup — so the comparison changes
exactly one variable (execution strategy) while holding the workload constant.

This replaces an earlier comparison that measured a single chat query against a
two-chain analysis run. That conflated workload size with execution strategy and
so measured neither; any latency figures from this repo's history predating the
toggle should be treated as unreliable.

## 💻 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd insightstream-rag
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   The OCR fallback also needs the Tesseract binary installed locally (it
   isn't pip-installable). On macOS:
   ```bash
   brew install tesseract tesseract-lang   # tesseract-lang adds non-English language data, incl. Hebrew
   ```
   Without Tesseract installed, standard PDF extraction still works — only
   the OCR fallback for scanned/corrupted PDFs is unavailable, and the app
   will show a clear error explaining that instead of crashing.

3. **Configure Environment:**
   Create a `.env` file:
   ```text
   OPENAI_API_KEY=your_api_key_here

   # Optional — LangSmith tracing, off by default. Flip to true to debug/demo;
   # tracing sends document chunks, prompts, and answers to LangSmith's servers.
   LANGSMITH_TRACING=false
   LANGSMITH_API_KEY=your_langsmith_key_here
   LANGSMITH_ENDPOINT=https://api.smith.langchain.com
   LANGSMITH_PROJECT=pdf-rag-explorer
   ```

4. **Launch:**
   ```bash
   streamlit run app.py
   ```

---
Built by **Roy Insler** | [LinkedIn](www.linkedin.com/in/roy-insler-3a8042120)
