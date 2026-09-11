# ⚡ InsightStream RAG
**High-Performance Document Intelligence & Parallel Orchestration**

InsightStream RAG is a **Retrieval-Augmented Generation (RAG)** engine built around three
problems that show up in real document work: extraction that fails silently, answers you
can't trace back to the source, and latency on multi-part analysis.

## 🚀 The Core Advantage: RAG vs. Vanilla LLM
Modern LLMs have large context windows but remain prone to hallucination and offer no
traceability. InsightStream grounds every answer in retrieved passages and shows them
alongside the response, so each claim can be checked against the source text. Grounding
reduces hallucination — it doesn't eliminate it, which is why the evidence is always one
click away rather than hidden.

## 🛠️ Technical Challenges & Engineering Solutions

### 1. High-Throughput Parallel Orchestration
Analysing a long document means asking several unrelated questions of it, which is slow
one at a time.
*   **The Fix:** `asyncio.gather` fires the question set concurrently, behind an
    `asyncio.Semaphore` that caps fan-out to respect API rate limits.
*   **Why separate calls:** each question needs *different chunks* — "payment terms" and
    "termination conditions" live in different parts of a contract — so one shared
    retrieval would serve all of them badly. The concurrency is architectural, not
    decorative.

### 2. Extraction That Fails Loudly, Not Silently
Text extraction can fail in two ways: returning nothing (scanned/image-only PDFs), or —
far worse — returning confident nonsense.
*   **The Finding:** a Hebrew contract whose embedded font carries no usable `ToUnicode`
    CMap decodes to plausible-looking ASCII mojibake. PDFMiner, pypdf and PyMuPDF all
    produce the *same* garbage, so no choice of loader fixes it. The file renders
    perfectly on screen, because rendering draws glyphs and never consults that table.
*   **The Fix:** a heuristic detector flags corrupted text layers, and extraction falls
    back to OCR — PyMuPDF rasterises each page at 300 DPI and Tesseract reads the pixels,
    bypassing the PDF's font structures entirely. If both paths fail, the app says so
    rather than embedding garbage and answering from it.

### 3. Retrieval Only Where It Earns Its Place
RAG is the wrong tool for a short document. Chunking a two-page CV and retrieving top-k
discards information, and "which skills are missing?" is unanswerable from partial text —
the model can't distinguish *absent from the document* from *absent from the chunks I
retrieved*.
*   **The Fix:** long documents fan out into parallel, independently-retrieved questions.
    The résumé path deliberately uses **neither** retrieval nor parallelism: one call, the
    complete document in context, a structured (Pydantic) result.

### 4. Adaptive Indexing Strategy
Data density varies by document type — a dense résumé versus a sprawling legal contract.
*   **The Fix:** a single **Document Type** control drives both halves of the pipeline:
    indexing (**Chunk Size**, **Overlap**, **Retrieval Depth k**) and analysis (which
    question set runs). A contract wants contract-shaped chunking *and* contract-shaped
    questions; they aren't independent choices.

### 5. Stateless & Privacy-First Architecture
*   **Zero-Storage Policy:** uploads exist only in ephemeral memory for the session. The
    temp file the loader needs is deleted in a `finally` block, so nothing is left on disk.
*   **Isolated Vector Collections:** each browser session gets a ChromaDB collection keyed
    to a session UUID — never the filename — so two users uploading `resume.pdf` never
    share indexed data.
*   **On-Demand Teardown:** the **🔥 Incinerate** control deletes the active collection and
    resets the session, so the data is gone when you say so rather than when the process
    restarts.

## 📊 Performance Benchmarks
The Strategic Analysis tab has an **"Also run sequentially (benchmark)"** toggle. It
re-runs the *identical* question set a second time, one call at a time, and reports both
timings plus the speedup — changing exactly one variable (execution strategy) while
holding the workload constant.

Earlier figures in this repo's history compared a single chat query against a two-chain
analysis run. That conflated workload size with execution strategy and therefore measured
neither; treat any latency numbers predating this toggle as unreliable.

## 🗺️ Roadmap
*   **Evaluation harness:** a golden Q/A set with retrieval-quality and groundedness
    evaluators, run as LangSmith experiments, so chunking changes can be judged on
    measurements rather than intuition.
*   **Node-wise tracing:** LangSmith tracing is wired up and **off by default** — enabling
    it sends document chunks, prompts and answers to LangSmith's servers, which is a
    deliberate choice rather than a default for anyone handling real contracts.
*   **Hebrew document intelligence:** deeper support for RTL legal documents, the area
    where the extraction work above has the most leverage.

## 💻 Installation & Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Tesseract** (required for the OCR fallback; not pip-installable). On macOS:
   ```bash
   brew install tesseract tesseract-lang   # tesseract-lang adds Hebrew and other languages
   ```
   Without it, normal PDFs work fine — only the OCR fallback is unavailable, and the app
   reports that clearly instead of crashing.

3. **Configure environment** — create a `.env` file:
   ```text
   OPENAI_API_KEY=your_api_key_here

   # Optional — LangSmith tracing, off by default. Enabling it sends document
   # content to LangSmith's servers.
   LANGSMITH_TRACING=false
   LANGSMITH_API_KEY=your_langsmith_key_here
   LANGSMITH_ENDPOINT=https://api.smith.langchain.com
   LANGSMITH_PROJECT=pdf-rag-explorer
   ```

4. **Launch:**
   ```bash
   streamlit run app.py
   ```

## 🧰 Utilities
`diagnose_pdf.py` compares PDF extraction backends against a single file — useful when
extracted text looks wrong and you need to tell "bad loader" apart from "broken PDF":
```bash
python diagnose_pdf.py /path/to/file.pdf
```

## 🛠️ Tech Stack
*   **Orchestration:** [LangChain](https://langchain.com) (LCEL, async API)
*   **LLM:** OpenAI GPT-4o-mini
*   **Vector DB:** [ChromaDB](https://trychroma.com)
*   **Extraction:** PDFMiner, with PyMuPDF + Tesseract OCR fallback
*   **Frontend:** [Streamlit](https://streamlit.io)

---
Built by **Roy Insler** | [LinkedIn](https://www.linkedin.com/in/roy-insler-3a8042120)
