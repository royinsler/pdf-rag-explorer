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

### 3. Knowing What OCR Destroyed
Extraction is the only stage where information *enters* the system. Everything downstream
can preserve or flag it; nothing downstream can recover it. So the pipeline measures the
damage rather than hoping there is none.

*   **Per-word confidence.** OCR reads through `image_to_data`, not `image_to_string`, so
    Tesseract's own per-word confidence survives. Tokens below 60% are named to the model
    as untrustworthy and listed in the UI. On the reference contract: 87.7% mean, 68
    tokens flagged.
*   **Why that is not enough, structurally.** On that same document a misread that
    destroyed a contractual date scored **82** — the engine was confidently wrong — and
    the date itself scored *nothing*, because it produced no token at all. Confidence is a
    property of an emitted token; a deletion emits nothing, so no threshold can reach it.
    Lowering the cutoff flags half the document and still misses it.
*   **So damage is detected structurally too** (`damaged_values.py`): a connector that
    promises a date (`החל`, `מיום`, `עד ליום`) with no date after it **and** evidence of
    damage in the gap — Latin characters in Hebrew prose, a low-confidence token, a bare
    structureless number. Two signals are required; one alone flags correct Hebrew
    constantly, since `מיום שנדרש` ("from the day required") is an ordinary phrase.
*   **What this does not catch, measured.** On a second contract the detector found
    nothing — and that was a miss, not a pass. OCR had deleted a payment date there too,
    leaving the same debris, but `ביום`, `מיום`, `ליום` and `בתאריך` appear *zero* times
    in that document's extracted text: the connectors were destroyed along with the
    values. The rule assumes the grammar survives even when the value does not, and the
    anchor words are short function words — exactly what OCR misreads most. Widening the
    word list does not help when no word survived, which is why the roadmap's fix is at
    extraction rather than in this heuristic.
*   **Bidi controls are stripped** from OCR output. `image_to_string` wraps Latin runs
    inside Hebrew lines in LRM/RLM marks — presentation hints that this app decides at
    render time anyway, and an invisible mark landing inside a date silently breaks the
    substring matching the verification layer depends on.

### 4. Verifying the Answer, Not Trusting It
Prompting proved insufficient three times over on one clause. Told nothing, the model
fabricated a date. Told values may be illegible, it passed OCR garbage through. Told
exactly which passages were damaged, it invented a fluent, plausible substitute with
nothing wrong-looking in it at all.

Each constraint made the failure *quieter*, not rarer. So three checks run after the
model, none of which need it to cooperate:

*   **`verify_quotes`** — does each cited passage actually appear in the document?
*   **`verify_answer_facts`** — does every date and amount in the answer appear in the
    source? (Dates are compared as `(day, month, year)`, so `1/7/2026` and `1.07.2026`
    match — a check that flags correct values is worse than no check.)
*   **`find_ocr_garbage_echoes`** and **`answer_rests_on_damage`** — has the answer copied
    OCR gibberish, and which flagged passages does it reproduce?

That last one answers **provenance**, and it is the only thing that catches the third
failure. Presence is not provenance: `verify_answer_facts` confirms a value appears
*somewhere*, and in this contract the date `15.5.2026` is printed for a payment — so a
date guessed for a different clause would have been confirmed against real text from an
unrelated section.

### 5. Retrieval Only Where It Earns Its Place
RAG is the wrong tool for a short document. Chunking a two-page CV and retrieving top-k
discards information, and "which skills are missing?" is unanswerable from partial text —
the model can't distinguish *absent from the document* from *absent from the chunks I
retrieved*.
*   **The Fix:** long documents fan out into parallel, independently-retrieved questions.
    The résumé path deliberately uses **neither** retrieval nor parallelism: one call, the
    complete document in context, a structured (Pydantic) result.

### 6. Adaptive Indexing Strategy
Data density varies by document type — a dense résumé versus a sprawling legal contract.
*   **The Fix:** a single **Document Type** control drives both halves of the pipeline:
    indexing (**Chunk Size**, **Overlap**, **Retrieval Depth k**) and analysis (which
    question set runs). A contract wants contract-shaped chunking *and* contract-shaped
    questions; they aren't independent choices.

### 7. Stateless & Privacy-First Architecture
*   **Zero-Storage Policy:** uploads exist only in ephemeral memory for the session. The
    temp file the loader needs is deleted in a `finally` block, so nothing is left on disk.
*   **Isolated Vector Collections:** each browser session gets a ChromaDB collection keyed
    to a session UUID — never the filename — so two users uploading `resume.pdf` never
    share indexed data.
*   **On-Demand Teardown:** the **🔥 Incinerate** control deletes the active collection and
    resets the session, so the data is gone when you say so rather than when the process
    restarts.

## 📊 Performance Benchmarks
`benchmark_parallelism.py` sweeps fan-out from 2 to 10, running the *identical* question
set both ways so exactly one variable changes. Seven reps per level, medians:

| Fan-out | Sequential | Parallel | Speedup | Efficiency |
|---|---|---|---|---|
| 2 | 3.31s | 1.90s | 1.74× | 87% |
| 5 | 9.49s | 2.97s | 3.20× | 64% |
| 10 | 16.17s | 3.97s | 4.07× | 41% |

**Efficiency decays for two compounding reasons, and neither is Amdahl's law** — there is
no serial fraction here, the calls are fully independent. Below the concurrency limit,
wall-clock equals the *slowest* call, so efficiency is `mean ÷ max` of the batch and the
expected maximum grows with N. Above the limit, throughput binds as well: N calls
pipeline through 5 slots.

An earlier version of this benchmark reported a 3.02× speedup on a two-call batch — an
efficiency of 151%, which is definitionally impossible, since you cannot beat running
every call at once. The cause was arm ordering: sequential always ran first and warmed the
connection for parallel. The runner now alternates arm order, warms up per level, and
flags any row exceeding its own ceiling as `INVALID` rather than printing it as a result.

## 🔬 Evaluation
`eval_rag.py` runs 17 golden questions against a synthetic contract across five indexing
configurations, scoring retrieval hit, answer correctness (both deterministic against
known ground truth) and groundedness (LLM-as-judge) *separately*:

| Config | chunk | k | Retrieval | Correct | Grounded |
|---|---|---|---|---|---|
| control-tiny | 200 | 2 | 82% | 76% | 100% |
| Technical/Code | 600 | 10 | 100% | 100% | 100% |
| Legal/Book | 800 | 7 | 100% | 100% | 100% |
| General | 1000 | 5 | 88% | 88% | 100% |
| Résumé | 1500 | 3 | 82% | 82% | 100% |

**`k` predicts the score; total context size does not.** Ordered by `k` the results are
monotonic (2→76%, 3→82%, 5→88%, 7→100%, 10→100%), yet the control saw 400 characters of
context and scored *below* the Résumé preset's 4,500. Each answer lives in one place, so
what matters is whether that place lands in the top-k — more shots at the target, not
bigger shots. On a contract corpus, raise `k` before you raise chunk size.

**Groundedness stayed at 100% even on wrong answers**, which is the metric working and
showing its own limit: when retrieval returns the confidentiality clause and the model
faithfully answers from it, that answer *is* grounded — just not in the right clause.
Faithfulness is not correctness, which is why they are scored apart.

*Caveat:* 17 questions on one synthetic document is a smoke test, not a benchmark.

## ✅ Tests
Everything except `smoke_resume.py` is deterministic and needs no API key:

```bash
python test_verification.py        # 25 cases — quotes, dates, garbage echoes, provenance
python test_damaged_values.py      # 10 cases — damage detection vs. correct Hebrew prose
python test_damage_regression.py   # the three failure modes of §4, each still caught
python smoke_resume.py             # one API call: exercises the résumé path
```

`test_damage_regression.py` is the interesting one. It replays the three wrong answers the
model produced for a single clause and asserts each is still caught — the fabricated date
by the fact check, the passed-through garbage by the echo check, and the fluent invented
substitute by provenance alone. Remove a layer and it fails.

## 🗺️ Roadmap
*   **Two-engine OCR consensus.** The principled fix for the blind spot in §3: OCR a page
    with a second engine and diff. Unlike confidence, a diff detects *absence* — one
    engine emits a date where the other emits nothing.
*   **Cross-encoder reranking** over a wider candidate set, the standard fix for the
    distractor failures the eval exposes. The eval now discriminates, so there is a way to
    tell whether it helped.
*   **Hedged requests** to attack the tail-latency decay measured above — the right answer
    to stragglers, at the cost of duplicate spend.
*   **Node-wise tracing:** LangSmith is wired up and **off by default** — enabling it
    sends document chunks, prompts and answers to a third party, which is a deliberate
    choice rather than a default for anyone handling real contracts.

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

## 🧰 Diagnostics
Each of these exists because a question needed evidence rather than an argument:

```bash
python diagnose_pdf.py file.pdf              # compare extraction backends on one file
python ocr_confidence_report.py file.pdf TOK # what Tesseract scored, incl. tokens it was sure of
python ocr_text_diff.py file.pdf             # image_to_string vs. rebuilt text, bidi-normalised
python damage_report.py file.pdf             # passages whose promised value is missing or corrupted
python quote_probe.py "<quote>" file.pdf     # why a quote failed verification, to the character
python chunk_report.py file.pdf              # what indexing actually produced
python stress_inputs.py                      # 9 adversarial PDFs against the extraction path
```

## 🛠️ Tech Stack
*   **Orchestration:** [LangChain](https://langchain.com) (LCEL, async API)
*   **LLM:** OpenAI GPT-4o-mini
*   **Vector DB:** [ChromaDB](https://trychroma.com)
*   **Extraction:** PDFMiner, with PyMuPDF + Tesseract OCR fallback
*   **Frontend:** [Streamlit](https://streamlit.io)

---
Built by **Roy Insler** | [LinkedIn](https://www.linkedin.com/in/roy-insler-3a8042120)
