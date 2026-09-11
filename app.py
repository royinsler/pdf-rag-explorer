import streamlit as st
import os
import tempfile
import asyncio
import re
import time
import uuid
import hashlib
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from typing import List

# Load API keys from environment
load_dotenv()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="InsightStream RAG", 
    page_icon="⚡", 
    layout="wide"
)

# --- API KEY VALIDATION ---
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ OPENAI_API_KEY is not set. Add it to your .env file before using InsightStream.")
    st.stop()

# --- INITIALIZE SESSION STATE ---
# Ensures all persistent variables are ready for the UI logic
if "chat_history" not in st.session_state:
    # Each turn: {"user": str, "bot": str, "sources": list[Document]}
    st.session_state.chat_history = []
if "analysis_report" not in st.session_state:
    st.session_state.analysis_report = None
if "last_chat_duration" not in st.session_state:
    st.session_state.last_chat_duration = None
if "analysis_time" not in st.session_state:
    st.session_state.analysis_time = None
if "session_id" not in st.session_state:
    # Unique per browser session. Used to scope Chroma collection names so
    # two users (or two tabs) never share a collection just because they
    # happened to upload a file with the same name.
    st.session_state.session_id = uuid.uuid4().hex[:8]
if "doc_key" not in st.session_state:
    st.session_state.doc_key = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "doc_text" not in st.session_state:
    # Full extracted text, kept for analyses that deliberately bypass
    # retrieval (see run_resume_analysis).
    st.session_state.doc_text = ""
if "uploader_key" not in st.session_state:
    # Bumped on incineration to force st.file_uploader to reset — Streamlit
    # keeps a widget's selected file tied to its key across reruns, so
    # clearing our own session_state alone doesn't clear the uploader UI.
    st.session_state.uploader_key = 0


def incinerate_document():
    """Delete the active Chroma collection and fully reset the session.
    Shared by both the sidebar and main-screen incinerate buttons so the
    reset logic lives in exactly one place."""
    if st.session_state.vectorstore is not None:
        try:
            st.session_state.vectorstore.delete_collection()
        except Exception:
            pass
    st.session_state.vectorstore = None
    st.session_state.chain = None
    st.session_state.doc_key = None
    st.session_state.doc_text = ""
    st.session_state.chat_history = []
    st.session_state.analysis_report = None
    st.session_state.last_chat_duration = None
    st.session_state.analysis_time = None
    st.session_state.uploader_key += 1


def looks_garbled(text: str) -> bool:
    """Heuristic detector for a corrupted PDF text layer — specifically the
    failure mode we hit in practice: a PDF whose embedded font has no
    working ToUnicode CMap, so every extraction library (we verified this
    against PDFMinerLoader, PyPDFLoader, and PyMuPDFLoader — all three
    produced near-identical garbage) decodes the same wrong byte values
    into plausible-looking but meaningless text instead of erroring.

    This is NOT a general-purpose gibberish detector — it's a narrow,
    honestly-imperfect signal: real prose in any language has a low density
    of stray apostrophes/quote marks, while this specific corruption
    pattern produces a lot of them (e.g. "l'l'", "til'l'l'"). False
    positives/negatives are possible, which is why this only triggers an
    automatic OCR retry rather than a hard failure."""
    if not text or not text.strip():
        return True
    letters = sum(c.isalpha() for c in text)
    if letters < 20:
        return False  # too short to judge reliably either way
    stray_marks = text.count("'") + text.count("’")
    return (stray_marks / letters) > 0.08


def ocr_extract_documents(file_bytes: bytes, lang: str = "heb+eng"):
    """Fallback extraction for PDFs whose text layer is missing or
    corrupted. Rasterizes each page with PyMuPDF and reads it with
    Tesseract OCR, bypassing the PDF's internal font/encoding structures
    entirely — this reads pixels, so a broken ToUnicode CMap can't affect
    it. Requires the `tesseract` binary installed locally (on macOS:
    `brew install tesseract tesseract-lang` for the Hebrew language data)
    plus the `pymupdf` and `pytesseract` Python packages."""
    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image
    from langchain_core.documents import Document

    docs = []
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        for page_num, page in enumerate(pdf):
            # Render at ~300 DPI — PDF default is 72 DPI, and OCR accuracy
            # drops sharply below ~200-300 DPI.
            pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(image, lang=lang)
            if text.strip():
                docs.append(Document(page_content=text, metadata={"page": page_num, "extraction": "ocr"}))
    finally:
        pdf.close()
    return docs


# --- ANALYSIS QUESTION SETS ---
# Each question is fired as its OWN call with its OWN retrieval. They are not
# merged into one prompt because they need different chunks: "payment terms"
# and "termination conditions" live in different parts of a long contract, so
# a single shared retrieval would serve all of them badly. That's what makes
# the concurrency here architectural rather than decorative.
ANALYSIS_QUESTION_SETS = {
    "Legal/Book": [
        ("Parties & Roles", "Who are the parties to this agreement, and what is each party's role and core obligations?"),
        ("Payment Terms", "What are the payment terms — amounts, schedule, currency, and any penalties or interest?"),
        ("Dates & Deadlines", "What are the key dates, deadlines, duration, and renewal terms?"),
        ("Termination", "Under what conditions can this agreement be terminated, and what notice is required?"),
        ("Risks & Unusual Clauses", "Identify any unusual, one-sided, or high-risk clauses a reviewer should pay attention to."),
    ],
    "Technical/Code": [
        ("Architecture", "What is the overall architecture, and how are the main components organized?"),
        ("Dependencies", "What external dependencies, libraries, or services does this rely on?"),
        ("Entry Points", "What are the main entry points, interfaces, or public APIs?"),
        ("Risks & Gaps", "What are the notable risks, limitations, TODOs, or known issues?"),
    ],
    "General": [
        ("Summary", "What is this document about — its purpose and main subject?"),
        ("Key Facts", "What are the most important facts, figures, and data points?"),
        ("Entities", "Who are the key people, organizations, and entities mentioned?"),
        ("Dates", "What dates, deadlines, or time-sensitive items appear?"),
    ],
}

# Caps fan-out so a larger question set (or the eval runner, which will reuse
# this pattern) can't blow through the API's rate limits.
MAX_CONCURRENT_CALLS = 5


class ResumeAnalysis(BaseModel):
    """Structured output schema for the resume path. Using a schema rather
    than free text keeps the score a real field instead of prose to be parsed
    — though note it's still a model estimate, not a calibrated metric."""
    match_score: int = Field(description="0-100 estimate of how well the resume matches the job description")
    rationale: str = Field(description="Two or three sentences explaining the score")
    action_items: List[str] = Field(description="Three concrete, specific changes that would improve the match")
    missing_keywords: List[str] = Field(description="Up to five skills or technologies present in the job description but absent from the resume")


async def run_parallel_analysis(questions):
    """Fire N independent questions concurrently, each through its own
    retrieval, and return their answers in order."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)

    async def ask(question):
        async with semaphore:
            result = await st.session_state.chain.ainvoke({"input": question, "chat_history": []})
            return result["answer"]

    return await asyncio.gather(*(ask(q) for _, q in questions))


async def run_sequential_analysis(questions):
    """The same workload, one call at a time. Exists so the parallel-vs-
    sequential comparison measures an identical workload with exactly one
    variable changed — the previous 'benchmark' compared a single chat query
    against two analysis queries, which conflated workload size with
    execution strategy and therefore measured nothing."""
    answers = []
    for _, question in questions:
        result = await st.session_state.chain.ainvoke({"input": question, "chat_history": []})
        answers.append(result["answer"])
    return answers


def run_resume_analysis(jd: str, resume_text: str) -> ResumeAnalysis:
    """Deliberately uses NEITHER retrieval NOR parallelism.

    A resume fits comfortably in a context window, so chunking it and
    retrieving top-k actively discards information — and "which skills are
    missing from this resume?" is unanswerable from partial text, since the
    model can't distinguish "absent from the document" from "absent from the
    three chunks I happened to retrieve." One call, whole document, structured
    output. Knowing where NOT to apply the technique is the point."""
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(ResumeAnalysis)
    return structured_llm.invoke(
        "You are an expert technical recruiter. Compare the candidate's complete "
        "resume against the job description and return a structured assessment.\n\n"
        f"=== JOB DESCRIPTION ===\n{jd}\n\n"
        f"=== RESUME (complete text) ===\n{resume_text}"
    )


# --- 1. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.title("⚡ InsightStream RAG")
    st.markdown("*High-Performance Career Intelligence*")
    st.markdown("---")
    
    # Live Performance Dashboard
    st.subheader("📊 Performance Metrics")
    if st.session_state.last_chat_duration:
        st.write(f"⏱️ **Baseline (Single):** {st.session_state.last_chat_duration:.2f}s")
    if st.session_state.analysis_time:
        st.write(f"🚀 **Async (Parallel):** {st.session_state.analysis_time:.2f}s")
    
    st.divider()
    st.subheader("🛠️ System Specs")
    st.write("- **Engine:** Async RAG Pipeline")
    st.write("- **Orchestration:** asyncio.gather")
    st.write("- **Storage:** ChromaDB (Isolated)")

    st.divider()
    st.subheader("🔒 Data & Privacy")
    if st.session_state.vectorstore is not None:
        st.caption("🔒 Document indexed", help="Permanently deletes this session's indexed document from ChromaDB and clears the chat history and analysis report. Cannot be undone — you'll need to re-upload the PDF to continue.")
        if st.button("🔥 Incinerate Document Data", key="sidebar_incinerate"):
            incinerate_document()
            st.rerun()
    else:
        st.caption("No document is currently indexed.")

    st.markdown("---")
    st.subheader("👤 Developer")
    st.markdown("[LinkedIn Profile](https://linkedin.com)")
    st.markdown("[GitHub Repository](https://github.com)")
    st.info("Built with LangChain & OpenAI GPT-4o-mini")

# --- 2. MAIN INTERFACE ---
st.title("⚡ InsightStream RAG")
st.markdown("Accelerated document intelligence with adaptive strategies and parallel LLM orchestration.")

col_file, col_opt = st.columns(2)

with col_file:
    uploaded_file = st.file_uploader("Upload Document (PDF)", type="pdf", key=f"pdf_uploader_{st.session_state.uploader_key}")

with col_opt:
    doc_type = st.selectbox(
        "Document Type:",
        ["General", "Resume", "Legal/Book", "Technical/Code"],
        help=(
            "Drives two things: how the document is chunked and retrieved "
            "(chunk size, overlap, k), and which question set the Strategic "
            "Analysis tab runs. Changing it re-indexes the document and clears "
            "the chat history, so pick it before you start rather than toggling "
            "mid-session."
        ),
    )

    # Adaptive chunking/retrieval logic
    if doc_type == "Resume":
        c_size, c_overlap, k_val = 1500, 100, 3
    elif doc_type == "Legal/Book":
        c_size, c_overlap, k_val = 800, 200, 7
    elif doc_type == "Technical/Code":
        c_size, c_overlap, k_val = 600, 50, 10
    else:
        c_size, c_overlap, k_val = 1000, 100, 5

    # Surface BOTH effects of this control, so its dual role is visible
    # rather than hidden behind the label.
    if doc_type == "Resume":
        analysis_summary = "1 structured call, no retrieval"
    else:
        _question_count = len(ANALYSIS_QUESTION_SETS.get(doc_type, ANALYSIS_QUESTION_SETS["General"]))
        analysis_summary = f"{_question_count} parallel questions"
    st.caption(f"Chunk {c_size} · overlap {c_overlap} · k {k_val} · analysis: {analysis_summary}")
    force_ocr = st.checkbox(
        "🔍 Force OCR",
        value=False,
        help=(
            "Skip standard text extraction and read the PDF via OCR instead. "
            "Use this if you already know the PDF is scanned or has a broken "
            "text layer. Slower than standard extraction — otherwise OCR "
            "kicks in automatically only when standard extraction fails or "
            "looks unreliable."
        ),
    )

if st.session_state.vectorstore is not None:
    with st.container(border=True):
        col_status, col_btn = st.columns([4, 1], vertical_alignment="center")
        with col_status:
            st.caption("🔒 Document indexed", help="Permanently deletes this session's indexed document from ChromaDB and clears the chat history and analysis report. Cannot be undone — you'll need to re-upload the PDF to continue.")
        with col_btn:
            if st.button("🔥 Incinerate", key="main_incinerate", use_container_width=True):
                incinerate_document()
                st.rerun()

# --- 3. CORE RAG ENGINE ---
if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    # Fingerprint the exact bytes + strategy. Re-indexing only happens when
    # this changes, instead of on every Streamlit rerun (every chat message,
    # every button click) — which previously re-embedded the whole document
    # through the OpenAI API each time and appended duplicate chunks into
    # the same Chroma collection on every rerun.
    doc_key = hashlib.sha256(file_bytes + doc_type.encode("utf-8")).hexdigest()[:16]

    if st.session_state.doc_key != doc_key:
        with st.spinner("Initializing InsightStream Engine..."):
            # Drop the previous collection so re-uploads (or switching the
            # strategy dropdown) don't leave stale duplicate chunks resident
            # in memory alongside the new ones.
            if st.session_state.vectorstore is not None:
                try:
                    st.session_state.vectorstore.delete_collection()
                except Exception:
                    pass

            # Serialize to a temp file for the loader, then always clean up —
            # previously this left every uploaded PDF sitting in /tmp forever.
            docs = []
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file_bytes)
                    tmp_path = tmp_file.name

                if not force_ocr:
                    loader = PDFMinerLoader(tmp_path)
                    docs = loader.load()
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

            combined_text = "".join(d.page_content for d in docs)
            used_ocr = False
            ocr_error = None

            # Fall back to OCR when the user asked for it explicitly, when
            # standard extraction found nothing (scanned/image-only PDF),
            # or when it looks garbled (a broken font encoding — see
            # looks_garbled's docstring; we hit and verified this exact
            # failure mode across three different extraction libraries).
            if force_ocr or not combined_text.strip() or looks_garbled(combined_text):
                with st.spinner("Standard extraction looked unreliable — retrying with OCR (slower)..."):
                    try:
                        ocr_docs = ocr_extract_documents(file_bytes, lang="heb+eng")
                        ocr_text = "".join(d.page_content for d in ocr_docs)
                        if ocr_docs and not looks_garbled(ocr_text):
                            docs = ocr_docs
                            used_ocr = True
                    except Exception as e:
                        ocr_error = str(e)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_overlap)
            splits = text_splitter.split_documents(docs)

            # Defensive guardrail: if we still have nothing usable after
            # trying OCR too, fail clearly instead of embedding garbage
            # (previously this crashed Chroma.from_documents with a raw
            # ValueError on empty input; worse, a garbled-but-non-empty
            # extraction would previously sail through silently and the LLM
            # would just answer from nonsense context).
            if not splits or looks_garbled(combined_text if not used_ocr else "".join(d.page_content for d in docs)):
                if ocr_error:
                    st.error(
                        f"❌ No readable text could be extracted from this PDF, and the OCR "
                        f"fallback failed to run ({ocr_error}). If this is expected to need OCR, "
                        f"make sure Tesseract is installed locally (macOS: `brew install tesseract "
                        f"tesseract-lang`) and the `pymupdf`/`pytesseract` packages are installed."
                    )
                else:
                    st.error(
                        "❌ No readable text could be extracted from this PDF, even after an OCR "
                        "retry. It may be corrupted, password-protected, or too low-resolution to OCR."
                    )
                st.session_state.doc_key = None
                st.session_state.vectorstore = None
                st.session_state.chain = None
                st.stop()

            if used_ocr:
                st.info("ℹ️ This document needed OCR — standard text extraction failed or looked unreliable.")

            # Collection name is scoped to THIS session + THIS exact document —
            # never the raw filename — so two users uploading "resume.pdf"
            # never land in the same Chroma collection.
            collection_name = re.sub(
                r'[^a-zA-Z0-9_-]', '_',
                f"sess_{st.session_state.session_id}_{doc_key}"
            )[:60]

            st.session_state.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
                collection_name=collection_name
            )

            # Chain Initialization (LCEL) — ConversationalRetrievalChain is a
            # legacy chain (moved to langchain-classic in LangChain v1); this
            # is the composition that replaced it.
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={'k': k_val})

            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "Given a chat history and the latest user question which might "
                    "reference context in the chat history, formulate a standalone "
                    "question which can be understood without the chat history. "
                    "Do NOT answer the question, just reformulate it if needed and "
                    "otherwise return it as is."
                )),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are InsightStream, an expert document consultant. Use the "
                    "following retrieved context to answer the user's question. If "
                    "the answer isn't in the context, say you don't know — do not "
                    "make anything up.\n\n{context}"
                )),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

            st.session_state.chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            st.session_state.doc_key = doc_key
            st.session_state.doc_text = "\n\n".join(d.page_content for d in docs)
            # A new document invalidates the old conversation and report.
            st.session_state.chat_history = []
            st.session_state.analysis_report = None

    # --- 4. FEATURE TABS ---
    st.divider()
    tab_chat, tab_analysis = st.tabs(["💬 Interactive Consultant", "🎯 Strategic Analysis"])

    # TAB 1: INTERACTIVE CHAT
    with tab_chat:
        if st.button("💬 New Conversation", help="Clears the chat transcript. The indexed document stays loaded — use Incinerate above to remove that."):
            st.session_state.chat_history = []
            st.session_state.last_chat_duration = None
            st.rerun()

        for turn in st.session_state.chat_history:
            with st.chat_message("user"): st.markdown(turn["user"])
            with st.chat_message("assistant"):
                st.markdown(turn["bot"])
                if turn["sources"]:
                    with st.expander("🔍 View Context Evidence"):
                        for doc in turn["sources"]:
                            st.markdown(f"> {doc.page_content[:400]}...")

        if prompt := st.chat_input("Consult the document..."):
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                start = time.time()
                history_messages = []
                for t in st.session_state.chat_history:
                    history_messages.append(HumanMessage(content=t["user"]))
                    history_messages.append(AIMessage(content=t["bot"]))
                res = st.session_state.chain.invoke({"input": prompt, "chat_history": history_messages})
                st.session_state.last_chat_duration = time.time() - start
                
                answer = res["answer"]
                sources = res.get("context", [])
                st.markdown(answer)
                if sources:
                    with st.expander("🔍 View Context Evidence"):
                        for doc in sources:
                            st.markdown(f"> {doc.page_content[:400]}...")
                st.session_state.chat_history.append({"user": prompt, "bot": answer, "sources": sources})
                st.rerun()

    # TAB 2: STRATEGIC ANALYSIS — branches on the selected strategy
    with tab_analysis:
        if doc_type == "Resume":
            st.subheader("🎯 Resume vs. Job Description")
            st.caption(
                "Single call with the complete resume in context — no retrieval, no parallelism. "
                "A resume fits in a context window, so chunking it would only lose information.",
                help=(
                    "Retrieval and concurrency are used for long documents (try the Legal/Book or "
                    "Technical/Code strategies), where each question needs different chunks. "
                    "Applying them to a two-page resume would degrade results, not improve them: "
                    "'which skills are missing' is unanswerable from partial text."
                ),
            )
            job_description = st.text_area("Target Job Description:", height=200)

            if st.button("Run Analysis", use_container_width=True):
                if job_description:
                    start_time = time.time()
                    with st.spinner("Analyzing full resume against the job description..."):
                        try:
                            result = run_resume_analysis(job_description, st.session_state.doc_text)
                            st.session_state.analysis_time = time.time() - start_time
                            st.session_state.analysis_report = (
                                f"### Match Score: {result.match_score}%\n\n"
                                f"{result.rationale}\n\n"
                                "### ✅ Action Items\n"
                                + "\n".join(f"- {item}" for item in result.action_items)
                                + "\n\n### 🔑 Missing Keywords\n"
                                + "\n".join(f"- {kw}" for kw in result.missing_keywords)
                                + "\n\n*Note: the match score is a model estimate, not a calibrated metric.*"
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
                else:
                    st.warning("Please input a Job Description to proceed.")

        else:
            questions = ANALYSIS_QUESTION_SETS.get(doc_type, ANALYSIS_QUESTION_SETS["General"])
            st.subheader(f"⚡ Parallel {doc_type} Analysis")
            st.caption(
                f"Fires {len(questions)} independent questions concurrently — each with its own retrieval.",
                help=(
                    "These are separate calls rather than one combined prompt because each question "
                    "needs different chunks of the document. A single shared retrieval would serve "
                    "all of them badly. Fan-out is capped by a semaphore to respect rate limits."
                ),
            )
            with st.expander("Questions in this set"):
                for label, question in questions:
                    st.markdown(f"**{label}** — {question}")

            benchmark = st.checkbox(
                "📊 Also run sequentially (benchmark)",
                value=False,
                help=(
                    "Re-runs the identical question set a second time, one call at a time, so the "
                    "parallel-vs-sequential comparison changes exactly one variable. Doubles the "
                    "API cost of the run."
                ),
            )

            if st.button("Run Analysis", use_container_width=True):
                try:
                    with st.spinner(f"Executing {len(questions)} chains concurrently..."):
                        start_time = time.time()
                        answers = asyncio.run(run_parallel_analysis(questions))
                        parallel_time = time.time() - start_time
                        st.session_state.analysis_time = parallel_time

                    sequential_time = None
                    if benchmark:
                        with st.spinner(f"Benchmark: re-running the same {len(questions)} chains sequentially..."):
                            seq_start = time.time()
                            asyncio.run(run_sequential_analysis(questions))
                            sequential_time = time.time() - seq_start

                    report = "\n\n".join(
                        f"### {label}\n{answer}"
                        for (label, _), answer in zip(questions, answers)
                    )
                    if sequential_time:
                        speedup = sequential_time / parallel_time if parallel_time else 0
                        report = (
                            f"> **Benchmark ({len(questions)} identical questions):** "
                            f"parallel {parallel_time:.2f}s vs sequential {sequential_time:.2f}s "
                            f"— {speedup:.2f}x speedup.\n\n" + report
                        )
                    st.session_state.analysis_report = report
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

        if st.session_state.analysis_report:
            st.success("✅ Analysis Complete")
            st.markdown(st.session_state.analysis_report)
            if st.button("Clear Report"):
                st.session_state.analysis_report = None
                st.rerun()

else:
    st.info("👋 Welcome to InsightStream RAG. Please upload a PDF to initialize the engine.")
