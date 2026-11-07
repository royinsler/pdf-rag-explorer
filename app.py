import streamlit as st
import html
import os
import asyncio
import re
import time
import uuid
import hashlib
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
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
if "full_context" not in st.session_state:
    # True when the document is small enough to answer from directly, with no
    # retrieval at all. See extraction.fits_in_context.
    st.session_state.full_context = False
if "doc_text" not in st.session_state:
    # Full extracted text, kept for analyses that deliberately bypass
    # retrieval (see run_resume_analysis).
    st.session_state.doc_text = ""
if "ocr_confidence" not in st.session_state:
    # Mean Tesseract word confidence for the loaded document, or None when the
    # document was read without OCR. Kept because it is the only quality signal
    # available at the point where information ENTERS the system — everything
    # downstream can preserve or flag it, but nothing downstream can recover it.
    st.session_state.ocr_confidence = None
if "low_confidence_terms" not in st.session_state:
    # [(token, score)] for tokens OCR itself scored below OCR_MIN_CONFIDENCE.
    st.session_state.low_confidence_terms = []
if "used_ocr" not in st.session_state:
    st.session_state.used_ocr = False
if "damaged_values" not in st.session_state:
    # Passages where the text promises a date or amount that extraction did not
    # deliver intact. Distinct from the list above: confidence scores the tokens
    # that WERE produced, and a deleted value produces no token to score.
    st.session_state.damaged_values = []
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
    st.session_state.full_context = False
    st.session_state.chat_history = []
    st.session_state.analysis_report = None
    st.session_state.last_chat_duration = None
    st.session_state.analysis_time = None
    st.session_state.ocr_confidence = None
    st.session_state.low_confidence_terms = []
    st.session_state.damaged_values = []
    st.session_state.used_ocr = False
    st.session_state.uploader_key += 1


# Question sets live in analysis_sets.py so scripts can import them
# without starting Streamlit.
from analysis_sets import ANALYSIS_QUESTION_SETS
# The résumé path lives in its own module so it can be run without
# launching Streamlit — see resume_analysis.py for why that matters.
from resume_analysis import run_resume_analysis
# Extraction pipeline lives in extraction.py so stress_inputs.py can
# exercise the real logic without starting Streamlit.
from extraction import (extract_document, drop_tiny_chunks, fits_in_context,
                        ocr_caveat, OCR_MIN_CONFIDENCE)
# Verification lives in verification.py so it can be tested directly.
from verification import (verify_quotes, verify_answer_facts,
                          find_ocr_garbage_echoes, answer_rests_on_damage, split_debris)
# Damaged-value detection lives in damaged_values.py — it is pure text analysis
# with no Streamlit or OpenAI dependency, so it can be tested directly.
from damaged_values import detect_damaged_values, damage_caveat

# Caps fan-out so a larger question set (or the eval runner, which will reuse
# this pattern) can't blow through the API's rate limits.
MAX_CONCURRENT_CALLS = 5

# The answering pieces (EXTRACTION_RULE, the caveat helpers, GroundedAnswer and
# answer_from_full_document) live in answering.py so the agent graph can import
# them without starting Streamlit. The retrieval path below still uses the same
# EXTRACTION_RULE and _with_caveat - one copy, imported, not duplicated.
from answering import (EXTRACTION_RULE, _escape_braces, _with_caveat,
                       GroundedAnswer, answer_from_full_document)


def document_caveat() -> str:
    """The per-document extraction caveat for the currently loaded document.

    Empty unless OCR ran and flagged specific tokens. This is the principled
    version of the blanket illegibility rule above: rather than telling the
    model to be careful in general, it names the tokens the OCR engine itself
    was unsure about, so "is this value legible?" stops being a guess.
    """
    # Two signals, deliberately kept separate. OCR confidence names tokens the
    # engine doubted; damage detection names places where a promised value is
    # missing or corrupted — including where the engine was CONFIDENT, which is
    # how this document's renovation date was lost (Tesseract scored the token
    # beside it 82 and emitted nothing at all for the date itself).
    return _escape_braces(
        ocr_caveat(st.session_state.low_confidence_terms)
        + damage_caveat(st.session_state.damaged_values)
    )


async def run_parallel_full_context(questions, doc_text, caveat=""):
    """Same fan-out, no retrieval: each question sees the whole document.

    Still N genuinely independent calls, so the concurrency is real — but the
    "each question needs its own retrieval" justification does not apply here,
    because there is no retrieval. They run in parallel simply because they are
    independent, which is a weaker claim and worth stating as such.
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages(_with_caveat([
        ("system",
         "You are InsightStream, an expert document consultant. The COMPLETE "
         "document is below. Answer only from it; say so plainly if it does not "
         "contain the answer. " + EXTRACTION_RULE +
         "\n\n=== DOCUMENT ===\n{document}"),
    ], caveat))
    chain = prompt | llm

    async def ask(question):
        async with semaphore:
            reply = await chain.ainvoke({"document": doc_text, "input": question})
            return reply.content

    return await asyncio.gather(*(ask(q) for _, q in questions))


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


def _is_rtl(text: str) -> bool:
    """True when the passage is predominantly right-to-left (Hebrew/Arabic).

    Rendered in a left-to-right container, an RTL excerpt puts its numbers,
    dates and punctuation in visibly wrong places. That matters more here than
    it would elsewhere: the evidence panel exists so someone can check a claim
    against the source text, and a citation you can't read is no citation.
    """
    rtl = sum(1 for ch in text if "\u0590" <= ch <= "\u08ff")
    letters = sum(1 for ch in text if ch.isalpha())
    return letters > 0 and (rtl / letters) > 0.3


def document_status():
    """Label and tooltip for the loaded document, reflecting which path ran.

    Worth surfacing rather than hiding: whether an answer came from retrieval
    or from the whole document changes how much you should trust it, and in
    full-context mode there is no vector store for the tooltip to talk about.
    """
    # OCR confidence belongs next to the mode, not buried in a one-time banner:
    # it bounds how much any answer can be trusted for the whole session, and
    # the banner is gone after the first rerun.
    confidence = st.session_state.ocr_confidence
    if confidence is None:
        suffix, extra = "", ""
    else:
        suffix = f" · OCR {confidence:.0f}%"
        suspect = len(st.session_state.low_confidence_terms)
        extra = (
            f" Text came from OCR at {confidence:.0f}% mean word confidence"
            + (f", with {suspect} token(s) below {OCR_MIN_CONFIDENCE:.0f}% flagged to the "
               "model as illegible." if suspect else ", with no token below "
               f"{OCR_MIN_CONFIDENCE:.0f}%.")
        )

    if st.session_state.full_context:
        return (
            "🔒 Loaded in full — no retrieval" + suffix,
            "This document is small enough to hand to the model whole, so nothing "
            "was embedded and no vector store exists: no chunk boundary can split "
            "a clause and no top-k can miss a paragraph. Incinerate clears the "
            "document text, chat history and analysis report." + extra,
        )
    return (
        "🔒 Indexed for retrieval" + suffix,
        "This document is too large to pass whole, so it was chunked and embedded "
        "into a session-private ChromaDB collection. Incinerate permanently deletes "
        "that collection and clears the chat history and analysis report. Cannot be "
        "undone — you'll need to re-upload the PDF." + extra,
    )


def render_extraction_report():
    """Show what extraction did to this document, on every run.

    This lived inside the one-time indexing block, which was wrong in both
    directions: the banner appeared once and was gone after the next widget
    click, while the OCR confidence in the status line rendered BEFORE indexing
    had computed it, so on the run that mattered it showed nothing at all. How
    damaged a document is bounds every answer drawn from it for the whole
    session, so it is state to display, not an event to announce.
    """
    if st.session_state.used_ocr:
        note = "ℹ️ This document needed OCR — standard text extraction failed or looked unreliable."
        if st.session_state.ocr_confidence is not None:
            note += f" Mean OCR confidence: **{st.session_state.ocr_confidence:.0f}%**."
        suspect = st.session_state.low_confidence_terms
        if suspect:
            note += (
                f" {len(suspect)} token(s) scored below {OCR_MIN_CONFIDENCE:.0f}% and are named "
                "to the model as untrustworthy."
            )
        st.info(note)
        if suspect:
            with st.expander(f"Tokens OCR was unsure about ({len(suspect)})"):
                st.caption(
                    "Tesseract's own per-word confidence, worst first. Note what is NOT "
                    "here: the token that destroyed this contract's renovation date scored "
                    "82, and the date itself produced no token at all. Confidence can only "
                    "score what the engine emitted."
                )
                st.table([{"token": token, "confidence": f"{score:.0f}%"}
                          for token, score in suspect[:40]])

    damaged = st.session_state.damaged_values
    if damaged:
        st.warning(
            f"⚠️ {len(damaged)} passage(s) promise a date or amount that extraction did "
            "not capture intact. These are the places where an answer is most likely to "
            "be wrong, and OCR confidence alone does not find them."
        )
        with st.expander(f"Passages with a damaged value ({len(damaged)})"):
            for item in damaged:
                st.caption(" · ".join(item.reasons))
                if _is_rtl(item.snippet):
                    st.markdown(
                        f"<div dir='rtl' style='background:#1a1c24;padding:10px;"
                        f"border-radius:6px;margin-bottom:10px'>{item.snippet}</div>",
                        unsafe_allow_html=True)
                else:
                    st.markdown(f"> {item.snippet}")


def render_damaged_quotes(runs):
    """A gentle note: the answer is clean, but the passage it quotes contains OCR damage."""
    if runs:
        st.caption("ℹ️ The quoted passage contains text extraction could not read ("
                   + ", ".join(runs) + "). The answer itself does not repeat it.")


def render_garbage_echoes(echoes):
    """Warn when an answer has copied OCR gibberish out of the document.

    The prompt already tells the model never to report damaged fragments as a
    value. It does it anyway. This check does not depend on the model complying
    — it reads the answer afterwards — which is the only kind of guarantee
    available when the component you are constraining is a language model.
    """
    if not echoes:
        return
    st.error(
        "⛔ This answer repeats text that extraction produced as garbage: "
        + ", ".join(f"**{run}**" for run in echoes)
        + ". Those characters are not words in the document — they are what OCR "
        "returned where it could not read it. Any value they stand in for is not "
        "legible and the answer should not be relied on for it."
    )


def render_damage_provenance(resting_on):
    """Say when an answer was drawn from a passage known to be damaged.

    Deliberately worded as provenance rather than as a verdict. The answer may
    be perfectly good — it may be saying the value is illegible, which is the
    behaviour we want — but the reader deserves to know which well it was drawn
    from, because that is the one thing the fact check cannot tell them.
    """
    if not resting_on:
        return
    st.warning(
        f"⚠️ This answer reproduces {len(resting_on)} passage(s) flagged as having a "
        "damaged value. Any date or amount it reports from them is suspect even if it "
        "reads cleanly — the fragments left behind by extraction can look like ordinary "
        "numbers."
    )
    with st.expander("Where this answer came from"):
        for snippet, overlap in resting_on:
            st.caption(f"Shared passage ({len(overlap)} characters):")
            if _is_rtl(snippet):
                st.markdown(
                    f"<div dir='rtl' style='background:#2a1e1e;padding:10px;"
                    f"border-radius:6px;margin-bottom:10px'>{snippet}</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(f"> {snippet}")


def _build_retrieval_chain(splits, k_val, doc_key, caveat=""):
    """Index the document and build the history-aware retrieval chain.

    Only used when the document is too large to answer from directly — see
    extraction.fits_in_context.
    """
    # Collection name is scoped to THIS session + THIS exact document — never
    # the raw filename — so two users uploading "resume.pdf" never land in the
    # same Chroma collection.
    collection_name = re.sub(
        r'[^a-zA-Z0-9_-]', '_',
        f"sess_{st.session_state.session_id}_{doc_key}"
    )[:60]

    st.session_state.vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name=collection_name,
    )

    # LCEL composition — ConversationalRetrievalChain is a legacy chain, moved
    # to langchain-classic in LangChain v1; this is what replaced it.
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": k_val})

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

    qa_prompt = ChatPromptTemplate.from_messages(_with_caveat([
        ("system", (
            "You are InsightStream, an expert document consultant. Use the "
            "following retrieved context to answer the user's question. If "
            "the answer isn't in the context, say you don't know — do not "
            "make anything up. " + EXTRACTION_RULE + "\n\n{context}"
        )),
        MessagesPlaceholder("chat_history"),
    ], caveat))
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    st.session_state.chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def render_fact_check(checked_facts):
    """Surface unverifiable figures prominently; confirm the rest quietly."""
    if not checked_facts:
        return
    missing = [value for value, found in checked_facts if not found]
    if missing:
        message = (
            "⚠️ These figures do not appear in the document text: "
            + ", ".join(f"**{value}**" for value in missing)
            + ". Extraction may have corrupted them, or the answer may have "
            "inferred them. Check against the original before relying on them."
        )
        # When OCR ran, the first explanation is much more likely than the
        # second — say so rather than leaving both equally weighted.
        if st.session_state.ocr_confidence is not None:
            message += (
                f" This document was read by OCR at {st.session_state.ocr_confidence:.0f}% "
                "mean confidence, so a figure that is absent from the text may still be "
                "present in the original scan."
            )
        st.warning(message)
    else:
        values = ", ".join(value for value, _ in checked_facts)
        st.caption(f"✅ Figures verified against the document text: {values}")


def render_quote_evidence(checked_quotes):
    """Render verified quotes using the same treatment as retrieved excerpts."""
    if not checked_quotes:
        st.caption("The model cited no supporting passage for this answer.")
        return
    for index, (quote, found) in enumerate(checked_quotes, 1):
        mark = "✅ found in document" if found else "⚠️ NOT found in document — treat with suspicion"
        st.caption(f"Quote {index} · {mark}")
        st.markdown(
            '<div dir="{direction}" style="'
            "white-space: pre-wrap; font-size: 0.88rem; line-height: 1.6; "
            "max-height: 320px; overflow-y: auto; "
            "padding: 0.75rem 0.95rem; margin-bottom: 1rem; "
            "border: 1px solid rgba(128,128,128,0.3); border-radius: 6px; "
            'background: rgba(128,128,128,0.07);">{body}</div>'.format(
                direction="rtl" if _is_rtl(quote) else "ltr",
                body=html.escape(quote.strip()),
            ),
            unsafe_allow_html=True,
        )


def render_evidence(sources):
    """Render the retrieved passages an answer was grounded in.

    Shows each excerpt in full rather than truncating: a legal citation cut
    off at 400 characters can't be verified, which defeats the point. Long
    passages scroll inside their own box instead.
    """
    for index, doc in enumerate(sources, 1):
        text = doc.page_content.strip()
        if not text:
            continue

        meta = [f"Excerpt {index}"]
        page = doc.metadata.get("page")
        if page is not None:
            meta.append(f"page {int(page) + 1}")
        if doc.metadata.get("extraction") == "ocr":
            meta.append("read via OCR")
        meta.append(f"{len(text):,} chars")
        st.caption(" · ".join(meta))

        # Document text is escaped before it reaches the DOM - it comes from an
        # arbitrary uploaded file, and pre-wrap keeps the line structure that
        # carries meaning in contracts.
        st.markdown(
            '<div dir="{direction}" style="'
            "white-space: pre-wrap; font-size: 0.88rem; line-height: 1.6; "
            "max-height: 320px; overflow-y: auto; "
            "padding: 0.75rem 0.95rem; margin-bottom: 1rem; "
            "border: 1px solid rgba(128,128,128,0.3); border-radius: 6px; "
            'background: rgba(128,128,128,0.07);">{body}</div>'.format(
                direction="rtl" if _is_rtl(text) else "ltr",
                body=html.escape(text),
            ),
            unsafe_allow_html=True,
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
    if st.session_state.doc_key is not None:
        _label, _tip = document_status()
        st.caption(_label, help=_tip)
        if st.button("🔥 Incinerate Document Data", key="sidebar_incinerate"):
            incinerate_document()
            st.rerun()
    else:
        st.caption("No document loaded.")

    st.markdown("---")
    st.subheader("👤 Developer")
    st.markdown("[GitHub Repository](https://github.com/royinsler/pdf-rag-explorer)")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/roy-insler-3a8042120)")
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

if st.session_state.doc_key is not None:
    with st.container(border=True):
        col_status, col_btn = st.columns([4, 1], vertical_alignment="center")
        with col_status:
            _label, _tip = document_status()
            st.caption(_label, help=_tip)
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

            extraction = extract_document(file_bytes, force_ocr=force_ocr)
            docs = extraction.docs

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_overlap)
            splits = text_splitter.split_documents(docs) if extraction.usable else []
            # Drop page headers/footers and stamp fragments before indexing:
            # they cannot answer a question but can still win a retrieval slot.
            splits, dropped_fragments = drop_tiny_chunks(splits)

            # Fail clearly rather than embedding nothing or embedding garbage.
            # Both halves of this matter: an empty index used to crash
            # Chroma.from_documents with a raw ValueError, and a garbled-but-
            # non-empty extraction used to sail through so the model answered
            # fluently from nonsense.
            if not extraction.usable or not splits:
                message = f"❌ Could not use this PDF: {extraction.reason or 'no readable text'}."
                if extraction.ocr_error:
                    message += (
                        f" The OCR fallback also failed to run ({extraction.ocr_error}). "
                        "If this document needs OCR, check that Tesseract is installed "
                        "locally (macOS: `brew install tesseract tesseract-lang`)."
                    )
                st.error(message)
                st.session_state.doc_key = None
                st.session_state.vectorstore = None
                st.session_state.chain = None
                st.stop()

            st.session_state.ocr_confidence = extraction.ocr_confidence
            st.session_state.low_confidence_terms = extraction.low_confidence_terms
            st.session_state.damaged_values = detect_damaged_values(
                extraction.text, extraction.low_confidence_terms)

            st.session_state.used_ocr = extraction.used_ocr

            doc_text = "\n\n".join(d.page_content for d in docs)
            st.session_state.full_context = fits_in_context(doc_text)

            if st.session_state.full_context:
                # Small enough to hand to the model whole. Skip embedding
                # entirely: no chunk boundary can split a clause, no top-k can
                # miss the relevant paragraph, and nothing is spent indexing.
                st.session_state.vectorstore = None
                st.session_state.chain = None
            else:
                _build_retrieval_chain(splits, k_val, doc_key, caveat=document_caveat())

            st.session_state.doc_key = doc_key
            st.session_state.doc_text = doc_text
            # A new document invalidates the old conversation and report.
            st.session_state.chat_history = []
            st.session_state.analysis_report = None


    render_extraction_report()

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
                render_garbage_echoes(turn.get("echoes", []))
                render_damaged_quotes(turn.get("damaged_quotes", []))
                render_damage_provenance(turn.get("rests_on", []))
                render_fact_check(turn.get("facts", []))
                if turn.get("quotes"):
                    with st.expander(f"🔍 View Context Evidence ({len(turn['quotes'])} quoted passages)"):
                        render_quote_evidence(turn["quotes"])
                elif turn.get("sources"):
                    with st.expander(f"🔍 View Context Evidence ({len(turn['sources'])} excerpts)"):
                        render_evidence(turn["sources"])

        if prompt := st.chat_input("Consult the document..."):
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                start = time.time()
                history_messages = []
                for t in st.session_state.chat_history:
                    history_messages.append(HumanMessage(content=t["user"]))
                    history_messages.append(AIMessage(content=t["bot"]))
                sources, quotes = [], []
                if st.session_state.full_context:
                    result = answer_from_full_document(
                        prompt, st.session_state.doc_text, history_messages,
                        caveat=document_caveat())
                    answer = result.answer
                    quotes = verify_quotes(result.quotes, st.session_state.doc_text)
                else:
                    res = st.session_state.chain.invoke(
                        {"input": prompt, "chat_history": history_messages})
                    answer = res["answer"]
                    sources = res.get("context", [])
                st.session_state.last_chat_duration = time.time() - start

                # Runs in BOTH modes: the fabricated date that motivated this
                # check came from the retrieval path, not from full context.
                facts = verify_answer_facts(answer, st.session_state.doc_text)
                # Check the cited passages too, not only the prose. An answer can
                # read cleanly while the evidence it offers is gibberish — which is
                # exactly what this document produced once the prompt stopped the
                # model quoting garbage in the answer itself.
                echoes, damaged_quotes = split_debris(
                    answer, [q for q, _ in quotes], st.session_state.doc_text)
                resting_on = answer_rests_on_damage(
                    answer, [d.snippet for d in st.session_state.damaged_values])

                st.markdown(answer)
                render_garbage_echoes(echoes)
                render_damaged_quotes(damaged_quotes)
                render_damage_provenance(resting_on)
                render_fact_check(facts)
                if quotes:
                    with st.expander(f"🔍 View Context Evidence ({len(quotes)} quoted passages)"):
                        render_quote_evidence(quotes)
                elif sources:
                    with st.expander(f"🔍 View Context Evidence ({len(sources)} excerpts)"):
                        render_evidence(sources)
                st.session_state.chat_history.append(
                    {"user": prompt, "bot": answer, "sources": sources,
                     "quotes": quotes, "facts": facts, "echoes": echoes,
                     "damaged_quotes": damaged_quotes,
                     "rests_on": resting_on})
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
                        if st.session_state.full_context:
                            answers = asyncio.run(run_parallel_full_context(
                                questions, st.session_state.doc_text,
                                caveat=document_caveat()))
                        else:
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
