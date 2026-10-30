"""
Document extraction pipeline, kept separate from app.py so it can be imported
without starting Streamlit — which is what lets stress_inputs.py exercise the
*real* decision logic rather than a copy that drifts.

The pipeline is: try the text layer, fall back to OCR when it is missing or
looks corrupted, then decide whether the result is usable at all.
"""
import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional

from langchain_community.document_loaders import PDFMinerLoader

OCR_LANG = "heb+eng"

# Tesseract scores every word 0-100. Below this we treat the word as unreliable
# and say so, rather than passing it downstream as if it were read cleanly.
#
# This exists because of a specific failure: OCR destroyed a date, leaving the
# token "DYN" where "8.8.2027" was printed, and the model then assembled a
# plausible date from a nearby clause number and presented it as fact. Tesseract
# almost certainly scored that token poorly — but the pipeline called
# image_to_string(), which returns text and discards confidence entirely. The
# engine knew it was guessing and we threw the signal away, then tried to infer
# the same thing downstream with prompt rules and regexes.
OCR_MIN_CONFIDENCE = 60.0

# Chunks shorter than this are dropped before indexing. OCR emits one Document
# per page, so a page that is mostly form scaffolding yields a fragment like
# "מוכר קונה 2012-351" (19 chars) - a header, not a clause. It cannot answer
# anything, but it CAN win a retrieval slot, because short text scores
# deceptively high cosine similarity against a short query. Observed in
# practice: one of seven slots on a real query went to exactly that fragment.
#
# Not measured by eval_rag.py - its corpus is clean text that never passes
# through OCR, so it contains no such fragments and the filter is a no-op
# there. Justified by reasoning and by chunk_report.py on real documents.
MIN_CHUNK_CHARS = 80

# Documents at or below this size are answered from full context instead of
# through retrieval. RAG earns its place when a document exceeds what you can
# hand the model directly; below that it strictly loses information, because
# chunking can split a clause and top-k can miss the relevant paragraph.
#
# The number is a judgement call, not a measured optimum: gpt-4o-mini has a
# 128k-token window, and 30k characters is roughly 8-15k tokens depending on
# script (Hebrew tokenizes worse than English), leaving ample room for the
# prompt, the chat history and the answer. Erring small keeps per-query cost
# predictable, since full-context mode resends the document every turn.
FULL_CONTEXT_MAX_CHARS = 30_000


def fits_in_context(text: str, limit: int = FULL_CONTEXT_MAX_CHARS) -> bool:
    """True when the whole document can be handed to the model directly."""
    return len(text) <= limit


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


# Unicode bidi control characters. Tesseract's rendered output wraps Latin runs
# inside Hebrew lines in these to tell a renderer which way to lay them out.
# They are a presentation hint, not content, and this pipeline sets direction at
# render time from the passage's own composition (see app._is_rtl), so storing
# them buys nothing and costs something: an invisible mark that lands inside a
# date or an amount silently breaks the substring matching that quote
# verification and fact checking depend on.
BIDI_CONTROLS = dict.fromkeys(
    [0x200E, 0x200F, 0x061C] + list(range(0x202A, 0x202F)) + list(range(0x2066, 0x206A))
)


def strip_bidi(text: str) -> str:
    """Remove bidi control characters, leaving the visible text untouched."""
    return text.translate(BIDI_CONTROLS)


def ocr_extract_documents(file_bytes: bytes, lang: str = OCR_LANG,
                          min_confidence: float = OCR_MIN_CONFIDENCE,
                          dpi: int = 300, only_pages=None):
    """Fallback extraction for PDFs whose text layer is missing or corrupted.

    Rasterizes each page with PyMuPDF and reads it with Tesseract, bypassing the
    PDF's font structures entirely — this reads pixels, so a broken ToUnicode
    CMap cannot affect it.

    Uses image_to_data rather than image_to_string so per-word confidence
    survives. Words scoring below min_confidence are recorded in metadata as
    unreliable; the text itself is left unmarked, so embeddings and display are
    not polluted by annotation.

    `dpi` and `only_pages` (0-based page numbers) exist for the agent's re-read
    tool: the same pipeline, pointed at one suspect page at a higher resolution.
    The defaults reproduce the original behaviour exactly - every page, 300 DPI.

    Requires the `tesseract` binary (macOS: `brew install tesseract
    tesseract-lang`) plus `pymupdf` and `pytesseract`.
    """
    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image
    from langchain_core.documents import Document

    docs = []
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        for page_num, page in enumerate(pdf):
            if only_pages is not None and page_num not in only_pages:
                continue
            # 300 DPI by default — PDF default is 72, and OCR accuracy drops sharply below ~200.
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            data = pytesseract.image_to_data(
                image, lang=lang, output_type=pytesseract.Output.DICT)

            # Rebuild line structure from Tesseract's block/paragraph/line ids —
            # line breaks carry meaning in contracts, and a flat word join
            # would lose them.
            lines, scores, suspect = {}, [], []
            for i, raw in enumerate(data["text"]):
                token = strip_bidi(raw).strip()
                if not token:
                    continue
                try:
                    score = float(data["conf"][i])
                except (TypeError, ValueError):
                    continue
                if score < 0:            # Tesseract marks non-text rows -1
                    continue
                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                lines.setdefault(key, []).append(token)
                scores.append(score)
                if score < min_confidence:
                    suspect.append((token, round(score, 1)))

            text = "\n".join(" ".join(words) for _, words in sorted(lines.items()))
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "page": page_num,
                        "extraction": "ocr",
                        "ocr_confidence": round(sum(scores) / len(scores), 1) if scores else None,
                        "low_confidence_terms": suspect,
                    },
                ))
    finally:
        pdf.close()
    return docs


@dataclass
class ExtractionResult:
    docs: List            # LangChain Documents, empty if nothing usable
    text: str             # concatenated page content
    used_ocr: bool
    ocr_confidence: Optional[float]      # mean Tesseract word confidence
    low_confidence_terms: List           # [(token, score)] Tesseract was unsure of
    load_error: Optional[str]
    ocr_error: Optional[str]
    usable: bool          # passed the final guard
    reason: Optional[str] # why not, when usable is False


def extract_document(file_bytes: bytes, force_ocr: bool = False,
                     lang: str = OCR_LANG) -> ExtractionResult:
    """Run the full extraction decision and report what happened.

    Mirrors exactly what app.py does on upload, so a test against this
    function is a test of the shipped behaviour.
    """
    docs = []
    load_error = None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        if not force_ocr:
            try:
                docs = PDFMinerLoader(tmp_path).load()
            except Exception as e:
                # Malformed, truncated, encrypted, or not-a-PDF-at-all input
                # makes the loader raise. Stress testing found this escaping
                # into Streamlit as a raw traceback: the user saw a stack
                # trace instead of an explanation, and st.stop() never ran so
                # session state was left half-initialised.
                load_error = f"{type(e).__name__}: {e}"
    finally:
        # Always clean up; uploaded PDFs used to accumulate in /tmp forever.
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    text = "".join(d.page_content for d in docs)
    used_ocr = False
    ocr_error = None

    # Fall back to OCR when asked explicitly, when the text layer is empty
    # (scanned/image-only), or when it looks corrupted.
    if force_ocr or not text.strip() or looks_garbled(text):
        try:
            ocr_docs = ocr_extract_documents(file_bytes, lang=lang)
            ocr_text = "".join(d.page_content for d in ocr_docs)
            if ocr_docs and not looks_garbled(ocr_text):
                docs, text, used_ocr = ocr_docs, ocr_text, True
        except Exception as e:
            ocr_error = f"{type(e).__name__}: {e}"

    # Aggregate the OCR confidence signal across pages, worst first, deduped.
    ocr_confidence, low_confidence_terms = None, []
    if used_ocr:
        page_scores = [d.metadata.get("ocr_confidence") for d in docs]
        page_scores = [s for s in page_scores if s is not None]
        if page_scores:
            ocr_confidence = round(sum(page_scores) / len(page_scores), 1)
        seen = set()
        for doc in docs:
            for token, score in doc.metadata.get("low_confidence_terms", []):
                if token not in seen:
                    seen.add(token)
                    low_confidence_terms.append((token, score))
        low_confidence_terms.sort(key=lambda pair: pair[1])

    if not text.strip():
        usable = False
        if load_error and ocr_error:
            reason = ("the file could not be read as a PDF — it may be corrupted, "
                      "password-protected, or not a PDF at all")
        elif load_error:
            reason = f"the PDF's text layer could not be read ({load_error})"
        else:
            reason = "no text could be extracted"
    elif looks_garbled(text):
        usable, reason = False, "extracted text looks corrupted"
    else:
        usable, reason = True, None

    return ExtractionResult(docs, text, used_ocr, ocr_confidence,
                            low_confidence_terms, load_error, ocr_error,
                            usable, reason)


def drop_tiny_chunks(splits, min_chars: int = MIN_CHUNK_CHARS):
    """Remove chunks too short to carry a usable answer.

    Returns (kept, dropped) so callers can report what was removed rather than
    silently discarding document content.
    """
    kept, dropped = [], []
    for chunk in splits:
        (kept if len(chunk.page_content.strip()) >= min_chars else dropped).append(chunk)
    return kept, dropped


def ocr_caveat(low_confidence_terms, limit: int = 30) -> str:
    """A prompt fragment naming the tokens OCR was unsure about.

    Passing this to the model is the principled version of the hand-written
    rule about not inferring dates: instead of telling it to be careful in
    general, it is told exactly which tokens are untrustworthy.
    """
    if not low_confidence_terms:
        return ""
    shown = ", ".join(token for token, _ in low_confidence_terms[:limit])
    more = "" if len(low_confidence_terms) <= limit else f" (and {len(low_confidence_terms) - limit} more)"
    return (
        " This document was read by OCR. These tokens were recognised with LOW "
        f"CONFIDENCE and are probably misread: {shown}{more}. Treat any value "
        "containing them as illegible — state that the document does not show it "
        "legibly rather than guessing what it was meant to be."
    )
