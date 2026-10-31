"""Read-only tools the verification agent uses to look at the document again.

Three tools, all of them reads:

  search_document - find where the document says something
  read_page       - read one page as extracted
  reocr_page      - read one page AGAIN from its pixels at a higher resolution

None of them writes anything, so none needs the human-approval gate a write
action would. They are real LangChain StructuredTools with strict argument
schemas, which means the same objects can be bound to a tool-calling model
later; agent_graph currently drives them from verification findings instead
(see investigate there), and says so.

Failure policy: a tool never raises into the graph. run_tool turns bad
arguments, a missing Tesseract, an unmapped page - anything - into
ToolResult(ok=False, error=...). A broken tool should cost the agent one piece
of evidence, not the whole run.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# Tool output goes into a prompt, so it is bounded. Both numbers are cost
# controls as much as safety ones: every character here is paid for again on
# each retry.
MAX_TOOL_OUTPUT_CHARS = 1200
REOCR_DPI = 400                    # the pipeline's default is 300

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip()


def dates_in(text: str):
    """The set of (day, month, year) dates written in `text`, whatever the separator or padding."""
    from verification import _DATE_RE, _canon_date
    return {d for d in (_canon_date(m) for m in re.findall(_DATE_RE, text)) if d}


def fmt_date(date) -> str:
    day, month, year = date
    return f"{day:02d}.{month:02d}.{year}"


def amounts_in(text: str) -> Dict[str, str]:
    """{normalised amount: as written} for the comma-formatted amounts verification checks."""
    from verification import _AMOUNT_RE
    found: Dict[str, str] = {}
    for raw in re.findall(_AMOUNT_RE, text):
        found.setdefault(re.sub(r"[,\s]", "", raw), raw)
    return found


@dataclass
class DocumentContext:
    """Everything the agent may look at, held OUTSIDE graph state.

    See agent_state: state is checkpointed, so the document stays here. This
    object is captured by the graph's nodes and never serialised.

    `pages` maps 1-based PDF page number to its extracted text, and is empty
    unless that mapping is trustworthy. OCR yields one Document per page, so it
    is; the PDF text layer comes back as one undifferentiated Document, so page
    numbers would be a guess. A guess presented as a page number is how a
    re-read ends up looking at the wrong page, so the tools refuse instead.
    """
    text: str
    pages: Dict[int, str] = field(default_factory=dict)
    file_bytes: Optional[bytes] = None
    damaged: List[Any] = field(default_factory=list)        # [DamagedValue]
    # The prompt fragment naming damaged tokens and passages (ocr_caveat +
    # damage_caveat), exactly what app.document_caveat() assembles. Raw text:
    # escaping for the prompt template is the answering layer's concern.
    caveat: str = ""
    ocr_lang: Optional[str] = None                          # None -> extraction.OCR_LANG
    # Every re-read of a page: page -> {(dpi, psm, threshold): text}. A re-read
    # is evidence, NOT an authority. It enters the verification corpus only
    # through corroborated(), i.e. when at least two different reads agree.
    #
    # An earlier version appended every re-read to the corpus on the theory that
    # a value found on a second read of the document's own pixels "is not an
    # invented one". probe_reread.py on a real contract showed otherwise: the
    # same printed date 4.9.27 was read correctly at 300 and 600 DPI and as
    # 24.09.2027 at 400, and that single misread would have passed the fact
    # check as "present in the document".
    rereads: Dict[int, Dict[Any, str]] = field(default_factory=dict)
    _cache: Dict[Any, Any] = field(default_factory=dict, repr=False)
    _norm_pages: Optional[Dict[int, str]] = field(default=None, repr=False)
    _bounds: Any = field(default=None, repr=False)

    @classmethod
    def from_extraction(cls, extraction, file_bytes=None, damaged=None, ocr_lang=None):
        """Build from extraction.extract_document's result.

        Mirrors what app.py does on upload so the agent sees the same document
        the chat does: the text is the pages joined with blank lines (app's
        doc_text), damage is detected on the raw extraction text, and the
        caveat is the same OCR-plus-damage fragment.
        """
        from damaged_values import damage_caveat, detect_damaged_values
        from extraction import ocr_caveat

        if damaged is None:
            damaged = detect_damaged_values(extraction.text, extraction.low_confidence_terms)
        pages = {}
        if extraction.used_ocr:
            for doc in extraction.docs:
                page = doc.metadata.get("page")
                if isinstance(page, int):
                    pages[page + 1] = doc.page_content      # metadata is 0-based
        return cls(
            text="\n\n".join(d.page_content for d in extraction.docs),
            pages=pages, file_bytes=file_bytes, damaged=list(damaged),
            caveat=ocr_caveat(extraction.low_confidence_terms) + damage_caveat(damaged),
            ocr_lang=ocr_lang)

    def verification_text(self) -> str:
        """What facts and quotes are checked against.

        The original extraction, plus only those dates and amounts that two or
        more different re-reads of a page agree on. The re-read TEXT is never
        added, so a quote that exists only in a single re-read is still flagged,
        and so is a value that only one re-read produced.
        """
        extra = [text for _, _, text in self._corroborated()]
        return self.text if not extra else self.text + "\n" + "\n".join(extra)

    def _corroborated(self, page=None, min_reads=2):
        """[(kind, key, text)] for values at least `min_reads` DIFFERENT reads of one page agree on.

        'Different' means a different (dpi, psm, threshold): the same
        configuration read twice is the same read and counts once.
        """
        found = []
        for number in ([page] if page is not None else sorted(self.rereads)):
            reads = list(self.rereads.get(number, {}).values())
            if len(reads) < min_reads:
                continue
            date_sets = [dates_in(t) for t in reads]
            for date in sorted(set().union(*date_sets)):
                if sum(date in s for s in date_sets) >= min_reads:
                    found.append(("date", date, fmt_date(date)))
            amount_maps = [amounts_in(t) for t in reads]
            for key in sorted(set().union(*amount_maps)):
                if sum(key in m for m in amount_maps) >= min_reads:
                    raw = next(m[key] for m in amount_maps if key in m)
                    found.append(("amount", key, raw))
        return found

    def corroborated_new(self, page: int) -> List[str]:
        """Values two or more re-reads of `page` agree on that the ORIGINAL page did not contain.

        These are the genuinely recovered ones. A value the original already had
        is not news, and a value only one read produced is not corroborated.
        """
        original = self.pages.get(page, "")
        had_dates, had_amounts = dates_in(original), set(amounts_in(original))
        return [text for kind, key, text in self._corroborated(page)
                if (kind == "date" and key not in had_dates)
                or (kind == "amount" and key not in had_amounts)]

    def page_at(self, offset: int) -> Optional[int]:
        """1-based page holding `offset` in the whitespace-normalised document text.

        Exact, but only when the pages really do add up to the document text. If
        they do not (a context built by hand, a blank page collapsing), the map
        is unusable and this returns None rather than a plausible wrong page -
        which is what the fragment search below is for.
        """
        if self._bounds is None:
            ordered = sorted(self.pages.items())
            normed = [(n, _norm(t)) for n, t in ordered]
            bounds, start = [], 0
            for number, text in normed:
                bounds.append((start, start + len(text), number))
                start += len(text) + 1                  # the single joining space
            joined = " ".join(t for _, t in normed)
            self._bounds = bounds if self.pages and joined == _norm(self.text) else []
        for start, end, number in self._bounds:
            if start <= offset <= end:
                return number
        return None

    def page_containing(self, fragment: str) -> Optional[int]:
        """1-based page whose text contains the fragment, if pages are mapped."""
        if not self.pages or not fragment:
            return None
        if self._norm_pages is None:
            self._norm_pages = {n: _norm(t).lower() for n, t in self.pages.items()}
        needle = _norm(fragment).lower()
        for number, text in sorted(self._norm_pages.items()):
            if needle in text:
                return number
        return None


class ToolResult(BaseModel):
    ok: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


# --- argument schemas --------------------------------------------------------

class SearchDocumentArgs(BaseModel):
    query: str = Field(min_length=2, max_length=200,
                       description="Words or a phrase to look for in the document.")
    max_hits: int = Field(default=3, ge=1, le=5, description="How many passages to return.")
    window: int = Field(default=120, ge=40, le=400,
                        description="Characters of context on each side of a hit.")


class ReadPageArgs(BaseModel):
    page: int = Field(ge=1, description="1-based PDF page number.")
    max_chars: int = Field(default=1500, ge=200, le=4000)


class ReocrPageArgs(BaseModel):
    page: int = Field(ge=1, description="1-based PDF page number.")
    dpi: int = Field(default=REOCR_DPI, ge=200, le=600,
                     description="Resolution to rasterise at before OCR.")
    psm: Optional[int] = Field(default=None, ge=0, le=13,
                               description="Tesseract page-segmentation mode (6 = one block, "
                                           "11/12 = sparse text). Default: Tesseract's own.")
    threshold: Optional[int] = Field(default=None, ge=1, le=254,
                                     description="Convert to black and white at this grey level "
                                                 "before OCR. Default: no conversion.")


# --- tool bodies -------------------------------------------------------------

def _search(ctx: DocumentContext, query: str, max_hits: int = 3, window: int = 120) -> Dict[str, Any]:
    text = _norm(ctx.text)
    haystack = text.lower()
    needle = _norm(query).lower()

    # Exact phrase first - it is the strongest evidence and the cheapest to read.
    starts, start = [], haystack.find(needle)
    while start != -1 and len(starts) < max_hits:
        starts.append(start)
        start = haystack.find(needle, start + len(needle))

    if not starts:
        # No phrase match: score candidate positions by how many distinct query
        # words fall inside the same window. Lexical on purpose - the tool must
        # work with no embedding call and no API key.
        words = {w for w in re.findall(r"\w+", needle) if len(w) >= 3}
        scored = []
        for word in words:
            for match in re.finditer(re.escape(word), haystack):
                span = haystack[max(0, match.start() - window): match.end() + window]
                scored.append((sum(w in span for w in words), match.start()))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        for score, position in scored:
            if score and all(abs(position - s) > window for s in starts):
                starts.append(position)
            if len(starts) >= max_hits:
                break

    hits = []
    for position in starts:
        excerpt = text[max(0, position - window): position + len(needle) + window]
        # The page of the MATCH, not of the excerpt: the excerpt starts up to a
        # window before the hit, which near a page boundary is the previous page.
        # (An earlier version took the excerpt's first characters and reported
        # page 1 for everything near the top of the document.)
        page = ctx.page_at(position) or ctx.page_containing(text[position: position + 60])
        hits.append({"page": page, "excerpt": excerpt})
    return {"query": query, "hits": hits}


def _read_page(ctx: DocumentContext, page: int, max_chars: int = 1500) -> Dict[str, Any]:
    if not ctx.pages:
        raise ValueError("page numbering is unavailable for this document "
                         "(text-layer extraction does not keep pages); use search_document")
    if page not in ctx.pages:
        raise ValueError(f"page {page} not found; pages are {min(ctx.pages)}-{max(ctx.pages)}")
    text = ctx.pages[page]
    return {"page": page, "chars": len(text), "truncated": len(text) > max_chars,
            "text": text[:max_chars]}


def _reocr(ctx: DocumentContext, page: int, dpi: int = REOCR_DPI,
           psm: Optional[int] = None, threshold: Optional[int] = None) -> Dict[str, Any]:
    if ctx.file_bytes is None:
        raise ValueError("the original PDF bytes are not available, so the page cannot be re-read")
    if page not in ctx.pages:
        raise ValueError(f"page {page} has no reliable mapping to the PDF "
                         "(only OCR extraction records page numbers)")
    config = (dpi, psm, threshold)
    key = ("reocr", page, config)
    if key in ctx._cache:                       # OCR is the expensive call; never repeat it
        return {**ctx._cache[key], "cached": True}

    from extraction import ocr_extract_documents, OCR_LANG   # lazy: pulls in PyMuPDF/Tesseract
    docs = ocr_extract_documents(ctx.file_bytes, lang=ctx.ocr_lang or OCR_LANG,
                                 dpi=dpi, only_pages=[page - 1], psm=psm, threshold=threshold)
    if not docs:
        raise ValueError("the re-read recognised no text on that page")
    doc = docs[0]
    result = {
        "page": page, "dpi": dpi, "psm": psm, "threshold": threshold,
        "mean_confidence": doc.metadata.get("ocr_confidence"),
        "low_confidence_terms": [t for t, _ in doc.metadata.get("low_confidence_terms", [])][:10],
        "text": doc.page_content,
        "cached": False,
    }
    ctx._cache[key] = result
    ctx.rereads.setdefault(page, {})[config] = doc.page_content
    return result


def build_tools(ctx: DocumentContext) -> Dict[str, StructuredTool]:
    """The tool set for one document, keyed by name."""
    tools = [
        StructuredTool.from_function(
            func=lambda **kw: _search(ctx, **kw), name="search_document",
            description="Find where the document mentions something. Returns short excerpts.",
            args_schema=SearchDocumentArgs),
        StructuredTool.from_function(
            func=lambda **kw: _read_page(ctx, **kw), name="read_page",
            description="Read one page of the document as it was extracted.",
            args_schema=ReadPageArgs),
        StructuredTool.from_function(
            func=lambda **kw: _reocr(ctx, **kw), name="reocr_page",
            description="Re-read one page from its pixels at higher resolution; "
                        "use when a value on that page looks damaged.",
            args_schema=ReocrPageArgs),
    ]
    return {t.name: t for t in tools}


def run_tool(tools: Dict[str, StructuredTool], name: str, **args) -> ToolResult:
    """Call a tool; never raise. Bad arguments and runtime failures both come back as ok=False."""
    tool = tools.get(name)
    if tool is None:
        return ToolResult(ok=False, error=f"unknown tool {name!r}")
    try:
        return ToolResult(ok=True, data=tool.invoke(args))
    except Exception as exc:                                    # noqa: BLE001 - by design
        return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")


def excerpt_around(text: str, needle: str, width: int = 70, max_hits: int = 2) -> List[str]:
    """Short windows around each occurrence of `needle` in `text`."""
    flat, out, start = _norm(text), [], 0
    while len(out) < max_hits:
        position = flat.find(needle, start)
        if position == -1:
            break
        out.append(flat[max(0, position - width): position + len(needle) + width])
        start = position + len(needle)
    return out
