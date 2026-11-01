"""Tests for the agent's read-only tools.

The contract being tested: a tool never raises into the graph, bad input is a
result not a crash, and nothing is guessed - an unmapped page is refused, not
approximated. The re-read test runs real PyMuPDF and real Tesseract on a PDF it
builds itself, and is skipped (loudly) if Tesseract is not installed.

    python3 test_agent_tools.py
"""
import sys
from types import SimpleNamespace

from agent_tools import (DocumentContext, build_tools, excerpt_around, run_tool)
from damaged_values import detect_damaged_values

results = []
skipped = []


def check(name, actual, expected):
    ok = actual == expected
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"         expected {expected!r}\n         got      {actual!r}")


PAGE1 = 'המוכר מתחייב לפנות את הדירה ולמסור אותה לידי הקונה ביום 4.11.2027.'
PAGE2 = 'סך של 410,000 ש"ח ישולם עד ליום 3.9.2027 כנגד קבלת כספי המשכנתא.'
ctx = DocumentContext(text=PAGE1 + "\n\n" + PAGE2, pages={1: PAGE1, 2: PAGE2})
tools = build_tools(ctx)

# --- search_document ---------------------------------------------------------
hit = run_tool(tools, "search_document", query="410,000")
check("exact phrase is found", hit.ok and len(hit.data["hits"]) == 1, True)
check("the hit carries the right page", hit.data["hits"][0]["page"], 2)

# Word-level fallback: no exact phrase, but the words co-occur.
fuzzy = run_tool(tools, "search_document", query="פינוי הדירה לקונה מתחייב")
check("no exact phrase falls back to word matching", fuzzy.ok and len(fuzzy.data["hits"]) >= 1, True)

none = run_tool(tools, "search_document", query="zzzz-not-there")
check("a query with no match is ok with zero hits, not an error", (none.ok, none.data["hits"]), (True, []))

# Regression: the page was once taken from the excerpt's first characters, so a
# hit on page 2 reported page 1 whenever the excerpt window reached back across
# the boundary. Both pages here are short enough that every window does.
check("a hit near the top of page 2 is attributed to page 2, not 1",
      run_tool(tools, "search_document", query="המשכנתא").data["hits"][0]["page"], 2)
check("a hit on page 1 is attributed to page 1",
      run_tool(tools, "search_document", query="לפנות את הדירה").data["hits"][0]["page"], 1)
# Pages that do not add up to the document text cannot be offset-mapped; the
# fallback is a fragment search, and with no match it must say None.
mismatch = DocumentContext(text="completely different text", pages={1: "something else entirely"})
check("pages that do not add up to the text give no page offset", mismatch.page_at(3), None)
check("a fragment on no page gives None", mismatch.page_containing("not anywhere"), None)

# --- argument validation: a result, never an exception ----------------------
short = run_tool(tools, "search_document", query="x")
check("a too-short query is rejected as a result", (short.ok, short.error.startswith("ValidationError")), (False, True))
wide = run_tool(tools, "search_document", query="valid", max_hits=99)
check("an out-of-range argument is rejected", wide.ok, False)
check("an unknown tool is a result, not a KeyError", run_tool(tools, "format_disk").ok, False)

# --- read_page ---------------------------------------------------------------
page = run_tool(tools, "read_page", page=1)
check("a mapped page is read", (page.ok, page.data["text"]), (True, PAGE1))
check("a max_chars under the schema minimum is rejected",
      run_tool(tools, "read_page", page=1, max_chars=100).ok, False)
long_tools = build_tools(DocumentContext(text="א" * 500, pages={1: "א" * 500}))
long_read = run_tool(long_tools, "read_page", page=1, max_chars=200)
check("a long page is truncated and says so",
      (long_read.ok, len(long_read.data["text"]), long_read.data["truncated"], long_read.data["chars"]),
      (True, 200, True, 500))
check("a page that does not exist is refused", run_tool(tools, "read_page", page=9).ok, False)

unmapped = build_tools(DocumentContext(text=PAGE1))     # text-layer document: no page map
refused = run_tool(unmapped, "read_page", page=1)
# The text layer comes back as one Document, so a page number would be a guess.
check("with no page map, read_page refuses instead of guessing",
      (refused.ok, "unavailable" in refused.error), (False, True))

# --- reocr_page: refusals ----------------------------------------------------
check("re-read without the original bytes is refused",
      run_tool(tools, "reocr_page", page=1).ok, False)
check("re-read of an unmapped page is refused even with bytes",
      run_tool(build_tools(DocumentContext(text="x", file_bytes=b"%PDF")), "reocr_page", page=1).ok, False)

# --- re-reads are evidence, not authority -------------------------------------
# The real case behind this rule (probe_reread.py on a real contract): the page
# prints 4.9.27. The original extraction read it right, and so did 600 DPI, but
# 400 DPI read it as 24.09.2027. Before the rule, that one misread joined the
# corpus and would have passed the fact check as "present in the document".
ORIGINAL = "Final payment on 4.9.27 as agreed, from the start."
FROM_400 = "Final payment on 24.09.2027 as agreed, from the start."
FROM_600 = "Final payment on 4.9.27 as agreed, from the start."
CFG = lambda dpi: (dpi, None, None)
from verification import verify_answer_facts, verify_quotes

real = DocumentContext(text=ORIGINAL, pages={1: ORIGINAL},
                       rereads={1: {CFG(400): FROM_400, CFG(600): FROM_600}})
check("REAL CASE: a date only one re-read produced is not in the verification corpus",
      "24.09.2027" in real.verification_text(), False)
check("REAL CASE: ...so an answer stating it is flagged",
      verify_answer_facts("Payment is due 24.09.2027", real.verification_text()), [("24.09.2027", False)])
check("REAL CASE: ...while the correct date, from the original, still verifies",
      verify_answer_facts("Payment is due 4.9.27".replace("4.9.27", "04.09.2027"), real.verification_text()),
      [("04.09.2027", True)])
check("REAL CASE: a date the original already had is not reported as recovered",
      real.corroborated_new(1), [])

agree = DocumentContext(text="Start on DYN 6.", pages={1: "Start on DYN 6."},
                        rereads={1: {CFG(400): "Start from 11.09.2027 onward", CFG(600): "Start from 11.9.27 onward"}})
check("two reads agreeing on a date in different notations corroborate it, once",
      agree.corroborated_new(1), ["11.09.2027"])
check("...and it verifies", verify_answer_facts("Starts 11.09.2027", agree.verification_text()), [("11.09.2027", True)])
check("...but the re-read TEXT is never added: a quote found only there is still flagged",
      verify_quotes(["Start from 11.9.27 onward"], agree.verification_text()), [("Start from 11.9.27 onward", False)])

# Both re-reads agree, but the ORIGINAL page already had the values: confirmed, not recovered.
known = DocumentContext(text="Due 11.09.2027 and 275,000.", pages={1: "Due 11.09.2027 and 275,000."},
                        rereads={1: {CFG(400): "Due 11.9.27 and 275,000.", CFG(600): "Due 11.09.2027 and 275,000."}})
check("values the original page already had are not reported as recovered, even when re-reads agree",
      known.corroborated_new(1), [])

disagree = DocumentContext(text="Start on DYN 6.", pages={1: "Start on DYN 6."},
                           rereads={1: {CFG(400): "Start 11.09.2027", CFG(600): "Start 16.05.2026"}})
check("two reads that disagree corroborate nothing", disagree.corroborated_new(1), [])

two_pages = DocumentContext(text="a b", pages={1: "a", 2: "b"},
                            rereads={1: {CFG(400): "on 11.09.2027"}, 2: {CFG(400): "on 11.09.2027"}})
check("the same date from single reads of two DIFFERENT pages is not corroboration",
      (two_pages.corroborated_new(1), two_pages.corroborated_new(2), "11.09.2027" in two_pages.verification_text()),
      ([], [], False))

amounts = DocumentContext(text="Pay DYN", pages={1: "Pay DYN"},
                          rereads={1: {CFG(400): "Pay 275,000 now", CFG(600): "Pay 275,000 now"}})
check("amounts are corroborated the same way", amounts.corroborated_new(1), ["275,000"])
check("a page with no re-reads has nothing to corroborate", DocumentContext(text="x", pages={1: "x"}).corroborated_new(1), [])

check("psm outside Tesseract's range is rejected as a result",
      run_tool(build_tools(DocumentContext(text="x", pages={1: "x"}, file_bytes=b"%PDF")),
               "reocr_page", page=1, psm=99).ok, False)
check("a threshold outside 1-254 is rejected as a result",
      run_tool(build_tools(DocumentContext(text="x", pages={1: "x"}, file_bytes=b"%PDF")),
               "reocr_page", page=1, threshold=0).ok, False)

# --- reocr_page: the real thing ---------------------------------------------
def tesseract_ready():
    try:
        import pytesseract
        return "eng" in pytesseract.get_languages()
    except Exception:
        return False


if tesseract_ready():
    import fitz

    pdf = fitz.open()
    pdf.new_page().insert_text((72, 120), "The seller shall deliver by 12.03.2028 after payment.", fontsize=14)
    pdf_bytes = pdf.tobytes()
    pdf.close()

    damaged_text = "The seller shall deliver by DYN 6 after payment."
    rctx = DocumentContext(text=damaged_text, pages={1: damaged_text}, file_bytes=pdf_bytes, ocr_lang="eng")
    rtools = build_tools(rctx)
    reread = run_tool(rtools, "reocr_page", page=1)
    check("re-read succeeds on a real PDF", reread.ok, True)
    check("the re-read recovers the date the first extraction lost",
          "12.03.2028" in reread.data["text"], True)
    check("the re-read reports the resolution it used", reread.data["dpi"], 400)
    # A single re-read is evidence, not authority: see the real-case test above.
    check("a single re-read is NOT added to the verification corpus",
          "12.03.2028" in rctx.verification_text(), False)
    check("...and the original extraction is untouched", rctx.text, damaged_text)

    again = run_tool(rtools, "reocr_page", page=1)
    check("a repeat call is served from cache", again.data["cached"], True)
    check("...and is still one read, not two", len(rctx.rereads[1]), 1)
    check("one read corroborates nothing", rctx.corroborated_new(1), [])

    second = run_tool(rtools, "reocr_page", page=1, dpi=600)
    check("a read at a different resolution is a second read", (second.ok, len(rctx.rereads[1])), (True, 2))
    check("two reads that agree corroborate the date the original lacked",
          rctx.corroborated_new(1), ["12.03.2028"])
    check("...and only then does it join the verification corpus",
          "12.03.2028" in rctx.verification_text(), True)

    check("psm and threshold pass through to the OCR and are reported back",
          (run_tool(rtools, "reocr_page", page=1, dpi=300, psm=6).data["psm"],
           run_tool(rtools, "reocr_page", page=1, dpi=300, threshold=150).data["threshold"]), (6, 150))
    # Returned-in-the-result is not the same as applied. Spy on what reaches Tesseract:
    # the config string, and the image (its mode, and for a converted one its pixel values).
    seen = []
    import pytesseract
    real_image_to_data = pytesseract.image_to_data

    def spy(image, *args, **kwargs):
        values = sorted({v for _, v in (image.getcolors(256) or [])}) if image.mode == "L" else None
        seen.append((kwargs.get("config"), image.mode, values))
        return real_image_to_data(image, *args, **kwargs)

    pytesseract.image_to_data = spy
    try:
        spy_tools = build_tools(DocumentContext(text=damaged_text, pages={1: damaged_text},
                                                file_bytes=pdf_bytes, ocr_lang="eng"))
        run_tool(spy_tools, "reocr_page", page=1, dpi=300)
        run_tool(spy_tools, "reocr_page", page=1, dpi=300, psm=11)
        run_tool(spy_tools, "reocr_page", page=1, dpi=300, threshold=150)
    finally:
        pytesseract.image_to_data = real_image_to_data
    check("with neither option the OCR is called exactly as before: no config, the colour image",
          seen[0], ("", "RGB", None))
    check("psm reaches Tesseract as --psm N", seen[1][0], "--psm 11")
    check("threshold converts the image to pure black and white before the OCR sees it",
          seen[2][1:], ("L", [0, 255]))
    check("a black-and-white conversion still reads clean text",
          "12.03.2028" in run_tool(rtools, "reocr_page", page=1, dpi=300, threshold=150).data["text"], True)
    check("a page past the end of the PDF is a result, not a crash",
          run_tool(build_tools(DocumentContext(text="x", pages={5: "x"}, file_bytes=pdf_bytes,
                                               ocr_lang="eng")), "reocr_page", page=5).ok, False)
else:
    skipped.append("reocr_page against a real PDF (Tesseract with 'eng' not available)")

# --- from_extraction ---------------------------------------------------------
# The preamble keeps the connector away from the page boundary. extraction.text
# is the pages joined with NO separator, so a snippet that reaches back past the
# start of a page spans two pages - and page_containing must say None then, not guess.
DAMAGED = ('סעיף 7. לצורך ביצוע עבדות שיפוץ בלבד החל DYN 6,. מובהר כי כניסת הקונים לנכס במועד האמור.')
ocr_extraction = SimpleNamespace(
    used_ocr=True, text=PAGE1 + DAMAGED, low_confidence_terms=[("DYN", 41.0)],
    docs=[SimpleNamespace(page_content=PAGE1, metadata={"page": 0}),
          SimpleNamespace(page_content=DAMAGED, metadata={"page": 1})])
built = DocumentContext.from_extraction(ocr_extraction)
check("OCR page metadata (0-based) becomes 1-based page numbers", sorted(built.pages), [1, 2])
check("the text is the pages joined with blank lines, as app.py's doc_text is",
      built.text, PAGE1 + "\n\n" + DAMAGED)
check("damage is detected from the extraction", len(built.damaged) >= 1, True)
check("the caveat names the OCR-flagged token and the damaged passage",
      ("DYN" in built.caveat, "החל" in built.caveat), (True, True))
check("the damaged passage maps to its page", built.page_containing(built.damaged[0].snippet), 2)

text_layer = SimpleNamespace(used_ocr=False, text=PAGE1, low_confidence_terms=[],
                             docs=[SimpleNamespace(page_content=PAGE1, metadata={"source": "x.pdf"})])
check("a text-layer extraction records NO page map",
      DocumentContext.from_extraction(text_layer).pages, {})
check("...and a clean document has an empty caveat",
      DocumentContext.from_extraction(text_layer).caveat, "")

# --- excerpt_around ----------------------------------------------------------
check("excerpt windows are whitespace-normalised and bounded",
      excerpt_around("a  b\nc DYN d   e", "DYN", width=4), ["b c DYN d e"])
check("no occurrence gives no excerpts", excerpt_around("abc", "zzz"), [])

for note in skipped:
    print(f"[SKIP] {note}")
print(f"\n{sum(results)}/{len(results)} passed" + (f", {len(skipped)} skipped" if skipped else ""))
sys.exit(0 if all(results) else 1)
