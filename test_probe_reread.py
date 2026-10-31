"""Tests for probe_reread.py - the experiment, not the agent.

A diagnostic that reports "recovered" when nothing was, or "none" when something
was, is worse than none: it would steer a design decision the wrong way. So the
report is checked against pages built here with a known answer. Needs Tesseract
with 'eng' (skipped, loudly, without it); no API key.

    python3 test_probe_reread.py
"""
import os
import sys
import tempfile

import probe_reread
from agent_tools import DocumentContext
from damaged_values import DamagedValue

results = []
skipped = []


def check(name, actual, expected):
    ok = actual == expected
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"         expected {expected!r}\n         got      {actual!r}")


def tesseract_ready():
    try:
        import pytesseract
        return "eng" in pytesseract.get_languages()
    except Exception:
        return False


# --- pure helpers (no OCR) -------------------------------------------------------
check("dates are found across separator styles",
      probe_reread.dates_in("on 4.11.2027 and 2027-12-01 and 5/6/28"),
      {(4, 11, 2027), (1, 12, 2027), (5, 6, 2028)})
check("no dates is an empty set", probe_reread.dates_in("clause 6, section 7"), set())
check("dates print zero-padded, so 12.03.2028 is not shown as 12.3.2028",
      probe_reread.fmt((12, 3, 2028)), "12.03.2028")

check("a date to look for is parsed whatever the notation",
      (probe_reread.parse_date("11.9.27"), probe_reread.parse_date("11/09/2027")),
      ((11, 9, 2027), (11, 9, 2027)))
check("something that is not a date is None", probe_reread.parse_date("tomorrow"), None)
check("configurations are labelled with only the options in use",
      (probe_reread.label(400), probe_reread.label(300, 6), probe_reread.label(300, 11, 150)),
      ("400 DPI", "300 DPI, psm 6", "300 DPI, psm 11, threshold 150"))

if not tesseract_ready():
    skipped.append("everything that re-reads a page (Tesseract with 'eng' not available)")
else:
    import fitz

    pdf = fitz.open()
    pdf.new_page().insert_text((72, 120), "The seller shall deliver by 12.03.2028 after payment.", fontsize=14)
    pdf_bytes = pdf.tobytes()
    pdf.close()

    def run(ctx, pages, dpis=(300, 400)):
        lines = []
        probe_reread.probe(ctx, pages, dpis, out=lines.append)
        return "\n".join(lines)

    # --- the value was lost in the first read and is present on the page ----------
    lost = "The seller shall deliver by DYN 6 after payment."
    ctx = DocumentContext(
        text=lost, pages={1: lost}, file_bytes=pdf_bytes, ocr_lang="eng",
        damaged=[DamagedValue(connector="by", snippet="seller shall deliver by DYN 6 after",
                              reasons=["Latin characters where Hebrew is expected", "a bare number"])])
    report = run(ctx, [1])
    check("a date missing from the original but on the page is reported as recovered",
          "dates only in the re-read:   12.03.2028" in report, True)
    check("...and is not also reported as lost", "dates only in the original:  none" in report, True)
    check("each resolution asked for is reported", ("300 DPI" in report, "400 DPI" in report), (True, True))
    check("the text around the damaged connector is shown", "near 'by'" in report, True)

    # --- the value was already read correctly: nothing is recovered ----------------
    fine = "The seller shall deliver by 12.03.2028 after payment."
    report = run(DocumentContext(text=fine, pages={1: fine}, file_bytes=pdf_bytes, ocr_lang="eng"), [1])
    check("a date present in both reads is not claimed as recovered",
          "dates only in the re-read:   none" in report, True)

    # --- a date the original has and the re-read does not is reported as lost -----
    phantom = "Delivery by 01.01.2030 and by 12.03.2028 as agreed."
    report = run(DocumentContext(text=phantom, pages={1: phantom}, file_bytes=pdf_bytes, ocr_lang="eng"), [1])
    check("a date in the original but not on the page is reported as lost",
          "dates only in the original:  01.01.2030" in report, True)

    # --- --expect: the question the script exists to answer ----------------------
    lines = []
    found = probe_reread.probe(
        DocumentContext(text=lost, pages={1: lost}, file_bytes=pdf_bytes, ocr_lang="eng"),
        [1], (300, 400), out=lines.append, expect=(12, 3, 2028))
    text = "\n".join(lines)
    check("--expect reports each configuration that found the date",
          (found, "12.03.2028: FOUND" in text), ({1: ["300 DPI", "400 DPI"]}, True))
    check("...and the closing line counts them", "was found by 2 configuration(s)" in text, True)

    lines = []
    found = probe_reread.probe(
        DocumentContext(text=lost, pages={1: lost}, file_bytes=pdf_bytes, ocr_lang="eng"),
        [1], (300,), out=lines.append, expect=(1, 1, 2031))
    check("a date that is not on the page is reported as not found, with no configuration credited",
          (found, "01.01.2031: not found" in "\n".join(lines)), ({1: []}, True))

    # --- psm and threshold multiply the configurations ------------------------------
    lines = []
    probe_reread.probe(DocumentContext(text=lost, pages={1: lost}, file_bytes=pdf_bytes, ocr_lang="eng"),
                       [1], (300,), out=lines.append, psms=(6, 11), thresholds=(None, 150))
    text = "\n".join(lines)
    check("every dpi x psm x threshold combination is run and labelled",
          [lab in text for lab in ("300 DPI, psm 6:", "300 DPI, psm 6, threshold 150:",
                                   "300 DPI, psm 11:", "300 DPI, psm 11, threshold 150:")], [True] * 4)
    check("the report ends with what two or more configurations agree on",
          "what the agent would accept): 12.03.2028" in text, True)

    # --- failures are lines in the report, not crashes ------------------------------
    report = run(DocumentContext(text="x", pages={1: "x"}, file_bytes=pdf_bytes, ocr_lang="eng"), [7])
    check("a page that does not exist is reported as a failure, not an exception",
          "failed" in report, True)

    # --- main(): the paths a user actually takes ------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "contract.pdf")
        with open(path, "wb") as handle:
            handle.write(pdf_bytes)

        def cli(argv):
            lines = []
            code = probe_reread.main(argv, out=lines.append)
            return code, "\n".join(lines)

        code, text = cli([path, "--lang", "eng"])
        check("a text-layer PDF has no page map: exit 1, and it says how to get one",
              (code, "--force-ocr" in text), (1, True))

        code, text = cli([path, "--lang", "eng", "--force-ocr"])
        check("with --force-ocr but nothing damaged it asks for --page",
              (code, "--page" in text), (1, True))

        code, text = cli([path, "--lang", "eng", "--force-ocr", "--page", "1", "--dpi", "300"])
        check("with --page it probes and exits 0", (code, "=== page 1" in text), (0, True))
        check("...naming the date the original extraction held, and claiming no recovery",
      ("dates there: 12.03.2028" in text, "dates only in the re-read:   none" in text), (True, True))

        code, text = cli([path, "--lang", "eng", "--force-ocr", "--page", "1", "--dpi", "300",
                          "--expect", "12.3.28"])
        check("--expect through the CLI finds the date and says so", (code, "FOUND" in text), (0, True))
        code, text = cli([path, "--expect", "soon"])
        check("an unreadable --expect exits 2 before doing any OCR", (code, "not a date" in text), (2, True))

        junk = os.path.join(tmp, "not.pdf")
        with open(junk, "wb") as handle:
            handle.write(b"not a pdf at all " * 10)
        code, text = cli([junk, "--lang", "eng"])
        check("an unreadable file exits 2 with the reason", (code, "Could not use this PDF" in text), (2, True))

for note in skipped:
    print(f"[SKIP] {note}")
print(f"\n{sum(results)}/{len(results)} passed" + (f", {len(skipped)} skipped" if skipped else ""))
sys.exit(0 if all(results) else 1)
