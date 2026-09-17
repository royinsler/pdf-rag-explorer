"""
Adversarial input testing for the extraction pipeline.

NOT a load test. This app is a single-user local tool, so throughput numbers
would measure Streamlit's execution model and OpenAI's rate limits rather than
anything we built. What actually matters here is robustness to hostile input,
because the pipeline's whole job is ingesting messy real-world documents — and
the one serious bug found so far (a Hebrew contract whose font carried no
usable ToUnicode CMap) was discovered by accident rather than by looking.

Every case is classified by HOW it fails, because the failure mode matters more
than the failure:

  OK      - produced usable text, correctly
  LOUD    - refused the document and said why (the good failure)
  SILENT  - returned text that is empty or garbage WITHOUT flagging it
            (the dangerous one: the model then answers from nonsense)
  CRASH   - raised an unhandled exception out of the pipeline

Corpus is generated synthetically, so it commits to the repo and reproduces
anywhere.

    .venv/bin/python stress_inputs.py
"""
import io
import sys
import time
import traceback

import fitz  # PyMuPDF

from extraction import extract_document


# --- corpus -------------------------------------------------------------

def _simple_pdf(text: str, pages: int = 1) -> bytes:
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        if text:
            page.insert_textbox(fitz.Rect(50, 50, 550, 750), text, fontsize=11)
    out = doc.tobytes()
    doc.close()
    return out


def make_zero_bytes() -> bytes:
    return b""


def make_not_a_pdf() -> bytes:
    return b"This is a plain text file that someone renamed to .pdf\n" * 20


def make_truncated() -> bytes:
    full = _simple_pdf("The quick brown fox jumps over the lazy dog. " * 40)
    return full[: len(full) // 2]          # chop mid-stream


def make_encrypted() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 550, 750),
                        "Confidential. " * 60, fontsize=11)
    out = doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256,
                      owner_pw="owner", user_pw="user")
    doc.close()
    return out


def make_blank_pages() -> bytes:
    return _simple_pdf("", pages=3)


def make_whitespace_only() -> bytes:
    return _simple_pdf("   \n\n   \t   \n" * 30)


def make_no_whitespace() -> bytes:
    """A wall of characters with no word boundaries — recursive splitters look
    for separators and have to fall back to splitting mid-token.

    NOTE: the first version of this used insert_textbox("A" * 6000), which
    silently inserted NOTHING, because unwrappable text does not fit a box and
    PyMuPDF just declines. The fixture was a blank page claiming to be a
    stress case - caught because its byte size was identical to the
    deliberately-blank one. insert_text writes at a fixed position instead."""
    doc = fitz.open()
    page = doc.new_page()
    y = 60
    for _ in range(40):
        page.insert_text((45, y), "A" * 95, fontsize=9)
        y += 15
    out = doc.tobytes()
    doc.close()
    return out


def make_scanned_image_only() -> bytes:
    """Text rendered as pixels with no text layer — the genuine scanned-PDF
    case. Exercises the OCR path, which until now had only ever been tested
    against one real Hebrew contract."""
    src = fitz.open()
    page = src.new_page()
    page.insert_textbox(
        fitz.Rect(50, 50, 550, 400),
        "PAYMENT TERMS\n\nThe Client shall pay a fixed fee of EUR 12,500 "
        "per month. Invoices are due within 30 days of the invoice date.",
        fontsize=16)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    src.close()

    out_doc = fitz.open()
    out_page = out_doc.new_page(width=pix.width / 2, height=pix.height / 2)
    out_page.insert_image(out_page.rect, pixmap=pix)
    out = out_doc.tobytes()
    out_doc.close()
    return out


def make_large() -> bytes:
    return _simple_pdf("Clause text for scale testing. " * 60, pages=120)


CASES = [
    ("zero_bytes",         make_zero_bytes,        "LOUD",   "empty upload"),
    ("not_a_pdf",          make_not_a_pdf,         "LOUD",   "text file renamed .pdf"),
    ("truncated",          make_truncated,         "LOUD",   "valid PDF cut in half"),
    ("encrypted",          make_encrypted,         "LOUD",   "password-protected"),
    ("blank_pages",        make_blank_pages,       "LOUD",   "3 pages, no content"),
    ("whitespace_only",    make_whitespace_only,   "LOUD",   "text layer is only spaces"),
    ("no_whitespace",      make_no_whitespace,     "OK",     "6000 chars, no separators"),
    ("scanned_image_only", make_scanned_image_only,"OK",     "no text layer; must OCR"),
    ("large",              make_large,             "OK",     "120 pages"),
]


# --- runner -------------------------------------------------------------

def classify(result) -> str:
    if result.usable:
        return "OK"
    # Refused, and the app would surface result.reason to the user.
    return "LOUD" if result.reason else "SILENT"


def main():
    print(f"{len(CASES)} adversarial inputs against extraction.extract_document()\n")
    rows = []
    for name, factory, expected, note in CASES:
        data = factory()
        start = time.time()
        try:
            result = extract_document(data)
            outcome = classify(result)
            detail = (result.reason or
                      f"{len(result.text):,} chars"
                      f"{' via OCR' if result.used_ocr else ''}")
        except Exception as e:
            outcome = "CRASH"
            detail = f"{type(e).__name__}: {str(e)[:70]}"
            traceback.print_exc(limit=1, file=sys.stderr)
        elapsed = time.time() - start

        rows.append((name, len(data), expected, outcome, f"{elapsed:.1f}s", note, detail))
        print(f"  {name:<20} {outcome:<7} {elapsed:>6.1f}s  {detail}", flush=True)

    print(f"\n| {'Input':<20} | {'Bytes':>9} | {'Expected':<8} | {'Actual':<7} | {'Time':>6} |")
    print("|" + "-" * 66 + "|")
    for name, size, expected, outcome, elapsed, note, detail in rows:
        flag = "" if outcome == expected else "   <-- MISMATCH"
        print(f"| {name:<20} | {size:>9,} | {expected:<8} | {outcome:<7} | {elapsed:>6} |{flag}")

    crashes = [r for r in rows if r[3] == "CRASH"]
    silent = [r for r in rows if r[3] == "SILENT"]
    print(f"\n{len(crashes)} crash(es), {len(silent)} silent failure(s).")
    if silent:
        print("SILENT failures are the serious ones: bad text reaches the model "
              "with nothing telling the user.")
    if crashes:
        print("CRASH means the exception escapes the pipeline — in the app that "
              "is a Streamlit traceback, not a handled error message.")


if __name__ == "__main__":
    main()
