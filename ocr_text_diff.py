"""Check that switching OCR from image_to_string to image_to_data preserved the text.

The confidence work required reading Tesseract's per-word output (image_to_data)
instead of its rendered text (image_to_string), and rebuilding the lines
ourselves from block/paragraph/line ids. That is a rewrite of the step where all
document text enters the system — so it needs verifying rather than assuming,
especially for Hebrew: if Tesseract emits RTL words in visual rather than logical
order, the reconstruction would silently reverse every line.

    python ocr_text_diff.py test.pdf

Prints, per page, the two texts' lengths, a similarity ratio, and the first
line where they diverge.
"""
import sys
from difflib import SequenceMatcher

import fitz
import pytesseract
from PIL import Image

from extraction import OCR_LANG, ocr_extract_documents, strip_bidi

path = sys.argv[1] if len(sys.argv) > 1 else "test.pdf"
file_bytes = open(path, "rb").read()

rebuilt = {d.metadata["page"]: d.page_content for d in ocr_extract_documents(file_bytes)}

pdf = fitz.open(stream=file_bytes, filetype="pdf")
for page_num, page in enumerate(pdf):
    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    baseline = pytesseract.image_to_string(image, lang=OCR_LANG)
    new = rebuilt.get(page_num, "")

    # Compare on whitespace-normalised token streams: the reconstruction drops
    # blank lines and column padding by design, and that difference is not a
    # defect. Word ORDER and word CONTENT are what must match.
    a, b = baseline.split(), new.split()
    raw_ratio = SequenceMatcher(None, a, b).ratio()

    # image_to_string runs each line through Tesseract's bidi layout pass, which
    # inserts LRM/RLM direction hints around Latin runs inside Hebrew lines; the
    # word stream carries no such hints. Those marks are presentation, not
    # content — and the pipeline now strips them — so the verdict is taken from
    # the normalised comparison. The raw number is still printed, because the
    # gap between the two is the thing worth seeing: 0.93 raw with 1.000
    # normalised means "identical text, different direction hints", while a low
    # normalised score would mean real text loss.
    na = [strip_bidi(w) for w in a]
    nb = [strip_bidi(w) for w in b]
    ratio = SequenceMatcher(None, na, nb).ratio()

    print(f"\n=== page {page_num} ===")
    print(f"image_to_string: {len(baseline):>6} chars / {len(a):>5} words")
    print(f"rebuilt        : {len(new):>6} chars / {len(b):>5} words")
    print(f"word-sequence similarity: raw {raw_ratio:.3f} / bidi-normalised {ratio:.3f}"
          + ("  ✅" if ratio > 0.97 else "  ⚠️  INVESTIGATE"))

    # Report differences as repr + codepoints, not as rendered text. The first
    # run of this script printed two lines that looked identical on screen while
    # scoring 0.93 — the differences were invisible characters (RTL marks,
    # unusual spaces), which is exactly what a visual diff cannot show.
    shown = 0
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, na, nb).get_opcodes():
        if tag == "equal":
            continue
        print(f"  --- {tag}: image_to_string[{i1}:{i2}] vs rebuilt[{j1}:{j2}]")
        for word in a[i1:i2][:3]:
            print("      string :", repr(word), [hex(ord(c)) for c in word])
        for word in b[j1:j2][:3]:
            print("      rebuilt:", repr(word), [hex(ord(c)) for c in word])
        shown += 1
        if shown >= 5:
            print("  (further differences suppressed)")
            break
pdf.close()
