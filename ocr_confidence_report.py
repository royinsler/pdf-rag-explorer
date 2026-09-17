"""What did OCR actually know about this document, and does the threshold catch the damage?

The confidence feature rests on an assumption worth testing directly rather than
inferring from the app's behaviour: that where OCR destroyed a value, Tesseract
itself scored that token poorly. If the engine was CONFIDENTLY wrong, no
threshold can catch it, and the feature has to be described as partial.

    python ocr_confidence_report.py [file.pdf] [TOKEN ...]

This runs its own image_to_data pass rather than reading extraction.py's
metadata, because that metadata deliberately keeps only the tokens BELOW the
threshold — and the question here is precisely about a token above it.
"""
import sys
from collections import defaultdict

import fitz
import pytesseract
from PIL import Image

from extraction import OCR_LANG, OCR_MIN_CONFIDENCE, extract_document, strip_bidi

path = sys.argv[1] if len(sys.argv) > 1 else "test.pdf"
probes = [t.lower() for t in sys.argv[2:]] or ["dyn"]
file_bytes = open(path, "rb").read()

result = extract_document(file_bytes, force_ocr=True)
print(f"file={path}  used_ocr={result.used_ocr}  mean confidence={result.ocr_confidence}")
print(f"threshold={OCR_MIN_CONFIDENCE}  flagged tokens={len(result.low_confidence_terms)}\n")

print("worst 15 tokens:")
for token, score in result.low_confidence_terms[:15]:
    print(f"  {score:>5.1f}  {token!r}")

# Full token -> score map, and the line each occurrence sits in.
scores = defaultdict(list)
contexts = defaultdict(list)
pdf = fitz.open(stream=file_bytes, filetype="pdf")
for page_num, page in enumerate(pdf):
    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    data = pytesseract.image_to_data(image, lang=OCR_LANG,
                                     output_type=pytesseract.Output.DICT)
    lines = defaultdict(list)
    for i, raw in enumerate(data["text"]):
        token = strip_bidi(raw).strip()
        if not token:
            continue
        try:
            score = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if score < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines[key].append((token, score))
        scores[token.lower()].append(score)
    for key, words in lines.items():
        for token, _ in words:
            contexts[token.lower()].append((page_num, words))
pdf.close()

print("\nprobes:")
for probe in probes:
    hits = sorted(tok for tok in scores if probe in tok)
    if not hits:
        print(f"  {probe!r}: not present in the OCR output at all")
        continue
    for tok in hits:
        vals = scores[tok]
        verdict = ("BELOW threshold — flagged" if min(vals) < OCR_MIN_CONFIDENCE
                   else "ABOVE threshold — Tesseract was CONFIDENT")
        print(f"  {tok!r}: score(s) {vals}  → {verdict}")
        page_num, words = contexts[tok][0]
        print(f"    page {page_num}, the line it sits in (score under each word):")
        print("      " + "  ".join(w for w, _ in words))
        print("      " + "  ".join(f"{s:.0f}".rjust(len(w)) for w, s in words))
