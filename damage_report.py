"""Run the damaged-value detector against a real document and show every hit.

test_damaged_values.py checks hand-picked passages, which only proves the rule
matches the examples I wrote for it. This runs the rule over the WHOLE extracted
text, where the false positives actually live — every connector in a four-page
contract, not the eight I thought of. Precision is judged here, by reading the
hits, not by the unit test passing.

    python damage_report.py contract.debug.pdf
"""
import sys

from extraction import extract_document
from damaged_values import detect_damaged_values, damage_caveat

path = sys.argv[1] if len(sys.argv) > 1 else "test.pdf"
result = extract_document(open(path, "rb").read(), force_ocr=True)

hits = detect_damaged_values(result.text, result.low_confidence_terms)
print(f"file={path}  chars={len(result.text)}  "
      f"ocr_confidence={result.ocr_confidence}  "
      f"low-confidence tokens={len(result.low_confidence_terms)}")
print(f"suspected damaged values: {len(hits)}\n")

for i, hit in enumerate(hits, 1):
    print(f"{i}. connector {hit.connector!r} — {', '.join(hit.reasons)}")
    print(f"   {hit.snippet}\n")

if hits:
    print("--- caveat the model would receive ---")
    print(damage_caveat(hits))
