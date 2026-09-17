"""Why did a quote fail verification?

verify_quotes answers found/not-found. When a quote LOOKS like it is in the
document and the check says it is not, the useful question is which character
the two stop agreeing on — a false negative here is as damaging as a false
positive, since it teaches the reader to ignore the badge.

    python quote_probe.py "<the quoted passage>" [test.pdf]
"""
import re
import sys
from difflib import SequenceMatcher

from extraction import extract_document

quote = sys.argv[1]
path = sys.argv[2] if len(sys.argv) > 2 else "test.pdf"
text = extract_document(open(path, "rb").read()).text

norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
haystack, needle = norm(text), norm(quote)
print(f"exact substring match: {needle in haystack}")

# Locate the best-matching window in the document and show exactly where the
# two diverge, character by character.
match = SequenceMatcher(None, haystack, needle).find_longest_match(0, len(haystack), 0, len(needle))
start = max(0, match.a - match.b - 20)
window = haystack[start:start + len(needle) + 40]
print(f"\nlongest common run: {match.size} of {len(needle)} chars")
print(f"\ndocument window:\n  {window!r}")
print(f"\nquote:\n  {needle!r}")

sm = SequenceMatcher(None, window, needle)
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == "equal":
        continue
    print(f"\n  {tag}:")
    print(f"    document: {window[i1:i2]!r} {[hex(ord(c)) for c in window[i1:i2]][:12]}")
    print(f"    quote   : {needle[j1:j2]!r} {[hex(ord(c)) for c in needle[j1:j2]][:12]}")
