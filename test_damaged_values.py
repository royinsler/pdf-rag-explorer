"""Tests for damaged_values.detect_damaged_values.

The cases are taken verbatim from the reference contract — the passage where
extraction lost a date, and passages that look similar to a naive check but are
perfectly correct. The second group is the point of the test: a detector
that flags correct text is worse than no detector, because it trains the reader
to dismiss the warning. This project has already made that mistake once
(verification._canon_date).

Passing these proves only that the detector does what I designed it to do — I
wrote both the rule and the examples. damage_report.py is the real test: it
runs against the whole extracted document, where the false positives live.
"""
import sys

from damaged_values import detect_damaged_values, damage_caveat

# Tesseract's own low-confidence output for this document. DWN and NW" belong
# here for a reason: in the real run they are what made the damaged variant of
# the "aforementioned date" phrase fire. An earlier version of this list left
# them out, so that case failed — the fixture, not the rule, was wrong.
SUSPECT = [("DWNT", 3.0), ("yt", 0.0), ("DION", 24.0), ("DWN", 41.0), ('NW"', 39.0)]

CASES = [
    ("THE FAILURE",
     "הצדדים מסכימים כי לקונים תינתן אפשרות להיכנס לנכס לצורך ביצוע עבדות "
     "שיפוץ בלבד החל DYN 6. מובהר כי כניסת הקונים לנכס במועד האמור",
     True),
    ("relational use, no date intended",
     "לא שילם סכום כאמור בתוך 14 יום מיום שנדרש על ידי הנאמן בכתב לעשות כן.",
     False),
    ("connector followed by a reference, not a value",
     "החל ממועד הנ\"ל הקונה יישא במלוא התשלומים השוטפים של הדירה לרבות דמי ניהול",
     False),
    ("the promise is kept — date present",
     "ישולם לשני יחידי המוכר עד ליום 15.5.2026 כנגד קבלת כספי המשכנתא",
     False),
    ("the promise is kept — other format",
     "מתחייב המוכר לפנות את הדירה ולמסור אותה לידי הקונה ביום 1.07.2026",
     False),
    ("spelled-out date in a preamble",
     "שנערך ונחתם במודיעין ביום 12 לחודש אפריל שנת 2026",
     False),
    ("legitimate Latin, no connector promise nearby",
     "כמו שהוא (\"As-Is\"), בכפוף לנכונות הצהרותיו של המוכר ומילוי חיוביו",
     False),
    ("clause number after an unrelated word",
     "כמפורט להלן לידי הקונה וכנגד מסירת החזקה בדירה לידי הקונה כמפורט בסעיף 6 להלן.",
     False),

    # The same sentence clean and damaged. Running the detector over the whole
    # contract showed this pair matters: on the page the phrase is
    # "החל ממועד הנ״ל" with no date and nothing wrong, but OCR rendered הנ״ל as
    # NW" and the rule fires. Both behaviours are correct — the difference is
    # not the grammar, it is the damage — and keeping both here stops a future
    # "fix" to the false positive from silently disabling the true one.
    ("aforementioned-date phrase, clean",
     "החל ממועד הנ\"ל הקונה יישא במלוא התשלומים השוטפים של הדירה",
     False),
    ("the same phrase after OCR damaged it",
     "החל ממועד NW\" DWN הקונה יישא במלוא התשלומים השוטפים של הדירה",
     True),
]

failures = 0
for name, text, should_fire in CASES:
    hits = detect_damaged_values(text, low_confidence_terms=SUSPECT)
    ok = bool(hits) == should_fire
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: "
          f"expected {'a flag' if should_fire else 'no flag'}, got {len(hits)}")
    for hit in hits:
        print(f"         → {hit.describe()}")

print("\ncaveat passed to the model:")
print(damage_caveat(detect_damaged_values(CASES[0][1], low_confidence_terms=SUSPECT)))
sys.exit(1 if failures else 0)
