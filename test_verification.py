"""Tests for the answer-verification layer.

These checks are what the project offers in place of trusting the model, so
"do the checks themselves work" is a fair question and this is the answer to it.
Every case here is drawn from a failure that actually occurred — each one is a
regression test for a specific wrong conclusion, not a hypothetical.

Runs on the standard library alone: python3 test_verification.py
"""
import sys

from verification import (verify_quotes, verify_answer_facts, _canon_date,
                          find_ocr_garbage_echoes, answer_rests_on_damage)

results = []


def check(name, actual, expected):
    ok = actual == expected
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"         expected {expected!r}\n         got      {actual!r}")


DOC = (
    "סך של 880,000 ש\"ח ישולם לשני יחידי המוכר עד ליום 15.5.2026 כנגד קבלת כספי המשכנתא.\n"
    "המוכר מתחייב לפנות את הדירה ולמסור אותה לידי הקונה ביום 1.07.2026.\n"
    "הדירה נמכרת במצבה כמו שהוא (\"As-Is\").\n"
    "לצורך ביצוע עבדות שיפוץ בלבד החל DYN 6,. מובהר כי כניסת DPN לנכס WAN במועד האמור."
)

# --- verify_quotes ---------------------------------------------------------
check("verbatim quote is found",
      verify_quotes(["המוכר מתחייב לפנות את הדירה"], DOC), [("המוכר מתחייב לפנות את הדירה", True)])
# Extraction and OCR break lines where a faithful quote will not, so the
# comparison normalises whitespace. Without this the badge fires on good quotes.
check("quote spanning a line break is still found",
      verify_quotes(["המשכנתא.\n   המוכר מתחייב"], DOC), [("המשכנתא.\n   המוכר מתחייב", True)])
check("invented quote is flagged",
      verify_quotes(["הקונה רשאי לבטל את ההסכם בכל עת"], DOC),
      [("הקונה רשאי לבטל את ההסכם בכל עת", False)])
check("empty quote is not silently 'found'", verify_quotes([""], DOC), [("", False)])

# --- _canon_date -----------------------------------------------------------
# The first version compared separator-stripped strings and called 1/7/2026 a
# mismatch for 1.07.2026 — a check that flags CORRECT values teaches the reader
# to dismiss every warning it raises.
check("zero-padding does not change the date", _canon_date("1/7/2026"), _canon_date("1.07.2026"))
check("separator does not change the date", _canon_date("15-5-2026"), _canon_date("15.5.2026"))
check("ISO order is recognised", _canon_date("2026-07-01"), (1, 7, 2026))
check("impossible month rejected", _canon_date("1.13.2026"), None)
check("non-date rejected", _canon_date("not/a/date"), None)

# --- verify_answer_facts ---------------------------------------------------
check("a date present in the document passes",
      verify_answer_facts("המסירה ביום 1.07.2026", DOC), [("1.07.2026", True)])
check("the same date written differently still passes",
      verify_answer_facts("המסירה ביום 1/7/2026", DOC), [("1/7/2026", True)])
check("a fabricated date is flagged",
      verify_answer_facts("הכניסה לשיפוץ ביום 6.6.2026", DOC), [("6.6.2026", False)])
check("an amount present in the document passes",
      verify_answer_facts("סכום של 880,000 ש\"ח", DOC), [("880,000", True)])
check("a fabricated amount is flagged",
      verify_answer_facts("סכום של 920,000 ש\"ח", DOC), [("920,000", False)])
check("a clean answer with no figures has nothing to check",
      verify_answer_facts("המסמך אינו מציג את התאריך באופן קריא.", DOC), [])

# --- find_ocr_garbage_echoes ----------------------------------------------
check("garbage copied from the document is caught",
      find_ocr_garbage_echoes("הכניסה החל DYN 6, וכן כניסת DPN", DOC), ["DYN", "DPN"])
check("a clean Hebrew answer is silent",
      find_ocr_garbage_echoes("המסמך אינו מציג את התאריך באופן קריא.", DOC), [])
# "As-Is" is real contract language, not OCR damage. Flagging it would make the
# banner meaningless on every property contract in existence.
check("legitimate Latin in the contract is allowed",
      find_ocr_garbage_echoes("הדירה נמכרת כמו שהוא (As-Is).", DOC), [])
check("Latin not present in the source is not an echo",
      find_ocr_garbage_echoes("This mentions ZQX which the document never did.", DOC), [])
# The signal is "Latin where Hebrew belongs", so it must switch itself off when
# Latin legitimately belongs — otherwise every English document is all-garbage.
check("the check disables itself on a non-Hebrew document",
      find_ocr_garbage_echoes("The term DYN appears", "The term DYN appears in an English contract"), [])

# --- answer_rests_on_damage ------------------------------------------------
DAMAGED = ["לצורך ביצוע עבדות שיפוץ בלבד החל DYN 6,. מובהר כי כניסת DPN לנכס"]
check("an answer reproducing a damaged passage is traced to it",
      len(answer_rests_on_damage("הקונה יכול להיכנס לנכס לצורך ביצוע עבדות שיפוץ בלבד החל מיום 6.", DAMAGED)), 1)
check("an answer that declines to guess is not flagged",
      answer_rests_on_damage("המסמך אינו מציג את התאריך באופן קריא.", DAMAGED), [])
check("an unrelated answer is not flagged",
      answer_rests_on_damage("התמורה הכוללת עומדת על 2,980,000 ש\"ח.", DAMAGED), [])
# A few shared words are not provenance; the overlap has to be substantial.
check("a short incidental overlap does not count",
      answer_rests_on_damage("ביצוע עבדות", DAMAGED), [])
check("no damaged passages means nothing to rest on",
      answer_rests_on_damage("כל תשובה שהיא", []), [])

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
