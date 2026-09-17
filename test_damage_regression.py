"""The §8 failure, as a regression test rather than a story.

Three rounds of constraining the model produced three different wrong answers
to the same question, each more plausible than the last. This asserts that
every one of them is still caught, by at least one layer that does not depend
on the model cooperating — so a future change that quietly removes a layer
fails here instead of in front of an interviewer.

The point of the file is the LAST case. Its answer is fluent Hebrew with no
garbled characters and no suspicious number; nothing about what it says is
wrong-looking. Only provenance catches it.

    python test_damage_regression.py
"""
import sys

from damaged_values import detect_damaged_values
from eval_fixtures import (DAMAGED_EXTRACTION, DAMAGED_LOW_CONFIDENCE_TERMS,
                           DAMAGE_FAILURE_MODES)
from verification import (verify_answer_facts, find_ocr_garbage_echoes,
                          answer_rests_on_damage)

results = []


def check(name, condition, detail=""):
    results.append(bool(condition))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


damaged = detect_damaged_values(DAMAGED_EXTRACTION, DAMAGED_LOW_CONFIDENCE_TERMS)
snippets = [d.snippet for d in damaged]

check("the damaged clause is detected", any("DYN" in s for s in snippets),
      f"{len(damaged)} passage(s) flagged")
# Correct clauses in the same text must stay clean, or the flag means nothing.
check("the delivery date clause is not flagged", not any("1.07.2026" in s for s in snippets))
check("the payment date clause is not flagged", not any("15.5.2026" in s for s in snippets))

print()
for label, answer in DAMAGE_FAILURE_MODES:
    facts = verify_answer_facts(answer, DAMAGED_EXTRACTION)
    echoes = find_ocr_garbage_echoes(answer, DAMAGED_EXTRACTION)
    rests_on = answer_rests_on_damage(answer, snippets)
    caught_by = ([f"fact check ({', '.join(v for v, ok in facts if not ok)})"] if any(not ok for _, ok in facts) else []) \
        + ([f"garbage echo ({', '.join(echoes)})"] if echoes else []) \
        + (["provenance"] if rests_on else [])
    check(f"caught: {label}", caught_by, " + ".join(caught_by) or "NOTHING CAUGHT IT")

# A correct answer must trip none of the three, or the warnings become noise
# that users learn to scroll past.
good = "המסמך אינו מציג את מועד הכניסה לשיפוץ באופן קריא; נראה כי חילוץ הטקסט פגע בו."
print()
check("a correct, honest answer trips nothing",
      not any(not ok for _, ok in verify_answer_facts(good, DAMAGED_EXTRACTION))
      and not find_ocr_garbage_echoes(good, DAMAGED_EXTRACTION)
      and not answer_rests_on_damage(good, snippets))

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
