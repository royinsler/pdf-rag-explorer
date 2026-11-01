"""Tests for eval_caveat.py: the comparison, on scripted answers. No API key.

    python3 test_eval_caveat.py
"""
import os
import sys

import dotenv

import eval_caveat
from answering import GroundedAnswer
from eval_agent import DAMAGED_CASES, DAMAGED_QUESTIONS

results = []


def check(name, actual, expected):
    ok = actual == expected
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"         expected {expected!r}\n         got      {actual!r}")


CASES = {c["question"]: c for c in list(DAMAGED_QUESTIONS) + [q for d in DAMAGED_CASES for q in d["questions"]]}


def answer_of(case):
    return (case.get("expect_any") or [case["expect"]])[0]


def model(question, doc_text, history, caveat, notes, config):
    """Over-refuses answerable questions when the caveat is present; invents a date without it."""
    case = CASES[question]
    if case["kind"] == "damaged":
        return GroundedAnswer(answer="הערך לא קריא" if caveat else "ביום 8.8.2027", quotes=[])
    if caveat:
        return GroundedAnswer(answer="המסמך לא מציג זאת", quotes=[])
    return GroundedAnswer(answer=f"The answer is {answer_of(case)}.", quotes=[])


lines = []
code = eval_caveat.main([], answerer=model, out=lines.append)
text = "\n".join(lines)
n_damaged = sum(1 for c in CASES.values() if c["kind"] == "damaged")
n_answerable = len(CASES) - n_damaged
check("it runs and says how many calls it will make", (code, f"{len(CASES) * 2} model calls." in text), (0, True))
check("it counts answerable questions the caveat made the model fail",
      f"answerable questions the caveat made the model fail: {n_answerable}" in text, True)
check("it counts damaged questions where dropping the caveat made the model fabricate",
      f"made it fabricate or echo: {n_damaged}" in text, True)

rows = eval_caveat.evaluate(model, __import__("eval_agent").build_suites()["damaged"], log=lambda *_: None)
check("with the caveat, damaged questions are honest",
      {r["with_caveat"]["outcome"] for r in rows if r["kind"] == "damaged"}, {"honest"})
check("without it, they are wrong", {r["without_caveat"]["outcome"] for r in rows if r["kind"] == "damaged"}, {"wrong"})

def neutral(question, doc_text, history, caveat, notes, config):
    case = CASES[question]
    if case["kind"] == "damaged":
        return GroundedAnswer(answer="הערך לא קריא", quotes=[])
    return GroundedAnswer(answer=f"The answer is {answer_of(case)}.", quotes=[])

lines = []
eval_caveat.main([], answerer=neutral, out=lines.append)
text = "\n".join(lines)
check("a caveat that changes nothing is reported as changing nothing",
      ("made the model fail: 0" in text, "fabricate or echo: 0" in text), (True, True))

saved = os.environ.pop("OPENAI_API_KEY", None)
real, dotenv.load_dotenv = dotenv.load_dotenv, lambda *a, **k: False
try:
    lines = []
    code = eval_caveat.main([], out=lines.append)
    check("with no API key it says so and exits 2", (code, any("OPENAI_API_KEY" in l for l in lines)), (2, True))
finally:
    dotenv.load_dotenv = real
    if saved is not None:
        os.environ["OPENAI_API_KEY"] = saved

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
