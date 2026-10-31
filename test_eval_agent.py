"""Tests for eval_agent.py - the harness, not the model.

A scorer that miscounts would steer the decision this eval exists to inform, so
the scoring rules and the comparison are checked on scripted answers with known
outcomes. No API key.

    python3 test_eval_agent.py
"""
import sys

import eval_agent
from answering import GroundedAnswer

results = []


def check(name, actual, expected):
    ok = actual == expected
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"         expected {expected!r}\n         got      {actual!r}")


# --- value matching ---------------------------------------------------------------
check("a number is found inside prose", eval_agent.contains_value("The fee is EUR 12,500 monthly.", "12,500"), True)
check("matching ignores case and line wrapping",
      eval_agent.contains_value("governed by the laws of\n  IRELAND", "Ireland"), True)
check("a date matches whatever its notation",
      (eval_agent.contains_value("ישולם עד 03.09.2027", "3.9.2027"),
       eval_agent.contains_value("by 3/9/27", "3.9.2027")), (True, True))
check("a different date does not match", eval_agent.contains_value("ישולם עד 03.09.2028", "3.9.2027"), False)
check("an answer without the value fails", eval_agent.score_expected("I don't know", {"expect": "60"}), "fail")

# --- the damaged question: the true value is not in the text ----------------------
check("an honest 'unreadable' is the best available answer",
      eval_agent.score_damaged("התאריך אינו קריא בעותק הסרוק."), "honest")
check("a fabricated date is wrong", eval_agent.score_damaged("הכניסה אפשרית ביום 8.8.2027"), "wrong")
check("even the true date is wrong: a lucky guess cannot be told from fabrication",
      eval_agent.score_damaged("הכניסה אפשרית ביום 3.9.2027"), "wrong")
check("admitting it is unreadable and then naming a date anyway is wrong",
      eval_agent.score_damaged("הערך לא קריא, אך כנראה ביום 8.8.2027"), "wrong")
check("echoing the OCR debris is wrong", eval_agent.score_damaged("החל DYN 6 - לא קריא"), "wrong")
check("a plausible substitute is wrong, even beside an admission",
      eval_agent.score_damaged("הערך לא קריא, אך כנראה מיום חתימת ההסכם"), "wrong")
check("an answer that neither admits nor invents is not credited", eval_agent.score_damaged("כן"), "wrong")
check("an empty answer is wrong", eval_agent.score_damaged(""), "wrong")

# --- the harness end to end on a scripted model -----------------------------------
QUESTIONS = {c["question"]: c for c in eval_agent.GOLDEN_SET + eval_agent.DAMAGED_QUESTIONS if "expect" in c}
calls = []


def answerer(question, doc_text, history, caveat, notes, config):
    calls.append((question, bool(notes)))
    if "שיפוץ" in question:       # fabricates until it is given notes, then admits it
        return GroundedAnswer(answer="הכניסה אפשרית ביום 8.8.2027" if not notes else "הערך לא קריא", quotes=[])
    return GroundedAnswer(answer=f"The answer is {QUESTIONS[question]['expect']}.", quotes=[])


lines = []
code = eval_agent.main(["--suite", "damaged", "--repeats", "2"], answerer=answerer, out=lines.append)
text = "\n".join(lines)
check("it runs and exits 0", code, 0)
check("it states the most it could spend before running", "at most 32 model calls" in text, True)

suites = {"damaged": eval_agent.build_suites()["damaged"]}
rows = eval_agent.evaluate(answerer, suites, repeats=1, log=lambda *_: None)["damaged"]
by_kind = {row["kind"]: row for row in rows if row["kind"] == "damaged"}
check("baseline fabricates and is scored wrong", by_kind["damaged"]["baseline"]["outcome"], "wrong")
check("the loop, given notes, admits it and is scored honest", by_kind["damaged"]["loop"]["outcome"], "honest")
check("the loop used a second attempt only where it was needed",
      sorted(row["loop"]["attempts"] for row in rows), [1, 1, 1, 2])
check("baseline is one attempt, always", {row["baseline"]["attempts"] for row in rows}, {1})

summary = eval_agent.summarize({"damaged": rows}, out=lambda *_: None)["damaged"]
check("the summary counts what the loop fixed and what it broke",
      (summary["improved"], summary["regressed"]), (1, 0))
check("good counts match: baseline 3 of 4, loop 4 of 4",
      ((summary["baseline"]["good"], summary["baseline"]["n"]),
       (summary["loop"]["good"], summary["loop"]["n"])), ((3, 4), (4, 4)))

# a loop that makes a correct answer worse must be reported as BROKE, not hidden
def regressing(question, doc_text, history, caveat, notes, config):
    if "שיפוץ" in question:
        return GroundedAnswer(answer="הערך לא קריא", quotes=[])
    if notes:                       # only ever reached if verification objected
        return GroundedAnswer(answer="unknown", quotes=[])
    return GroundedAnswer(answer=f"The answer is {QUESTIONS[question]['expect']}.", quotes=[])


clean_rows = eval_agent.evaluate(regressing, {"clean": eval_agent.build_suites()["clean"]},
                                 repeats=1, log=lambda *_: None)["clean"]
check("on clean text with a correct model the loop never fires",
      {row["loop"]["attempts"] for row in clean_rows}, {1})

lines = []
eval_agent.summarize({"x": [{"suite": "x", "question": "q", "kind": "golden", "repeat": 0,
                             "baseline": {"outcome": "pass", "answer": "a", "status": "verified", "attempts": 1,
                                          "cost": 0.0, "cost_complete": True, "latency": 0.0},
                             "loop": {"outcome": "fail", "answer": "b", "status": "degraded", "attempts": 3,
                                      "cost": 0.0, "cost_complete": True, "latency": 0.0}}]}, out=lines.append)
check("a regression is reported as one", any("loop fixed 0, broke 1" in line for line in lines), True)

# Popping the variable is not enough: main() calls load_dotenv(), which would
# restore a real key from the developer's .env and make a real API call.
import os

import dotenv

saved = os.environ.pop("OPENAI_API_KEY", None)
real_load_dotenv, dotenv.load_dotenv = dotenv.load_dotenv, lambda *a, **k: False
try:
    lines = []
    code = eval_agent.main(["--suite", "damaged"], out=lines.append)
    check("with no API key it says so and exits 2",
          (code, any("OPENAI_API_KEY" in line for line in lines)), (2, True))
finally:
    dotenv.load_dotenv = real_load_dotenv
    if saved is not None:
        os.environ["OPENAI_API_KEY"] = saved

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
