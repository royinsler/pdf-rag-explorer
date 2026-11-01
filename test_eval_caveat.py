"""Tests for eval_caveat.py and the candidate caveat: the comparison, on scripted answers.
No API key.

    python3 test_eval_caveat.py
"""
import os
import sys

import dotenv

import eval_caveat
from answering import GroundedAnswer
from damaged_values import DamagedValue, damage_caveat, damage_caveat_narrow
from eval_agent import DAMAGED_CASES, DAMAGED_QUESTIONS, build_suites, score_relational

results = []


def check(name, actual, expected):
    ok = actual == expected
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"         expected {expected!r}\n         got      {actual!r}")


# --- the candidate wording --------------------------------------------------------
dv = [DamagedValue(connector="החל", snippet="החל DYN 6,", reasons=["x"])]
check("no damage, no caveat (both wordings)", (damage_caveat([]), damage_caveat_narrow([])), ("", ""))
check("the narrow wording names the damaged passage and restricts the refusal",
      ("החל DYN 6," in damage_caveat_narrow(dv), "ONLY when" in damage_caveat_narrow(dv),
       "do not repeat the broken characters" in damage_caveat_narrow(dv)), (True, True, True))
check("the current wording is unchanged: it still tells the model to refuse whenever a question depends on it",
      "If a question depends on one of these values" in damage_caveat(dv), True)

# --- a relational answer must not carry debris ------------------------------------
case = {"expect_any": ["החמישה עשר", "15"]}
check("the right value without debris passes", score_relational("מהיום החמישה עשר לאיחור", case), "pass")
check("the right value WITH debris copied in fails", score_relational("מהיום החמישה עשר DWNT לאיחור", case), "fail")
check("the wrong part of the clause fails", score_relational("למסירה בפועל", case), "fail")

# --- the experiment on a scripted model -------------------------------------------
CASES = {c["question"]: c for c in list(DAMAGED_QUESTIONS) + [q for d in DAMAGED_CASES for q in d["questions"]]}


def right(case):
    return (case.get("expect_any") or [case["expect"]])[0]


def model(question, doc_text, history, caveat, notes, config):
    """none: invents. current: admits, but also over-refuses answerable ones. narrow: admits, answers the rest."""
    case = CASES[question]
    if case["kind"] == "damaged":
        return GroundedAnswer(answer="ביום 8.8.2027" if not caveat else "הערך לא קריא", quotes=[])
    if "If a question depends on one of these values" in caveat:
        return GroundedAnswer(answer="המסמך לא מציג זאת", quotes=[])
    return GroundedAnswer(answer=f"The answer is {right(case)}.", quotes=[])


rows = eval_caveat.evaluate(model, build_suites()["damaged"], repeats=2, log=lambda *_: None)
n_damaged = sum(1 for c in CASES.values() if c["kind"] == "damaged")
n_answerable = len(CASES) - n_damaged
check("none: every damaged question is invented",
      eval_caveat.tally(rows, "damaged", "none"), {"wrong": n_damaged * 2})
check("current and narrow admit it on every damaged question",
      (eval_caveat.tally(rows, "damaged", "current"), eval_caveat.tally(rows, "damaged", "narrow")),
      ({"honest": n_damaged * 2}, {"honest": n_damaged * 2}))
check("current over-refuses every answerable question; narrow and none do not",
      (eval_caveat.tally(rows, "control", "current"), eval_caveat.tally(rows, "control", "narrow"),
       eval_caveat.tally(rows, "relational", "current"), eval_caveat.tally(rows, "relational", "narrow")),
      ({"fail": 16}, {"pass": 16}, {"fail": 2}, {"pass": 2}))

lines = []
code = eval_caveat.main(["--repeats", "2"], answerer=model, out=lines.append)
text = "\n".join(lines)
check("it states its call count", (code, f"{len(CASES) * 3 * 2} model calls." in text), (0, True))
check("a deterministic model shows no unstable question/prompt pairs", "varied between repeats: 0" in text, True)

flip = {"n": 0}


def flaky(question, doc_text, history, caveat, notes, config):
    flip["n"] += 1
    return GroundedAnswer(answer="הערך לא קריא" if flip["n"] % 2 else "ביום 8.8.2027", quotes=[])


lines = []
eval_caveat.main(["--repeats", "2"], answerer=flaky, out=lines.append)
check("a model that answers differently on repeat is reported as unstable",
      any("varied between repeats" in l and not l.endswith(": 0") for l in lines), True)

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
