"""Debris in a faithful quote must not fail a correct answer; debris the answer repeats still must.

Built from the 87-run live eval: "750 ש\"ח" (correct, clean) was marked degraded because its
supporting quote, copied exactly as the quote check demands, contained the OCR run DWNT.

    python3 test_quote_debris.py
"""
import sys

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from agent_graph import build_agent, collect_findings
from agent_tools import DocumentContext
from answering import GroundedAnswer
from damaged_values import damage_caveat, detect_damaged_values
from eval_fixtures import DAMAGED_CASES

results = []


def check(name, actual, expected):
    ok = actual == expected
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"         expected {expected!r}\n         got      {actual!r}")


case = next(c for c in DAMAGED_CASES if c["id"] == "relational")
damaged = detect_damaged_values(case["text"], case["low_confidence"])
ctx = DocumentContext(text=case["text"], damaged=list(damaged), caveat=damage_caveat(damaged))
DAMAGED_QUOTE = 'ישלם הספק פיצוי של 750 ש"ח, החל מהיום החמישה עשר DWNT לאיחור ועד למסירה בפועל.'


class Scripted:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0

    def __call__(self, question, doc, history, caveat, notes, config):
        self.calls += 1
        answer, quotes = self.replies[min(self.calls, len(self.replies)) - 1]
        message = AIMessage(content="", usage_metadata={"input_tokens": 1000, "output_tokens": 100,
                                                        "total_tokens": 1100},
                            response_metadata={"model_name": "gpt-4o-mini"})
        for callback in config["callbacks"]:
            callback.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
        return GroundedAnswer(answer=answer, quotes=quotes)


check("a correct answer whose faithful quote holds debris has no findings",
      collect_findings('750 ש"ח', [DAMAGED_QUOTE], ctx), [])
model = Scripted(('הסכום הוא 750 ש"ח.', [DAMAGED_QUOTE]))
report = build_agent(ctx, model).run("מהו סכום הפיצוי?")
check("...so it is verified on the first attempt, with no wasted retry",
      (report["status"], report["attempts"], model.calls), ("verified", 1, 1))
check("...and the damaged evidence is still reported, separately",
      (report["damaged_quotes"], report["echoes"]), (["DWNT"], []))

check("debris the ANSWER repeats is still a finding",
      [f["kind"] for f in collect_findings("החל מהיום החמישה עשר DWNT לאיחור", [DAMAGED_QUOTE], ctx)
       if f["kind"] == "garbage_echo"], ["garbage_echo"])
check("a quote that is not in the document is still a finding",
      [f["kind"] for f in collect_findings("750 ש\"ח", ["פיצוי של 999 ש\"ח"], ctx)], ["fabricated_quote"])
check("a date the document does not contain is still a finding",
      "unverified_fact" in [f["kind"] for f in collect_findings("עד 1.1.2031", [DAMAGED_QUOTE], ctx)], True)

# With no verified figure behind the answer, debris in the quote stays as hard as debris in the
# answer: a fluent invented substitute is caught, and so is the wrong word-only "המסירה בפועל.".
invented = collect_findings("הפיצוי מתחיל ביום חתימת ההסכם", [DAMAGED_QUOTE], ctx)
check("a fluent invented substitute with a damaged quote is still caught",
      [f["kind"] for f in invented], ["garbage_echo"])
wrong = collect_findings("המסירה בפועל.", [DAMAGED_QUOTE], ctx)
check("the wrong word-only answer from the live eval is still caught",
      [f["kind"] for f in wrong], ["garbage_echo"])

check("the original failure shape, a bare clause number as the answer, stays caught",
      [f["kind"] for f in collect_findings("החל מיום 6", [DAMAGED_QUOTE.replace("החמישה עשר", "6")], ctx)
       if f["kind"] == "garbage_echo"], ["garbage_echo"])
check("a bare year borrowed from elsewhere does not ground an answer",
      [f["kind"] for f in collect_findings("בשנת 2027", [DAMAGED_QUOTE], ctx)
       if f["kind"] == "garbage_echo"], ["garbage_echo"])
check("a figure the document does contain, with a faithful damaged quote, grounds it",
      collect_findings("הפיצוי הוא 750 ש\"ח לכל איחור", [DAMAGED_QUOTE], ctx), [])

# KNOWN REMAINING COST, asserted so it stays visible: a CORRECT word-only answer ("ארבע שעות")
# looks exactly like the wrong one above without reading the question, so it is flagged too.
case2 = next(c for c in DAMAGED_CASES if c["id"] == "relational2")
d2 = detect_damaged_values(case2["text"], case2["low_confidence"])
ctx2 = DocumentContext(text=case2["text"], damaged=list(d2), caveat=damage_caveat(d2))
q2 = "זמן התגובה לתקלה קריטית הוא ארבע שעות, החל מרגע קבלת הפנייה DION במוקד."
check("KNOWN REMAINING COST: a correct word-only answer with a damaged quote is still flagged",
      [f["kind"] for f in collect_findings("ארבע שעות", [q2], ctx2)], ["garbage_echo"])

# Direct checks of the grounding rule, built so each condition has a test that fails without it.
from verification import _grounding_figures, split_debris

check("grounding figures: 1-2 digit numbers are ignored", _grounding_figures("סעיף 6 ו-15"), set())
check("grounding figures: bare years are ignored", _grounding_figures("בשנת 2027 ו-1999"), set())
check("grounding figures: 3+ digit amounts count, commas removed", _grounding_figures('750 ש"ח ו-1,250'), {"750", "1250"})
DOC = 'פיצוי של 750 ש"ח, בסעיף 6, בשנת 2027, החל מהיום DWNT לאיחור.'
Q = ["החל מהיום DWNT לאיחור"]
check("a bare number present in the document does not ground: debris stays hard",
      split_debris("סעיף 6", Q, DOC), (["DWNT"], []))
check("a bare year present in the document does not ground: debris stays hard",
      split_debris("בשנת 2027", Q, DOC), (["DWNT"], []))
check("a real figure grounds: debris becomes a soft note",
      split_debris('750 ש"ח', Q, DOC), ([], ["DWNT"]))
check("a real figure alongside a date the document lacks does not ground",
      split_debris('750 ש"ח עד 1.1.2031', Q, DOC), (["DWNT"], []))

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
