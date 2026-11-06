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
      [f["kind"] for f in collect_findings("עד 1.1.2031", [DAMAGED_QUOTE], ctx)], ["unverified_fact"])

# KNOWN LIMITATION, asserted so it stays visible: a fluent invented answer with no figures, no
# long overlap with the damaged passage, and debris only in its quote now passes. Before this
# change the debris in the quote caught it. Delete this block deliberately if that is fixed.
invented = collect_findings("הפיצוי מתחיל ביום חתימת ההסכם", [DAMAGED_QUOTE], ctx)
check("KNOWN LIMITATION: a fluent invented substitute with a damaged quote is no longer caught", invented, [])

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
