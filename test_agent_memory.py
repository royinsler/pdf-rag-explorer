"""Tests for agent_memory.py and its use by the agent loop.

The memory stores counters per finding kind and nothing else; these tests check that
(a) the counters mean what they say, (b) the policy built on them stops only when it
should and keeps probing, (c) no text from a document, question or answer can end up in
the file, (d) the loop uses it, and (e) a run that tells us nothing is not recorded.
Only the language model is faked. No API key.

    python3 test_agent_memory.py
"""
import contextlib
import io
import os
import sqlite3
import sys
import tempfile

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

import run_agent
from agent_graph import build_agent, stop_reason
from agent_memory import KNOWN_KINDS, RetryMemory
from agent_tools import DocumentContext
from answering import GroundedAnswer
from damaged_values import damage_caveat, detect_damaged_values

results = []


def check(name, actual, expected):
    ok = actual == expected
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"         expected {expected!r}\n         got      {actual!r}")


tmp = tempfile.TemporaryDirectory()


def db(name):
    return os.path.join(tmp.name, name)


# --- A. the counters mean what they say --------------------------------------------
m = RetryMemory(db("a.sqlite"))
check("A a new memory knows nothing and says retrying is worth it",
      (m.stats(), m.worth_retrying(["garbage_echo"])), ({}, True))
m.record(["garbage_echo"], [], retried=True)
check("A a retry that removed the finding is a try and a fix",
      m.stats()["garbage_echo"], {"tries": 1, "fixes": 1, "hopeless": 0})
m.record(["garbage_echo"], ["garbage_echo"], retried=True)
check("A a retry that left it in place is a try and not a fix",
      m.stats()["garbage_echo"], {"tries": 2, "fixes": 1, "hopeless": 0})
m.record(["garbage_echo"], ["garbage_echo"], retried=False)
check("A a run with no retry says nothing about whether retries work",
      m.stats()["garbage_echo"]["tries"], 2)
m.record(["unverified_fact", "garbage_echo"], ["unverified_fact"], retried=True)
check("A each kind is judged on its own: one fixed, one not",
      (m.stats()["garbage_echo"]["fixes"], m.stats()["unverified_fact"]["fixes"]), (2, 0))
m.close()

m = RetryMemory(db("a.sqlite"))
check("A it persists across instances", m.stats()["garbage_echo"]["tries"], 3)
other = RetryMemory(db("a.sqlite"), scope="another-model")
other.record(["garbage_echo"], [], retried=True)
check("A scopes are kept apart", (m.stats()["garbage_echo"]["tries"], other.stats()["garbage_echo"]["tries"]), (3, 1))
other.forget()
check("A forgetting one scope leaves the other", (other.stats(), "garbage_echo" in m.stats()), ({}, True))

# --- B. the policy -----------------------------------------------------------------
m = RetryMemory(db("b.sqlite"), min_observations=3, min_fix_rate=0.5, probe_every=100)
for _ in range(2):
    m.record(["garbage_echo"], ["garbage_echo"], retried=True)
check("B two failures are not enough to give up", m.worth_retrying(["garbage_echo"]), True)
m.record(["garbage_echo"], ["garbage_echo"], retried=True)
check("B three failures in three tries: not worth retrying", m.worth_retrying(["garbage_echo"]), False)
check("B a kind with no history is still worth retrying", m.worth_retrying(["fabricated_quote"]), True)
check("B ...and so is a mix with one promising kind", m.worth_retrying(["garbage_echo", "fabricated_quote"]), True)
for _ in range(3):
    m.record(["garbage_echo"], [], retried=True)
check("B at exactly the floor (3 fixes of 6) it is worth retrying again", m.worth_retrying(["garbage_echo"]), True)
check("B no findings, nothing to judge", m.worth_retrying([]), True)
check("B an unknown kind is never judged hopeless", m.worth_retrying(["something_new"]), True)

# --- C. it keeps probing: a policy that stops trying can never learn it was wrong ---
m = RetryMemory(db("c.sqlite"), min_observations=2, min_fix_rate=0.5, probe_every=3)
for _ in range(2):
    m.record(["garbage_echo"], ["garbage_echo"], retried=True)
seen = []
for _ in range(6):
    seen.append(m.worth_retrying(["garbage_echo"]))
    m.record(["garbage_echo"], ["garbage_echo"], retried=seen[-1], hopeless=True)
check("C every third hopeless run is retried anyway, evenly spaced",
      seen, [False, False, True, False, False, True])

# --- D. nothing but counters can be written ------------------------------------------
m = RetryMemory(db("d.sqlite"))
m.record(["garbage_echo", "SECRET CONTRACT TEXT 12.03.2028", "fabricated_quote"],
         ["SECRET ANSWER"], retried=True)
dump = "\n".join(m._db.iterdump())
check("D unknown kinds are ignored, so free text cannot be stored",
      sorted(m.stats()), ["fabricated_quote", "garbage_echo"])
check("D nothing resembling document, answer or question text is in the file",
      ("SECRET" in dump, "12.03.2028" in dump), (False, False))
check("D every stored kind is from the fixed list", set(m.stats()) <= set(KNOWN_KINDS), True)
raw = open(db("d.sqlite"), "rb").read()
check("D ...and the raw file bytes agree", b"SECRET" in raw, False)
check("D describe() reports counts only", m.describe(["garbage_echo"]),
      "garbage_echo: 1 of 1 past retries resolved it")

# --- E. the loop uses it ---------------------------------------------------------------
DOC = (
    'סך של 410,000 ש"ח ישולם עד ליום 3.9.2027 כנגד קבלת כספי המשכנתא.\n'
    'המוכר מתחייב למסור את הדירה לידי הקונה ביום 4.11.2027.\n'
    'לצורך ביצוע עבדות שיפוץ בלבד החל DYN 6,. מובהר כי כניסת DPN לנכס WAN במועד האמור.\n'
)
DAMAGED = detect_damaged_values(DOC, [("DYN", 40.0)])
assert DAMAGED
ECHO = ("הכניסה לשיפוץ אפשרית החל DYN 6,", [])
GOOD = ("הפינוי ביום 4.11.2027", ["המוכר מתחייב למסור את הדירה"])


def ctx():
    return DocumentContext(text=DOC, damaged=list(DAMAGED), caveat=damage_caveat(DAMAGED))


class Scripted:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0

    def __call__(self, question, doc, history, caveat, notes, config):
        self.calls += 1
        reply = self.replies[min(self.calls, len(self.replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        message = AIMessage(content="", usage_metadata={"input_tokens": 1000, "output_tokens": 100,
                                                        "total_tokens": 1100},
                            response_metadata={"model_name": "gpt-4o-mini"})
        for callback in config["callbacks"]:
            callback.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
        return GroundedAnswer(answer=reply[0], quotes=reply[1])


plain_model = Scripted(ECHO)
plain = build_agent(ctx(), plain_model).run("מתי ניתן להיכנס לשיפוץ?")
kinds = sorted({f["kind"] for f in plain["findings"]})
check("E without memory the loop retries a stubborn finding", (plain["attempts"] > 1, plain_model.calls > 1), (True, True))
check("E ...and the findings are of known kinds", set(kinds) <= set(KNOWN_KINDS) and bool(kinds), True)

hopeless_memory = RetryMemory(db("e1.sqlite"))
for _ in range(10):
    hopeless_memory.record(kinds, kinds, retried=True)
model = Scripted(ECHO)
report = build_agent(ctx(), model, memory=hopeless_memory).run("מתי ניתן להיכנס לשיפוץ?")
check("E with history saying retries never fix these, it makes one call and stops",
      (model.calls, report["attempts"], report["stop_reason"], report["status"]), (1, 1, "hopeless", "degraded"))
check("E ...and says why, in numbers, in the report",
      ("Stopped without retrying" in report["warnings"][0], "0 of 10" in report["warnings"][0]), (True, True))
check("E ...and the unresolved findings are still listed", bool(report["findings"]), True)

promising = RetryMemory(db("e2.sqlite"))
for _ in range(10):
    promising.record(kinds, [], retried=True)
model = Scripted(ECHO)
report = build_agent(ctx(), model, memory=promising).run("מתי ניתן להיכנס לשיפוץ?")
check("E with history saying retries usually fix these, it retries as before",
      (report["attempts"] > 1, report["stop_reason"] != "hopeless"), (True, True))

# a clean answer is never second-guessed by memory
model = Scripted(GOOD)
report = build_agent(ctx(), model, memory=hopeless_memory).run("מתי הפינוי?")
check("E a clean answer is verified whatever the memory says", (report["status"], model.calls), ("verified", 1))
check("E ...and a clean first attempt records nothing", hopeless_memory.stats()["garbage_echo"]["tries"], 10)

# --- F. what a run teaches it ---------------------------------------------------------------
fresh = RetryMemory(db("f1.sqlite"))
BAD_FACT = ("הפינוי ביום 8.8.2027", [])
report = build_agent(ctx(), Scripted(BAD_FACT, GOOD), memory=fresh).run("מתי הפינוי?")
check("F a retry that fixed the answer is recorded as a fix",
      (report["status"], fresh.stats()["unverified_fact"]), ("verified", {"tries": 1, "fixes": 1, "hopeless": 0}))

fresh = RetryMemory(db("f2.sqlite"))
report = build_agent(ctx(), Scripted(BAD_FACT, RuntimeError("down")), memory=fresh).run("מתי הפינוי?")
check("F a retry that FAILED to run (the model went down) is not recorded: it says nothing about retries",
      (report["stop_reason"], fresh.stats()), ("error", {}))

fresh = RetryMemory(db("f3.sqlite"))
build_agent(ctx(), Scripted(GOOD), memory=fresh).run("מתי הפינוי?")
check("F a clean first answer has no findings and teaches nothing", fresh.stats(), {})

# the probe, through the real loop: every second hopeless run retries anyway
probing = RetryMemory(db("f4.sqlite"), probe_every=2)
for _ in range(10):
    probing.record(kinds, kinds, retried=True)
attempts = []
for _ in range(4):
    attempts.append(build_agent(ctx(), Scripted(ECHO), memory=probing).run("מתי?")["attempts"])
check("F through the loop: hopeless runs stop at 1 attempt, every second one probes",
      [a > 1 for a in attempts], [False, True, False, True])
check("F ...and the probe fed its outcome back into the statistics",
      probing.stats()[kinds[0]]["tries"] > 10, True)

# --- G. stop_reason stays pure and orders the new reason last ------------------------------
state = {"attempt": 1, "findings": [{"kind": "garbage_echo", "detail": "x"}], "cost_usd": 0.0,
         "last_call_cost": 0.0, "attempts_log": [{"findings": [{"kind": "garbage_echo", "detail": "x"}]}]}
check("G a hopeless judgement stops a loop that is otherwise free to continue",
      stop_reason(state, 3, 0.05, lambda s: True), "hopeless")
check("G ...but not when the judgement is no", stop_reason(state, 3, 0.05, lambda s: False), None)
check("G ...and not when there is no judge at all (memory off)", stop_reason(state, 3, 0.05), None)
check("G a clean answer outranks it", stop_reason({**state, "findings": []}, 3, 0.05, lambda s: True), "verified")
check("G the attempt limit outranks it", stop_reason({**state, "attempt": 3}, 3, 0.05, lambda s: True), "max_attempts")

# --- H. the CLI -------------------------------------------------------------------------------
def cli(argv, answerer=None):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = run_agent.main(argv, answerer=answerer)
        except SystemExit as exit_:
            code = exit_.code
    return code, out.getvalue(), err.getvalue()


cli_db = db("h.sqlite")
code, out, _ = cli(["--memory-db", cli_db, "--memory-stats"])
check("H --memory-stats on an empty memory says so", (code, "empty" in out), (0, True))
RetryMemory(cli_db).record(["garbage_echo"], [], retried=True)
code, out, _ = cli(["--memory-db", cli_db, "--memory-stats"])
check("H --memory-stats prints counters and a rate, no content",
      (code, "garbage_echo" in out, "tries=1" in out, "100%" in out), (0, True, True, True))
code, out, _ = cli(["--memory-db", cli_db, "--forget-memory"])
code2, out2, _ = cli(["--memory-db", cli_db, "--memory-stats"])
check("H --forget-memory wipes it", (code, "empty" in out2), (0, True))
code, _, err = cli(["--memory-stats"])
check("H --memory-stats without --memory-db is a usage error, not a crash", code, 2)
code, _, err = cli([])
check("H no PDF and no memory command is a usage error", code, 2)

try:
    import fitz
    import pytesseract
    have_pdf = "eng" in pytesseract.get_languages()
except Exception:
    have_pdf = False
if have_pdf:
    pdf_path = db("contract.pdf")
    doc = fitz.open()
    doc.new_page().insert_text((72, 120), "The seller shall deliver by 12.03.2028 after payment.", fontsize=14)
    doc.save(pdf_path)
    doc.close()
    memory_db = db("h2.sqlite")
    code, out, _ = cli([pdf_path, "When is delivery?", "--memory-db", memory_db],
                       Scripted(("Delivery is due by 1.1.2030", []),
                                ("Delivery is due by 12.03.2028", ["shall deliver by 12.03.2028"])))
    stats = RetryMemory(memory_db).stats()
    check("H a real run through the CLI teaches the memory", (code, stats.get("unverified_fact")),
          (0, {"tries": 1, "fixes": 1, "hopeless": 0}))
    check("H ...without writing the contract's text to the memory file",
          (b"12.03.2028" in open(memory_db, "rb").read(), b"deliver" in open(memory_db, "rb").read()), (False, False))
else:
    print("[SKIP] CLI run with a real PDF (Tesseract with 'eng' not available)")

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
