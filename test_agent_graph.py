"""Tests for the verification agent graph.

Real: LangGraph, the verification checks, the damage detector, the tools, the
SQLite checkpointer, PyMuPDF/Tesseract and PDFMiner extraction (for the CLI
test). Faked: only the language model - a scripted answerer that also reports
token usage through the same callback a real call would, so cost accounting is
exercised too. No API key needed:

    python3 test_agent_graph.py

Every scenario here is a way the loop can go wrong - not terminating, hiding a
failed answer, losing work on a crash, persisting a document, mispricing a call.
"""
import contextlib
import glob
import io
import json
import os
import sqlite3
import sys
import tempfile

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.runnables import RunnableLambda

import agent_graph
import agent_llm
import answering
import run_agent
from agent_graph import (build_agent, collect_findings, purge_thread, sqlite_checkpointer,
                         stop_reason, _best_attempt)
from agent_state import SCHEMA_VERSION
from agent_tools import DocumentContext
from answering import GroundedAnswer
from damaged_values import DamagedValue, damage_caveat, detect_damaged_values

results = []
skipped = []


def check(name, actual, expected):
    ok = actual == expected
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"         expected {expected!r}\n         got      {actual!r}")


def raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


# --- fixtures ----------------------------------------------------------------

DOC = (
    'סך של 410,000 ש"ח ישולם לשני יחידי המוכר עד ליום 3.9.2027 כנגד קבלת כספי המשכנתא.\n'
    'המוכר מתחייב לפנות את הדירה ולמסור אותה לידי הקונה ביום 4.11.2027.\n'
    'הדירה נמכרת במצבה כמו שהוא ("As-Is").\n'
    'לצורך ביצוע עבדות שיפוץ בלבד החל DYN 6,. מובהר כי כניסת DPN לנכס WAN במועד האמור.\n'
    'SENTINEL-CLAUSE-7731 סעיף שאין לו קשר לשאלות.'
)
DAMAGED = detect_damaged_values(DOC, [("DYN", 40.0)])
assert DAMAGED, "fixture is meant to contain a damaged passage"
SNIPPET = DAMAGED[0].snippet
CALL_COST = 0.00021               # 1000 in + 100 out at the gpt-4o-mini rate

GOOD = ("הפינוי ביום 4.11.2027", ["המוכר מתחייב לפנות את הדירה"])


def make_ctx():
    return DocumentContext(text=DOC, damaged=list(DAMAGED), caveat=damage_caveat(DAMAGED))


def usage(config, model="gpt-4o-mini", tokens_in=1000, tokens_out=100):
    """Report token usage exactly as a real model call does: through the callback."""
    message = AIMessage(content="", usage_metadata={
        "input_tokens": tokens_in, "output_tokens": tokens_out,
        "total_tokens": tokens_in + tokens_out}, response_metadata={"model_name": model})
    for callback in config["callbacks"]:
        callback.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))


class Scripted:
    """Plays back replies in order; the last one repeats. An Exception is raised."""

    def __init__(self, *replies, model="gpt-4o-mini", report_usage=True):
        self.replies, self.model, self.report_usage, self.calls = list(replies), model, report_usage, []

    def __call__(self, question, doc, history, caveat, notes, config):
        self.calls.append({"question": question, "notes": notes, "caveat": caveat, "doc": doc})
        reply = self.replies[min(len(self.calls), len(self.replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        if self.report_usage:
            usage(config, self.model)
        return GroundedAnswer(answer=reply[0], quotes=reply[1])


def nodes(report):
    return [entry["node"] for entry in report["trace"]]


# --- A. a clean answer passes straight through -------------------------------
model = Scripted(GOOD)
report = build_agent(make_ctx(), model).run("מתי הפינוי?")
check("A clean answer is verified on the first attempt",
      (report["status"], report["attempts"], report["stop_reason"]), ("verified", 1, "verified"))
check("A ...and never enters the investigate step", nodes(report), ["answer", "verify"])
check("A cost is read from the callback and priced", (report["cost_usd"], report["cost_complete"]),
      (CALL_COST, True))
check("A tokens are counted", (report["tokens_in"], report["tokens_out"]), (1000, 100))
check("A the model is handed the document and the damage caveat",
      (model.calls[0]["doc"] == DOC, model.calls[0]["caveat"] == make_ctx().caveat), (True, True))
check("A the first attempt carries no notes", model.calls[0]["notes"], "")
check("A the report has the UI's shape: quotes, facts, echoes, rests_on",
      (report["quotes"], report["facts"], report["echoes"], report["rests_on"]),
      ([("המוכר מתחייב לפנות את הדירה", True)], [("4.11.2027", True)], [], []))
check("A latency is measured", report["latency_s"] >= 0, True)

# --- B. a fabricated date is caught, investigated, corrected -----------------
model = Scripted(("הפינוי ביום 8.8.2027", []), GOOD)
report = build_agent(make_ctx(), model).run("מתי הפינוי?")
check("B the loop recovers: verified on attempt 2",
      (report["status"], report["attempts"]), ("verified", 2))
check("B the path through the graph is answer, verify, investigate, answer, verify",
      nodes(report), ["answer", "verify", "investigate", "answer", "verify"])
check("B the retry prompt names the value that was rejected", "8.8.2027" in model.calls[1]["notes"], True)
check("B the first attempt had no notes, the second had", (model.calls[0]["notes"], bool(model.calls[1]["notes"])),
      ("", True))
check("B cost accumulates across both attempts", report["cost_usd"], round(2 * CALL_COST, 6))
check("B the final answer is the corrected one", "4.11.2027" in report["answer"], True)

# --- C. a loop that cannot succeed must stop, and must say so ----------------
model = Scripted(("הפינוי ביום 8.8.2027", []))
report = build_agent(make_ctx(), model).run("מתי הפינוי?")
check("C identical findings twice running stop the loop", (report["stop_reason"], report["attempts"]),
      ("no_progress", 2))
check("C ...and the result is degraded, not relabelled as fine", report["status"], "degraded")
check("C ...the bad value is still marked unverified in the facts", report["facts"], [("8.8.2027", False)])
check("C ...and the warning says why it stopped and what is unresolved",
      ("same findings" in report["warnings"][0], "unverified_fact" in report["warnings"][0]), (True, True))

model = Scripted(("8.8.2027", []), ("9.9.2027", []), ("7.7.2027", []))
report = build_agent(make_ctx(), model).run("מתי הפינוי?")
check("C changing wrong answers hit the attempt cap instead",
      (report["stop_reason"], report["attempts"], len(model.calls)), ("max_attempts", 3, 3))
check("C ...the cap is configurable",
      build_agent(make_ctx(), Scripted(("8.8.2027", []), ("9.9.2027", [])),
                  max_attempts=2).run("q")["attempts"], 2)

# --- D. best-of: a retry that made things worse is not what the user sees ----
model = Scripted(("1.1.2027 ו-2.2.2027", []),              # 2 unverified
                 ("4.11.2027 ו-5.5.2027", []),             # 1 unverified (4.11.2027 is real)
                 ("6.6.2027 ו-7.7.2027", []))              # 2 unverified
report = build_agent(make_ctx(), model).run("q")
check("D the least-wrong attempt is returned, not the last",
      (report["answer"], len(report["findings"])), ("4.11.2027 ו-5.5.2027", 1))
check("D best_attempt prefers the later attempt on a tie",
      _best_attempt([{"attempt": 1, "findings": [1]}, {"attempt": 2, "findings": [1]}])["attempt"], 2)
check("D best_attempt of nothing is None", _best_attempt([]), None)

# --- E. the budget is checked BEFORE paying for another attempt --------------
model = Scripted(("הפינוי ביום 8.8.2027", []), GOOD)
report = build_agent(make_ctx(), model, budget_usd=0.0003).run("q")
check("E a retry that would exceed the budget is not made",
      (report["status"], report["stop_reason"], report["attempts"], len(model.calls)),
      ("budget_exceeded", "budget", 1, 1))
check("E ...and the warning names the budget", "$0.00" in report["warnings"][0], True)

# --- F. the model being down is an outcome, not a crash ----------------------
report = build_agent(make_ctx(), Scripted(RuntimeError("boom"))).run("q")
check("F a model that never answers gives status failed, no exception",
      (report["status"], report["answer"], report["attempts"]), ("failed", "", 1))
check("F ...with the reason in the warnings", "boom" in report["warnings"][0], True)

report = build_agent(make_ctx(), Scripted(("הפינוי ביום 8.8.2027", []), RuntimeError("down"))).run("q")
check("F a failure on a RETRY returns the best earlier attempt, degraded",
      (report["status"], report["stop_reason"], "8.8.2027" in report["answer"]), ("degraded", "error", True))
check("F ...and says the retry failed", "retry failed" in report["warnings"][0], True)

# --- G. damage provenance + garbage echo, no page map -------------------------
model = Scripted((SNIPPET, []), ("המסמך אינו מציג את התאריך באופן קריא", []))
report = build_agent(make_ctx(), model).run("מה התאריך לשיפוץ?")
check("G copying a damaged passage is caught and then corrected",
      (report["status"], report["attempts"]), ("verified", 2))
check("G the retry notes explain the damage and the echoed debris",
      ("damaged" in model.calls[1]["notes"], "DYN" in model.calls[1]["notes"]), (True, True))
investigate = [t for t in report["trace"] if t["node"] == "investigate"][0]
check("G with no page map no tool is attempted, rather than guessing a page", investigate["tool_calls"], [])

# --- H. the re-read tool recovers a lost value, end to end --------------------
def tesseract_ready():
    try:
        import pytesseract
        return "eng" in pytesseract.get_languages()
    except Exception:
        return False


if tesseract_ready():
    import fitz
    pdf = fitz.open()
    pdf.new_page().insert_text((72, 120), "The seller shall deliver by 12.03.2028 after payment.", fontsize=14)
    pdf_bytes = pdf.tobytes()
    pdf.close()

    damaged_text = "The seller shall deliver by DYN 6 after payment."
    ctx = DocumentContext(
        text=damaged_text, pages={1: damaged_text}, file_bytes=pdf_bytes, ocr_lang="eng",
        damaged=[DamagedValue(connector="by", snippet="seller shall deliver by DYN 6 after",
                              reasons=["Latin characters where Hebrew is expected", "a bare number"])])
    model = Scripted(("The seller shall deliver by DYN 6 after", []), ("Delivery is due by 12.03.2028.", []))
    report = build_agent(ctx, model).run("When must the seller deliver?")
    check("H the damaged passage triggers a re-read, and the retry succeeds",
          (report["status"], report["attempts"]), ("verified", 2))
    investigate = [t for t in report["trace"] if t["node"] == "investigate"][0]
    check("H the re-read tool ran and worked",
          investigate["tool_calls"], [{"tool": "reocr_page", "ok": True, "error": None}])
    check("H the second reading reached the model", "12.03.2028" in model.calls[1]["notes"], True)
    check("H the recovered date verifies against the re-read page", report["facts"], [("12.03.2028", True)])
    check("H the original extraction is untouched", ctx.text, damaged_text)
else:
    skipped.append("H re-read end to end (Tesseract with 'eng' not available)")

# --- I. checkpointing: resume after a crash, and what the file holds ----------
with tempfile.TemporaryDirectory() as tmp:
    db = os.path.join(tmp, "agent.sqlite")
    saver = sqlite_checkpointer(db)
    model = Scripted(GOOD)
    agent = build_agent(make_ctx(), model, checkpointer=saver)

    original, state = agent_graph.collect_findings, {"n": 0}

    def flaky(*args, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("simulated crash in verify")
        return original(*args, **kwargs)

    agent_graph.collect_findings = flaky
    try:
        crashed = False
        try:
            agent.run("מתי הפינוי?", thread_id="t1")
        except RuntimeError:
            crashed = True
        check("I the run crashes mid-graph, after the model was paid for", (crashed, len(model.calls)), (True, 1))

        report = agent.resume("t1")
        check("I resume finishes the run from the checkpoint", report["status"], "verified")
        check("I ...WITHOUT calling the model again", len(model.calls), 1)
    finally:
        agent_graph.collect_findings = original

    stored = agent.graph.get_state({"configurable": {"thread_id": "t1"}}).values
    check("I the stored state carries the schema version", stored["schema_version"], SCHEMA_VERSION)

    blob = b"".join(open(path, "rb").read() for path in glob.glob(db + "*"))
    check("I the document itself is never written to the checkpoint file",
          b"SENTINEL-CLAUSE-7731" not in blob, True)
    check("I ...but derived content (the answer) is, which is why purge exists",
          "הפינוי".encode() in blob, True)

    second = agent.run("שאלה שנייה?", thread_id="t2")
    check("I each question gets its own thread: no attempts or notes leak between them",
          (second["attempts"], nodes(second)), (1, ["answer", "verify"]))

    purge_thread(saver, "t1")
    rows = sqlite3.connect(db)
    left = [rows.execute(f"SELECT count(*) FROM {table} WHERE thread_id='t1'").fetchone()[0]
            for table in ("checkpoints", "writes")]
    other = rows.execute("SELECT count(*) FROM checkpoints WHERE thread_id='t2'").fetchone()[0]
    rows.close()
    check("I purge removes that thread's checkpoints and writes", left, [0, 0])
    check("I ...and leaves other threads alone", other > 0, True)

check("I resume without a checkpointer is refused clearly",
      raises(ValueError, lambda: build_agent(make_ctx(), Scripted(GOOD)).resume("x")), True)

# --- J. cost: unpriced calls are reported as unpriced ------------------------
report = build_agent(make_ctx(), Scripted(GOOD, model="mystery-model-9")).run("q")
check("J an unknown model is flagged incomplete rather than free",
      (report["cost_usd"], report["cost_complete"]), (0.0, False))
report = build_agent(make_ctx(), Scripted(GOOD, report_usage=False)).run("q")
check("J a call that reported no usage is also incomplete, not $0.00 and fine",
      report["cost_complete"], False)

# --- K. the pure stopping rule, including precedence --------------------------
F = [{"kind": "unverified_fact", "detail": "8.8.2027"}]
G = [{"kind": "unverified_fact", "detail": "9.9.2027"}]
base = {"attempt": 1, "findings": F, "cost_usd": 0.0, "last_call_cost": 0.0,
        "attempts_log": [{"findings": F}]}
check("K keeps going when nothing says stop", stop_reason(base, 3, 1.0), None)
check("K error beats everything", stop_reason({**base, "error": "x", "findings": []}, 3, 1.0), "error")
check("K a clean answer beats the attempt cap", stop_reason({**base, "attempt": 3, "findings": []}, 3, 1.0), "verified")
check("K the attempt cap", stop_reason({**base, "attempt": 3}, 3, 1.0), "max_attempts")
check("K the budget guard assumes the next call costs what the last did",
      stop_reason({**base, "cost_usd": 0.6, "last_call_cost": 0.5}, 3, 1.0), "budget")
check("K different findings are progress",
      stop_reason({**base, "attempt": 2, "attempts_log": [{"findings": F}, {"findings": G}]}, 3, 1.0), None)
check("K the same findings twice are not",
      stop_reason({**base, "attempt": 2, "attempts_log": [{"findings": F}, {"findings": F}]}, 3, 1.0), "no_progress")

# --- L. all four finding kinds at once ----------------------------------------
found = collect_findings(SNIPPET + " 8.8.2027", ["a quote the document never contained"], make_ctx())
check("L all four checks feed findings", {f["kind"] for f in found},
      {"fabricated_quote", "unverified_fact", "garbage_echo", "damage_provenance"})
check("L a clean answer yields none", collect_findings(GOOD[0], GOOD[1], make_ctx()), [])

# --- M. the OpenAI answerer: backoff, fallback, escaping -----------------------
calls, sleeps = [], []
real_afd = agent_llm.answer_from_full_document


def flaky_afd(question, doc, history, caveat="", notes="", model=None, config=None):
    calls.append((model, caveat))
    if len(calls) <= 2:
        raise TimeoutError("slow")
    return GroundedAnswer(answer="ok", quotes=[])


agent_llm.answer_from_full_document = flaky_afd
try:
    fn = agent_llm.make_openai_answerer(models=("m1", "m2"), attempts_per_model=2,
                                        base_delay=1.0, sleep=sleeps.append)
    answer = fn("q", "doc", [], "{braces}", "", {})
    check("M two failures on the first model fall back to the second",
          ([c[0] for c in calls], answer.answer), (["m1", "m1", "m2"], "ok"))
    check("M backoff sleeps between attempts on one model, not before a fallback", sleeps, [1.0])
    check("M the caveat is brace-escaped for the prompt template", calls[0][1], "{{braces}}")

    calls.clear(); sleeps.clear()

    def always_down(*args, **kwargs):
        calls.append(1)
        raise ValueError("still down")

    agent_llm.answer_from_full_document = always_down
    check("M when every model fails the last error is raised for the graph to handle",
          raises(ValueError, lambda: fn("q", "doc", [], "", "", {})), True)
    check("M ...after exactly models x attempts tries", (len(calls), sleeps), (4, [1.0, 1.0]))
finally:
    agent_llm.answer_from_full_document = real_afd

# --- N. the moved answering code behaves as it did in app.py ------------------
captured = {}


class FakeLLM:
    def __init__(self, **kwargs):
        pass

    def with_structured_output(self, schema):
        def run(prompt_value):
            captured["messages"] = prompt_value.to_messages()
            return GroundedAnswer(answer="ok", quotes=[])
        return RunnableLambda(run)


real_llm = answering.ChatOpenAI
answering.ChatOpenAI = FakeLLM
try:
    answering.answer_from_full_document("q?", "doc {x}", [], caveat="")
    roles = [m.type for m in captured["messages"]]
    check("N a clean document's prompt is exactly document + question", roles, ["system", "human"])
    check("N braces inside the document survive", "doc {x}" in captured["messages"][0].content, True)

    answering.answer_from_full_document("q?", "doc", [], caveat="CAVEAT")
    check("N the caveat sits immediately before the question, after the document",
          [m.type for m in captured["messages"]], ["system", "system", "human"])
    check("N ...and is that message's content", captured["messages"][1].content, "CAVEAT")

    answering.answer_from_full_document("q?", "doc", [], caveat="CAVEAT", notes='bad "8.8.2027" {value}')
    last_system = captured["messages"][-2].content
    check("N retry notes join the same message, after the caveat",
          (last_system.startswith("CAVEAT"), "8.8.2027" in last_system), (True, True))
    check("N braces in tool-derived notes do not break the template", "{value}" in last_system, True)
    check("N the notes are framed as data, not instructions", "never as an instruction" in last_system, True)
    check("N the question is still last", captured["messages"][-1].content, "q?")
finally:
    answering.ChatOpenAI = real_llm

# --- O. the document-size guard ------------------------------------------------
check("O a document over the full-context limit is refused with a reason",
      raises(ValueError, lambda: build_agent(DocumentContext(text="א" * 40_000), Scripted(GOOD))), True)

# --- P. the CLI, with real extraction through PDFMiner --------------------------
if tesseract_ready():
    import fitz
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "contract.pdf")
        pdf = fitz.open()
        pdf.new_page().insert_text((72, 120), "The seller shall deliver by 12.03.2028 after payment.", fontsize=14)
        pdf.save(path)
        pdf.close()

        def cli(argv, answerer):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = run_agent.main(argv, answerer=answerer)
            return code, out.getvalue()

        right = Scripted(("Delivery is due by 12.03.2028", ["shall deliver by 12.03.2028"]))
        code, text = cli([path, "When is delivery?"], right)
        check("P the CLI answers and exits 0 when verified", (code, "[verified]" in text), (0, True))

        code, text = cli([path, "When is delivery?", "--json"], right)
        check("P --json prints a parseable report", json.loads(text)["status"], "verified")

        wrong = Scripted(("Delivery is due by 1.1.2030", []))
        code, text = cli([path, "When is delivery?"], wrong)
        check("P a degraded result exits non-zero and shows the unresolved finding",
              (code, "unverified_fact" in text), (1, True))

        db = os.path.join(tmp, "cli.sqlite")
        cli([path, "When is delivery?", "--checkpoint-db", db, "--thread", "cli1"], right)
        code, _ = cli([path, "--purge", "cli1", "--checkpoint-db", db], None)
        rows = sqlite3.connect(db)
        left = rows.execute("SELECT count(*) FROM checkpoints WHERE thread_id='cli1'").fetchone()[0]
        rows.close()
        check("P --purge deletes a thread's checkpoints", (code, left), (0, 0))

        saved = os.environ.pop("OPENAI_API_KEY", None)
        try:
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = run_agent.main([path, "q"])
            check("P with no API key the CLI says so and exits 2", (code, "OPENAI_API_KEY" in err.getvalue()), (2, True))
        finally:
            if saved is not None:
                os.environ["OPENAI_API_KEY"] = saved
else:
    skipped.append("P CLI end to end (Tesseract with 'eng' not available, PDF fixture skipped)")

for note in skipped:
    print(f"[SKIP] {note}")
print(f"\n{sum(results)}/{len(results)} passed" + (f", {len(skipped)} skipped" if skipped else ""))
sys.exit(0 if all(results) else 1)
