"""The verification loop, as a LangGraph graph.

    START -> answer -> verify --clean--------------------------> finalize -> END
                          |                                          ^
                          +--findings--> investigate -> answer ...   |
                                  (bounded: attempts, budget, progress)

What this adds to the pipeline: verification.py already knows when an answer is
wrong. Until now that knowledge ended in a warning badge. Here it drives a
bounded retry - the failed answer's findings are turned into evidence (tools
re-read the document), the model answers again, and the checks run again.

Design decisions worth stating, because each one is a place the loop could
quietly go wrong:

  * Stopping is a single pure function (stop_reason), used by both the router
    and finalize, so "why did it stop" and "what do we report" cannot disagree.
  * The loop is bounded three ways: max_attempts, a cost budget (checked BEFORE
    paying for another attempt), and a no-progress rule (identical findings two
    attempts running means a third will not differ).
  * Hitting a bound is never silent. The best attempt is returned with
    status="degraded" and the unresolved findings listed. An answer that failed
    its checks is not relabelled as fine because we ran out of tries.
  * "Best" is chosen across ALL attempts (fewest findings, latest on ties), so a
    retry that made things worse is not what the user sees.
  * The investigate step is rule-driven, not model-driven: findings map to tool
    calls deterministically. That keeps it cheap and testable, at the price of
    being less flexible than a model choosing tools. The tools are real
    StructuredTools, so a model-driven policy can replace it without touching
    the rest.
  * Scope: full-context documents only (the path whose answers carry verifiable
    quotes). Larger documents go through retrieval and are not covered yet.
"""
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from agent_state import (MAX_ATTEMPTS, SCHEMA_VERSION, STATUS_BUDGET,
                         STATUS_DEGRADED, STATUS_FAILED, STATUS_VERIFIED,
                         AgentState)
from agent_tools import (DocumentContext, MAX_TOOL_OUTPUT_CHARS, build_tools,
                         excerpt_around, run_tool)
from cost_tracking import UsageCallback
from verification import (answer_rests_on_damage, find_ocr_garbage_echoes,
                          verify_answer_facts, verify_quotes)

# Tool calls per investigate step. Re-OCR is the slow one; this keeps a single
# retry from fanning out across every damaged passage in a long contract.
MAX_TOOL_CALLS_PER_STEP = 4
# Notes carried into a retry prompt. Older ones fall off: every note is paid for
# again on each attempt.
MAX_NOTES_IN_PROMPT = 8

# answer_fn(question, doc_text, history_messages, caveat, notes, config) -> object
# with .answer (str) and .quotes (list[str]) - answering.GroundedAnswer.
AnswerFn = Callable[..., Any]


# --- verification ------------------------------------------------------------

def collect_findings(answer: str, quotes: List[str], ctx: DocumentContext) -> List[Dict[str, str]]:
    """Run all four checks and flatten what they report into findings.

    Calls the same functions app.py calls, on the same inputs, so the agent and
    the UI cannot disagree about whether an answer passed.
    """
    doc = ctx.verification_text()
    findings = []
    for quote, found in verify_quotes(quotes, doc):
        if not found:
            findings.append({"kind": "fabricated_quote", "detail": quote})
    for value, found in verify_answer_facts(answer, doc):
        if not found:
            findings.append({"kind": "unverified_fact", "detail": value})
    for run in find_ocr_garbage_echoes(answer + " " + " ".join(quotes), doc):
        findings.append({"kind": "garbage_echo", "detail": run})
    for snippet, _overlap in answer_rests_on_damage(answer, [d.snippet for d in ctx.damaged]):
        findings.append({"kind": "damage_provenance", "detail": snippet})
    return findings


def _signature(findings: List[Dict[str, str]]) -> frozenset:
    return frozenset((f["kind"], f["detail"]) for f in findings)


def _best_attempt(attempts_log: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Fewest findings wins; on a tie the later attempt (it saw more evidence)."""
    if not attempts_log:
        return None
    return min(reversed(attempts_log), key=lambda a: len(a["findings"]))


def stop_reason(state: AgentState, max_attempts: int, budget_usd: float) -> Optional[str]:
    """None to keep going, otherwise why the loop is ending.

    Order matters and is the order of severity: a model that cannot be reached
    outranks everything, a clean answer outranks every bound.
    """
    if state.get("error"):
        return "error"
    if not state.get("findings"):
        return "verified"
    if state.get("attempt", 0) >= max_attempts:
        return "max_attempts"
    # Checked before paying for another attempt: assume the next costs what the
    # last one did. Stopping after exceeding the budget is just reporting it.
    if state.get("cost_usd", 0.0) + state.get("last_call_cost", 0.0) > budget_usd:
        return "budget"
    log = state.get("attempts_log", [])
    if len(log) >= 2 and _signature(log[-1]["findings"]) == _signature(log[-2]["findings"]):
        return "no_progress"
    return None


# --- helpers -----------------------------------------------------------------

def _messages(history: List[Dict[str, str]]):
    out = []
    for turn in history:
        out.append(HumanMessage(content=turn["user"]))
        out.append(AIMessage(content=turn["bot"]))
    return out


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --- graph -------------------------------------------------------------------

@dataclass
class Agent:
    graph: Any
    ctx: DocumentContext
    max_attempts: int
    budget_usd: float
    has_checkpointer: bool

    def run(self, question: str, history: Optional[List[Dict[str, str]]] = None,
            thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Answer one question and return the finished report.

        One thread per QUESTION, not per conversation. Accumulating fields
        (attempts, notes, trace) would otherwise carry one question's retries
        into the next. Conversation memory is the `history` argument, owned by
        the caller exactly as it is in app.py.
        """
        config = self._config(thread_id or uuid.uuid4().hex)
        initial: AgentState = {
            "schema_version": SCHEMA_VERSION, "question": question,
            "history": history or [], "attempt": 0, "cost_usd": 0.0,
            "last_call_cost": 0.0, "tokens_in": 0, "tokens_out": 0,
            "unpriced_calls": 0, "findings": [], "error": None,
        }
        return self.graph.invoke(initial, config)["report"]

    def resume(self, thread_id: str) -> Dict[str, Any]:
        """Continue a run that was interrupted (a crash, a kill) from its last checkpoint."""
        if not self.has_checkpointer:
            raise ValueError("resume needs a checkpointer; this agent was built without one")
        return self.graph.invoke(None, self._config(thread_id))["report"]

    def _config(self, thread_id: str) -> Dict[str, Any]:
        return {"configurable": {"thread_id": thread_id},
                "recursion_limit": 4 * self.max_attempts + 5}


def build_agent(ctx: DocumentContext, answer_fn: AnswerFn, *,
                budget_usd: float = 0.05, max_attempts: int = MAX_ATTEMPTS,
                checkpointer=None) -> Agent:
    """Assemble the graph for one document.

    budget_usd is per question, not per session. The default is a guard rail
    sized for gpt-4o-mini on a contract-length document - a deliberately low
    number, so that a runaway loop is a small bill; raise it knowingly.
    """
    from extraction import FULL_CONTEXT_MAX_CHARS, fits_in_context
    if not fits_in_context(ctx.text):
        raise ValueError(
            f"document is {len(ctx.text):,} characters, over the {FULL_CONTEXT_MAX_CHARS:,} "
            "full-context limit. The agent covers the full-context path only; larger "
            "documents go through retrieval, which it does not handle yet.")

    tools = build_tools(ctx)

    def answer_node(state: AgentState):
        started = time.perf_counter()
        attempt = state.get("attempt", 0) + 1
        usage = UsageCallback()
        notes = "\n".join(state.get("notes", [])[-MAX_NOTES_IN_PROMPT:])
        try:
            result = answer_fn(state["question"], ctx.text, _messages(state.get("history", [])),
                               ctx.caveat, notes, {"callbacks": [usage]})
        except Exception as exc:        # noqa: BLE001 - the model being down is an outcome, not a crash
            return {"attempt": attempt, "error": f"{type(exc).__name__}: {exc}",
                    "trace": [{"node": "answer", "attempt": attempt, "ok": False,
                               "seconds": round(time.perf_counter() - started, 3),
                               "error": f"{type(exc).__name__}: {exc}"}]}
        # A call that reported no usage at all is UNPRICED, not free: 0.0 here
        # would read as "this cost nothing" and quietly weaken the budget guard.
        cost = usage.cost_usd() if usage.calls else None
        return {
            "attempt": attempt, "answer": result.answer, "quotes": list(result.quotes),
            "error": None,
            "cost_usd": state.get("cost_usd", 0.0) + (cost or 0.0),
            "last_call_cost": cost or 0.0,
            "tokens_in": state.get("tokens_in", 0) + usage.input_tokens,
            "tokens_out": state.get("tokens_out", 0) + usage.output_tokens,
            "unpriced_calls": state.get("unpriced_calls", 0) + (1 if cost is None else 0),
            "trace": [{"node": "answer", "attempt": attempt, "ok": True,
                       "seconds": round(time.perf_counter() - started, 3),
                       "cost_usd": None if cost is None else round(cost, 8),
                       "tokens_in": usage.input_tokens,
                       "tokens_out": usage.output_tokens,
                       "had_notes": bool(notes)}],
        }

    def verify_node(state: AgentState):
        started = time.perf_counter()
        findings = collect_findings(state["answer"], state.get("quotes", []), ctx)
        return {
            "findings": findings,
            "attempts_log": [{"attempt": state["attempt"], "answer": state["answer"],
                              "quotes": state.get("quotes", []), "findings": findings}],
            "trace": [{"node": "verify", "attempt": state["attempt"],
                       "seconds": round(time.perf_counter() - started, 3),
                       "findings": [f["kind"] for f in findings]}],
        }

    def investigate_node(state: AgentState):
        started = time.perf_counter()
        notes, calls = [], []
        reread_pages = set()

        def call(name, **args):
            result = run_tool(tools, name, **args)
            calls.append({"tool": name, "ok": result.ok, "error": result.error})
            return result

        for finding in state.get("findings", []):
            if len(calls) >= MAX_TOOL_CALLS_PER_STEP:
                break
            kind, detail = finding["kind"], finding["detail"]

            if kind == "unverified_fact":
                notes.append(f'The value "{detail}" in your previous answer does not '
                             "appear anywhere in the document. Do not state it.")

            elif kind == "fabricated_quote":
                note = (f'The passage "{_clip(detail, 120)}" that you quoted is not in the '
                        "document. Quote only text that is really there, character for character.")
                hit = call("search_document", query=" ".join(detail.split()[:6]), max_hits=1)
                if hit.ok and hit.data.get("hits"):
                    note += (' The nearest real text is: "'
                             + _clip(hit.data["hits"][0]["excerpt"], 240) + '"')
                notes.append(note)

            elif kind == "garbage_echo":
                notes.append(f'"{detail}" is character debris left by OCR where it could not '
                             "read the page. It is not content; do not repeat it.")

            elif kind == "damage_provenance":
                note = ("Your answer rests on a passage whose extraction is damaged: "
                        f'"{_clip(detail, 160)}". If the question depends on a value there, '
                        "say the document does not show it legibly.")
                page = ctx.page_containing(detail)
                if page is not None and page not in reread_pages:
                    reread_pages.add(page)
                    reread = call("reocr_page", page=page)
                    connector = next((d.connector for d in ctx.damaged if d.snippet == detail), "")
                    if reread.ok and connector:
                        seen = excerpt_around(reread.data["text"], connector)
                        if seen:
                            note += (f" A second reading of page {page} at {reread.data['dpi']} DPI "
                                     'shows, near the same point: "' + '" / "'.join(seen) + '"')
                notes.append(note)

        return {
            "notes": [_clip(n, MAX_TOOL_OUTPUT_CHARS) for n in notes],
            "trace": [{"node": "investigate", "attempt": state.get("attempt", 0),
                       "seconds": round(time.perf_counter() - started, 3),
                       "tool_calls": calls}],
        }

    def finalize_node(state: AgentState):
        started = time.perf_counter()
        reason = stop_reason(state, max_attempts, budget_usd) or "max_attempts"
        best = _best_attempt(state.get("attempts_log", []))
        doc = ctx.verification_text()
        warnings: List[str] = []

        if best is None:                                  # the model never produced anything
            status = STATUS_FAILED
            warnings.append(f"The model could not be reached: {state.get('error')}")
            report = {"answer": "", "quotes": [], "facts": [], "echoes": [], "rests_on": [],
                      "findings": []}
        else:
            answer, quotes = best["answer"], best["quotes"]
            report = {
                "answer": answer,
                "quotes": verify_quotes(quotes, doc),
                "facts": verify_answer_facts(answer, doc),
                "echoes": find_ocr_garbage_echoes(answer + " " + " ".join(quotes), doc),
                "rests_on": answer_rests_on_damage(answer, [d.snippet for d in ctx.damaged]),
                "findings": best["findings"],
            }
            if reason == "verified":
                status = STATUS_VERIFIED
            else:
                status = STATUS_BUDGET if reason == "budget" else STATUS_DEGRADED
                unresolved = ", ".join(f"{f['kind']} ({_clip(f['detail'], 40)!r})"
                                       for f in best["findings"]) or "none recorded"
                warnings.append({
                    "max_attempts": f"Stopped after {state.get('attempt')} attempts.",
                    "budget": f"Stopped to stay inside the ${budget_usd:.2f} budget.",
                    "no_progress": "Stopped: a retry produced the same findings as the one before.",
                    "error": f"A retry failed ({state.get('error')}); returning the best earlier attempt.",
                }.get(reason, "Stopped.") + f" Unresolved: {unresolved}.")

        trace = list(state.get("trace", []))
        total_seconds = sum(entry.get("seconds", 0.0) for entry in trace)
        report.update({
            "status": status, "warnings": warnings, "stop_reason": reason,
            "attempts": state.get("attempt", 0),
            "cost_usd": round(state.get("cost_usd", 0.0), 6),
            "cost_complete": state.get("unpriced_calls", 0) == 0,
            "tokens_in": state.get("tokens_in", 0), "tokens_out": state.get("tokens_out", 0),
            "latency_s": round(total_seconds + (time.perf_counter() - started), 3),
            "trace": trace,
        })
        return {"status": status, "report": report,
                "trace": [{"node": "finalize", "seconds": round(time.perf_counter() - started, 3),
                           "status": status}]}

    def after_answer(state: AgentState) -> str:
        return "finalize" if state.get("error") else "verify"

    def after_verify(state: AgentState) -> str:
        return "finalize" if stop_reason(state, max_attempts, budget_usd) else "investigate"

    graph = StateGraph(AgentState)
    graph.add_node("answer", answer_node)
    graph.add_node("verify", verify_node)
    graph.add_node("investigate", investigate_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "answer")
    graph.add_conditional_edges("answer", after_answer, {"verify": "verify", "finalize": "finalize"})
    graph.add_conditional_edges("verify", after_verify,
                                {"investigate": "investigate", "finalize": "finalize"})
    graph.add_edge("investigate", "answer")
    graph.add_edge("finalize", END)

    return Agent(graph=graph.compile(checkpointer=checkpointer), ctx=ctx,
                 max_attempts=max_attempts, budget_usd=budget_usd,
                 has_checkpointer=checkpointer is not None)


# --- checkpointing -----------------------------------------------------------

def sqlite_checkpointer(path: str):
    """A SQLite-backed checkpointer, opt-in.

    What it writes: per-question state - the question, answers, quoted passages,
    findings, notes, cost. What it never writes: the document itself (see
    agent_state). That is still content derived from the document, so purge it
    with purge_thread when the user asks to incinerate - the same promise the
    UI's Incinerate button makes for the vector store.
    """
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def purge_thread(checkpointer, thread_id: str) -> None:
    """Delete everything a checkpointer holds for one thread."""
    checkpointer.delete_thread(thread_id)
