"""State schema for the verification agent graph.

Two rules shape what is in here.

1. The document is NOT in the state. State is what a checkpointer writes to
   disk, and this project's promise is that an uploaded document is never
   persisted (see README, "Stateless & Privacy-First"). The document lives in a
   DocumentContext captured by the graph's nodes instead. What the state does
   carry - the question, the answer, quoted passages, findings - is derived from
   the document and WOULD be written by a checkpointer, which is why
   checkpointing is opt-in and agent_graph.purge_thread exists.

2. Fields written by more than one node, or accumulated across retries, have an
   explicit reducer. Without one LangGraph silently overwrites, and a retry loop
   that overwrites its own history cannot report what it tried.
"""
import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

# Bump when a field is renamed or changes meaning. A checkpoint written under an
# older schema is then recognisable instead of failing somewhere downstream.
SCHEMA_VERSION = 1

# Upper bound on answer attempts per question (the first try plus retries).
MAX_ATTEMPTS = 3

STATUS_VERIFIED = "verified"            # every check passed
STATUS_DEGRADED = "degraded"            # best answer returned WITH unresolved findings
STATUS_BUDGET = "budget_exceeded"       # stopped retrying to stay inside the budget
STATUS_FAILED = "failed"                # the model could not be reached at all


class AgentState(TypedDict, total=False):
    schema_version: int
    question: str
    history: List[Dict[str, str]]       # [{"user": ..., "bot": ...}], oldest first

    attempt: int                        # answer attempts made so far
    answer: str                         # latest answer
    quotes: List[str]                   # latest verbatim quotes the model offered
    findings: List[Dict[str, str]]      # latest verification findings; [] = clean
    error: Optional[str]                # set when the model call failed outright

    # Accumulated across the loop, so each has a reducer.
    attempts_log: Annotated[List[Dict[str, Any]], operator.add]   # every attempt, for best-of
    notes: Annotated[List[str], operator.add]                     # evidence the tools gathered
    trace: Annotated[List[Dict[str, Any]], operator.add]          # per-node timing and detail

    # Cost. Totals live here (not in a tracker object) so a resumed run
    # continues the same bill.
    cost_usd: float
    last_call_cost: float
    tokens_in: int
    tokens_out: int
    unpriced_calls: int

    status: str
    report: Dict[str, Any]              # the finished, UI-shaped result
