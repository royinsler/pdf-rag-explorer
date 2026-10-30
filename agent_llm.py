"""The OpenAI-backed answer function for the agent graph.

Separate from agent_graph so the graph can be tested with a fake model and no
API key - the graph takes any callable with this module's signature.

Reliability lives here: bounded retries with exponential backoff per model, then
fallback to the next model in the list. Only after every model has failed does
the exception reach the graph, which turns it into status="failed" instead of
a traceback.
"""
import time
from typing import Callable, Sequence

from answering import DEFAULT_MODEL, _escape_braces, answer_from_full_document


def make_openai_answerer(models: Sequence[str] = (DEFAULT_MODEL,),
                         attempts_per_model: int = 2,
                         base_delay: float = 1.0,
                         sleep: Callable[[float], None] = time.sleep):
    """Build answer_fn(question, doc_text, history, caveat, notes, config).

    `sleep` is injectable so the backoff can be tested without waiting.
    """
    def answer(question, doc_text, history, caveat, notes, config):
        last_error = None
        for model in models:
            for attempt in range(attempts_per_model):
                try:
                    return answer_from_full_document(
                        question, doc_text, history,
                        caveat=_escape_braces(caveat), notes=notes,
                        model=model, config=config)
                except Exception as exc:        # noqa: BLE001 - rate limits, timeouts, 5xx
                    last_error = exc
                    if attempt + 1 < attempts_per_model:
                        sleep(base_delay * (2 ** attempt))
        raise last_error

    return answer
