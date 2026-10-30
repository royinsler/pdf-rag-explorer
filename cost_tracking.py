"""Token and cost accounting for one LLM call.

Kept deliberately small. The agent graph creates a fresh UsageCallback for every
node that calls a model and adds the result to a running total in graph state.
Per-call rather than a shared tracker, for two reasons: it is trivially
thread-safe, and the total then lives in the checkpointed state, so a resumed
run continues the same bill instead of restarting at zero.

Prices are CONFIGURATION, not facts the code can know. They change, and the
only number that matters is the one on your provider invoice. The figure below
was read from a third-party aggregator (llmreference.com, page refreshed
2026-10-17), not from OpenAI's own pricing page - check it before quoting a
cost anywhere.
"""
from typing import Dict, Optional, Tuple

from langchain_core.callbacks import BaseCallbackHandler

# USD per 1,000,000 tokens: (input, output).
PRICES_PER_MTOK: Dict[str, Tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
}


def price_tokens(model: str, input_tokens: int, output_tokens: int,
                 prices: Dict[str, Tuple[float, float]] = PRICES_PER_MTOK) -> Optional[float]:
    """Cost in USD, or None when the model has no entry.

    None rather than 0.0 on purpose: reporting "$0.00" for a model we cannot
    price would read as "free" and quietly defeat the budget guard. Callers
    count unpriced calls and say so.
    """
    key = _match_model(model, prices)
    if key is None:
        return None
    in_price, out_price = prices[key]
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


def _match_model(model: str, prices) -> Optional[str]:
    """Resolve a reported model name to a price key.

    APIs report dated snapshots ("gpt-4o-mini-2024-07-18") for the alias you
    asked for, so exact matching alone would mark every real call unpriced.
    Longest prefix wins, so "gpt-4o-mini-..." never falls into a "gpt-4o" entry.
    """
    if model in prices:
        return model
    candidates = [k for k in prices if model and model.startswith(k)]
    return max(candidates, key=len) if candidates else None


class UsageCallback(BaseCallbackHandler):
    """Collects token usage from every LLM call made while it is attached.

    Reads usage_metadata off the returned message, which is where
    langchain-openai puts it for both plain and structured-output calls. One
    record per call, so each is priced by the model that actually served it -
    which matters when the answerer falls back to a different model mid-node.
    """

    def __init__(self):
        self.records = []         # [(model, input_tokens, output_tokens)] per call

    def on_llm_end(self, response, **kwargs):
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None) or {}
                meta = getattr(message, "response_metadata", None) or {}
                self.records.append((
                    meta.get("model_name") or meta.get("model") or "",
                    int(usage.get("input_tokens", 0) or 0),
                    int(usage.get("output_tokens", 0) or 0),
                ))

    @property
    def calls(self) -> int:
        return len(self.records)

    @property
    def input_tokens(self) -> int:
        return sum(r[1] for r in self.records)

    @property
    def output_tokens(self) -> int:
        return sum(r[2] for r in self.records)

    def cost_usd(self, prices=PRICES_PER_MTOK) -> Optional[float]:
        """Total for every call seen, or None if ANY of them could not be priced.

        All-or-nothing: a partial sum presented as the cost would understate it.
        """
        total = 0.0
        for model, tokens_in, tokens_out in self.records:
            cost = price_tokens(model, tokens_in, tokens_out, prices)
            if cost is None:
                return None
            total += cost
        return total
