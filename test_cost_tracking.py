"""Tests for token and cost accounting.

The failure these guard against is the quiet one: a cost figure that looks fine
and is wrong - "$0.00" for a model nobody priced, or a partial sum shown as the
total. Needs langchain-core but no API key: python3 test_cost_tracking.py
"""
import sys

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from cost_tracking import UsageCallback, price_tokens

results = []


def check(name, actual, expected):
    ok = actual == expected
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"         expected {expected!r}\n         got      {actual!r}")


def llm_result(model, tokens_in, tokens_out):
    message = AIMessage(content="x",
                        usage_metadata={"input_tokens": tokens_in, "output_tokens": tokens_out,
                                        "total_tokens": tokens_in + tokens_out},
                        response_metadata={"model_name": model})
    return LLMResult(generations=[[ChatGeneration(message=message)]])


# --- price_tokens ------------------------------------------------------------
check("1M in + 1M out at the configured gpt-4o-mini rate",
      round(price_tokens("gpt-4o-mini", 1_000_000, 1_000_000), 6), 0.75)
# APIs report dated snapshots for the alias you asked for. Exact-match-only
# would mark every real call unpriced.
check("a dated snapshot resolves to its base model",
      round(price_tokens("gpt-4o-mini-2024-07-18", 1_000_000, 0), 6), 0.15)
check("an unpriced model is None, never 0.0", price_tokens("some-new-model", 1000, 1000), None)
check("a shorter model name does not match a longer price key",
      price_tokens("gpt-4o", 1000, 1000), None)
check("an empty model name is unpriced", price_tokens("", 1000, 1000), None)

# --- UsageCallback -----------------------------------------------------------
cb = UsageCallback()
check("no calls costs nothing and says zero calls", (cb.calls, cb.cost_usd()), (0, 0.0))

cb.on_llm_end(llm_result("gpt-4o-mini-2024-07-18", 1000, 100))
check("one call is priced from its reported usage", round(cb.cost_usd(), 8), 0.00021)
check("tokens are summed", (cb.input_tokens, cb.output_tokens), (1000, 100))

cb.on_llm_end(llm_result("gpt-4o-mini", 2000, 200))
check("a second call adds to the total", round(cb.cost_usd(), 8), 0.00063)
check("call count follows", cb.calls, 2)

# All-or-nothing. A partial sum shown as "the cost" would understate it, and the
# budget guard would be comparing against a number that is too small.
cb.on_llm_end(llm_result("brand-new-model", 500, 50))
check("one unpriced call makes the whole total None", cb.cost_usd(), None)

# A message with no usage metadata at all (some providers, some failure modes).
bare = UsageCallback()
bare.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="x"))]]))
check("a call with no usage is recorded as one call with unknown model",
      (bare.calls, bare.input_tokens, bare.cost_usd()), (1, 0, None))

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
