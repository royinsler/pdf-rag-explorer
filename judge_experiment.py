"""Offline experiment: can an LLM judge tell correct from wrong on the ambiguous cases?

The ambiguous cases are answers that are clean on their own, whose supporting quote holds OCR
debris, and that carry no verified figure - so the deterministic rule (verification.split_debris)
cannot tell the correct ones ("ארבע שעות") from the wrong ones ("המסירה בפועל.") and flags both.
The truth for each comes from the eval's own scorer, which knows the expected value.

    python judge_experiment.py out3.json --dry-run     # list the cases, no API calls
    python judge_experiment.py out3.json               # judge them (needs OPENAI_API_KEY)
    python judge_experiment.py out3.json --repeats 3   # repeat each judgement

The judge sees ONLY the question, the answer and the quotes - never the document and never the
expected value. Integrate it into the product only if every correct case is judged correct and
every wrong one incorrect, in every repeat. A model checking a model is a weaker guarantee than the
deterministic checks, so anything short of a clean separation means: do not integrate.
"""
import argparse
import json
import sys

from pydantic import BaseModel, Field

from eval_agent import build_suites, scorer_for
from verification import find_ocr_garbage_echoes, split_debris

GOOD = {"pass", "honest"}

JUDGE_PROMPT = (
    "You check an answer against the passages quoted as its evidence. You are given a question, "
    "an answer, and quotes copied from a contract. Some quotes contain garbled OCR text (e.g. "
    "DWNT, DION, WN); ignore the garbled tokens themselves. Decide whether the answer directly "
    "answers the question, as asked, AND is supported by the quotes. An answer that is fluent "
    "but answers a different question, or names a different event than the one asked about, is "
    "incorrect. If the quotes do not settle it, say cannot_tell."
)


class Verdict(BaseModel):
    verdict: str = Field(description="correct, incorrect, or cannot_tell")
    reason: str = Field(description="one short sentence")


def ambiguous_cases(results):
    """[(question, answer, quotes, truth_is_correct, arm)] for the ambiguous final answers."""
    index = {}
    for groups in build_suites().values():
        for ctx, cases in groups:
            for case in cases:
                index[case["question"]] = (ctx, case)
    found, seen = [], set()
    for rows in results.values():
        for row in rows:
            if row["question"] not in index:
                continue
            ctx, case = index[row["question"]]
            for arm in ("baseline", "loop"):
                run = row[arm]
                answer = run["answer"]
                reply = next((r for r in reversed(run.get("replies", [])) if r["answer"] == answer), None)
                if not reply or find_ocr_garbage_echoes(answer, ctx.text):
                    continue
                hard, _ = split_debris(answer, reply["quotes"], ctx.text)
                if not hard:
                    continue
                key = (row["question"], answer, tuple(reply["quotes"]))
                if key in seen:
                    continue
                seen.add(key)
                truth = scorer_for(case)(answer, case) in GOOD
                found.append((row["question"], answer, reply["quotes"], truth, arm))
    return found


def judge_once(question, answer, quotes, model="gpt-4o-mini"):
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model_name=model, temperature=0).with_structured_output(Verdict)
    text = f"Question: {question}\nAnswer: {answer}\nQuotes:\n" + "\n".join(f"- {q}" for q in quotes)
    return llm.invoke([("system", JUDGE_PROMPT), ("human", text)]).verdict.strip().lower()


def main(argv=None, judge=judge_once, out=print):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("json_path")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args(argv)
    with open(args.json_path, encoding="utf-8") as handle:
        cases = ambiguous_cases(json.load(handle))
    good = sum(1 for c in cases if c[3])
    out(f"{len(cases)} ambiguous cases: {good} correct, {len(cases) - good} wrong (truth from the scorer)")
    wrong_calls = 0
    for question, answer, quotes, truth, arm in cases:
        if args.dry_run:
            out(f"  [{'correct' if truth else 'WRONG '}] {answer!r}  <- {question}")
            continue
        verdicts = [judge(question, answer, quotes) for _ in range(args.repeats)]
        ok = all(v == ("correct" if truth else "incorrect") for v in verdicts)
        wrong_calls += not ok
        out(f"  [{'ok  ' if ok else 'MISS'}] truth={'correct' if truth else 'wrong  '} "
            f"judge={','.join(verdicts)}  {answer!r}")
    if not args.dry_run:
        out(f"\n{len(cases) - wrong_calls}/{len(cases)} judged as the truth says"
            + ("  -> clean separation" if not wrong_calls else "  -> NOT clean: do not integrate"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
