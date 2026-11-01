"""Does the damage caveat make the model refuse questions it could answer?

The caveat (damaged_values.damage_caveat) tells the model: if a question depends on a
damaged value, say the document does not show it legibly. It exists to stop the model
inventing a date. The live eval then showed two answers that were intact in the text
yet came back as "does not show it legibly" - suggesting the caveat also buys
over-refusal. This runs the SAME questions with the caveat and without it, one attempt
each, no agent, so the caveat is the only thing that changes.

    python eval_caveat.py
    python eval_caveat.py --json caveat.json

Reads the same synthetic documents and uses the same scorer as eval_agent.py (so
'unclear' means the same: read those answers). Needs OPENAI_API_KEY; about 28 calls.

What each outcome means, per question kind:
  damaged     the value is gone. Without the caveat a model may invent one ('wrong'); with
              it should say so. If 'wrong' does not rise without the caveat, the caveat
              is not earning its keep on this sample.
  control     the answer is intact. Any drop in 'pass' WITH the caveat is over-refusal.
  relational  damage nearby, answer intact. Same reading as control.
"""
import argparse
import json
import os
import sys

from eval_agent import GOOD, build_suites, scorer_for


def run_arm(answerer, ctx, case, with_caveat):
    result = answerer(case["question"], ctx.text, [], ctx.caveat if with_caveat else "", "", None)
    return {"answer": result.answer, "outcome": scorer_for(case)(result.answer, case)}


def evaluate(answerer, groups, log=print):
    rows = []
    for ctx, cases in groups:
        for case in cases:
            row = {"question": case["question"], "kind": case.get("kind", "golden")}
            for name, flag in (("with_caveat", True), ("without_caveat", False)):
                row[name] = run_arm(answerer, ctx, case, flag)
            rows.append(row)
            log(f"  [{row['kind']:<10}] with={row['with_caveat']['outcome']:<7} "
                f"without={row['without_caveat']['outcome']}")
    return rows


def summarize(rows, out=print):
    out("\n=== the damage caveat, on vs off (one attempt, no agent)")
    summary = {}
    for kind in sorted({row["kind"] for row in rows}):
        group = [row for row in rows if row["kind"] == kind]
        counts = {arm: {} for arm in ("with_caveat", "without_caveat")}
        for row in group:
            for arm in counts:
                counts[arm][row[arm]["outcome"]] = counts[arm].get(row[arm]["outcome"], 0) + 1
        summary[kind] = counts
        out(f"  {kind:<10} n={len(group)}  with caveat {counts['with_caveat']}   "
            f"without {counts['without_caveat']}")
    refused = [row for row in rows if row["kind"] in ("control", "relational")
               and row["without_caveat"]["outcome"] in GOOD and row["with_caveat"]["outcome"] not in GOOD]
    helped = [row for row in rows if row["kind"] == "damaged"
              and row["without_caveat"]["outcome"] == "wrong" and row["with_caveat"]["outcome"] != "wrong"]
    summary["refused_only_with_caveat"], summary["fabricated_only_without_caveat"] = len(refused), len(helped)
    out(f"  answerable questions the caveat made the model fail: {len(refused)}")
    out(f"  damaged questions where dropping the caveat made it fabricate or echo: {len(helped)}")
    for row in rows:
        if row["with_caveat"]["outcome"] != row["without_caveat"]["outcome"] or row["kind"] == "relational":
            out(f"    [{row['kind']}] {row['question']}\n"
                f"      with caveat    [{row['with_caveat']['outcome']}]: {row['with_caveat']['answer'][:160]!r}\n"
                f"      without caveat [{row['without_caveat']['outcome']}]: {row['without_caveat']['answer'][:160]!r}")
    out("\nA handful of questions at temperature 0 is a small sample: it can show the effect exists, "
        "not how large it is.")
    return summary


def main(argv=None, answerer=None, out=print):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json", metavar="PATH")
    args = parser.parse_args(argv)
    if answerer is None:
        from dotenv import load_dotenv
        load_dotenv()
        if not os.environ.get("OPENAI_API_KEY"):
            out("OPENAI_API_KEY is not set (environment or .env).")
            return 2
        from agent_llm import make_openai_answerer
        answerer = make_openai_answerer()
    groups = build_suites()["damaged"]
    out(f"{sum(len(c) for _, c in groups) * 2} model calls.")
    rows = evaluate(answerer, groups, log=out)
    summarize(rows, out=out)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
        out(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
