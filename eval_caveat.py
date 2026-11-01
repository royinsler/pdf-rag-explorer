"""Does the damage caveat earn its place, and does a narrower wording do better?

The caveat (damaged_values.damage_caveat) tells the model to say a value is not legible
when a question depends on a damaged one. It exists to stop the model inventing or echoing
a date. The first live evals suggested a cost: a question whose answer was intact, with
damage nearby, came back as "does not show it legibly". This runs the SAME questions under
three prompts, with no agent and one attempt each, so the caveat is the only thing that
changes:

  none     no caveat at all
  current  damage_caveat - what the app sends today
  narrow   damage_caveat_narrow - a candidate: refuse only when the damaged value IS the
           answer, otherwise answer from intact text and do not repeat the debris

    python eval_caveat.py                      # 3 repeats per question
    python eval_caveat.py --repeats 5 --json caveat.json

Repeats matter: temperature 0 is not deterministic on the API (one refusal in an earlier run
did not reproduce), so a single sample can mislead. The report counts, per question and
arm, how many repeats landed in each outcome, and lists questions whose answer VARIED.

Same synthetic documents and scorer as eval_agent.py ('unclear' = read those answers).
Needs OPENAI_API_KEY; about 14 questions x 3 prompts x repeats calls.

How to read it, per kind:
  damaged     the value is gone: 'wrong' = invented or echoed it. A caveat earns its keep
              by lowering this.
  control     the answer is intact: any fall in 'pass' is over-refusal.
  relational  damage nearby, answer intact, and 'fail' also covers debris copied in.
Adopt the narrow wording only if it keeps 'damaged wrong' no higher than current AND raises
'relational pass'; a draw on a sample this small is not a reason to change the prompt.
"""
import argparse
import json
import os
import sys

from damaged_values import damage_caveat, damage_caveat_narrow
from eval_agent import build_suites, scorer_for

ARMS = {
    "none": lambda ctx: "",
    "current": lambda ctx: damage_caveat(ctx.damaged),
    "narrow": lambda ctx: damage_caveat_narrow(ctx.damaged),
}


def evaluate(answerer, groups, repeats=3, log=print):
    rows = []
    for ctx, cases in groups:
        for case in cases:
            row = {"question": case["question"], "kind": case.get("kind", "golden"),
                   **{arm: [] for arm in ARMS}}
            for arm, caveat_for in ARMS.items():
                caveat = caveat_for(ctx)
                for _ in range(repeats):
                    answer = answerer(case["question"], ctx.text, [], caveat, "", None).answer
                    row[arm].append({"answer": answer, "outcome": scorer_for(case)(answer, case)})
            rows.append(row)
            log(f"  [{row['kind']:<10}] " + "  ".join(
                f"{arm}={','.join(r['outcome'] for r in row[arm])}" for arm in ARMS))
    return rows


def tally(rows, kind, arm):
    counts = {}
    for row in rows:
        if row["kind"] == kind:
            for run in row[arm]:
                counts[run["outcome"]] = counts.get(run["outcome"], 0) + 1
    return dict(sorted(counts.items()))


def summarize(rows, out=print):
    out("\n=== caveat: none vs current vs narrow (outcomes summed over repeats)")
    summary = {}
    for kind in sorted({row["kind"] for row in rows}):
        n = sum(len(row[next(iter(ARMS))]) for row in rows if row["kind"] == kind)
        summary[kind] = {arm: tally(rows, kind, arm) for arm in ARMS}
        out(f"  {kind:<10} runs per arm={n}")
        for arm in ARMS:
            out(f"      {arm:<8} {summary[kind][arm]}")
    unstable = [(row, arm) for row in rows for arm in ARMS
                if len({run["outcome"] for run in row[arm]}) > 1]
    summary["unstable"] = len(unstable)
    out(f"\n  question/prompt pairs whose outcome varied between repeats: {len(unstable)}")
    for row, arm in unstable:
        out(f"    [{row['kind']}] {arm}: {[run['outcome'] for run in row[arm]]}  {row['question']}")
    out("\n  answers to read (relational, and any damaged question that was ever 'wrong'):")
    for row in rows:
        if row["kind"] == "relational" or any(run["outcome"] == "wrong" for arm in ARMS for run in row[arm]):
            out(f"    [{row['kind']}] {row['question']}")
            for arm in ARMS:
                out(f"      {arm:<8} {[ (r['outcome'], r['answer'][:90]) for r in row[arm][:1]]}")
    out("\nSmall sample: it can show an effect exists, not how large it is.")
    return summary


def main(argv=None, answerer=None, out=print):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repeats", type=int, default=3)
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
    calls = sum(len(c) for _, c in groups) * len(ARMS) * args.repeats
    out(f"{calls} model calls.")
    rows = evaluate(answerer, groups, repeats=args.repeats, log=out)
    summarize(rows, out=out)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
        out(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
