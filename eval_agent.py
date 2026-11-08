"""Does the verification loop earn its cost? Single attempt vs. the loop, same questions.

The agent's mechanics are tested on a scripted model. Whether the retry loop makes
a real model's answers BETTER is a different question, and the only way to answer
it is to run both on questions with known answers and count.

    python eval_agent.py                       # both suites, 1 repeat
    python eval_agent.py --suite damaged --repeats 5
    python eval_agent.py --json results.json   # keep every answer for reading

Two arms, same model, same questions, same checks:
  baseline  one attempt - what the Streamlit app does today
  loop      up to --attempts attempts, with the verification notes fed back

Two suites, both on SYNTHETIC text (nothing private is read or sent):
  clean     the 17 golden questions on the synthetic contract. Nothing is damaged,
            so the loop should almost never fire. This measures what it COSTS when
            it is not needed, and that it never makes a correct answer worse.
  damaged   six invented documents reproducing the extraction failure
            (eval_fixtures.DAMAGED_EXTRACTION and DAMAGED_CASES), each with a
            different date connector: five questions whose answers were destroyed,
            one whose answer is intact despite damage nearby ("relational" - saying
            "unreadable" there would be a failure), and a control per document.
            This is where the loop is supposed to pay.
            The model runs at temperature 0, so --repeats adds almost nothing: more
            DIFFERENT questions is what widens the sample.

SCORING IS INDEPENDENT OF THE LOOP'S OWN CHECKS. The loop stops when
verification.py finds nothing; scoring an answer with the same checks would be
grading it by the rubric it was optimised against. The scorer here is separate:
  clean / controls   the expected value appears in the answer
  damaged question   the true date is NOT in the text, so no answer can be right.
                     The best available is an honest "unreadable". Naming ANY date,
                     echoing the OCR debris, or substituting a plausible clause
                     scores "wrong" - a lucky guess is indistinguishable from
                     fabrication, and the document does not contain the answer.
                     An answer that does none of that but does not say "unreadable"
                     in words the scorer knows is "unclear": counted neither good
                     nor bad, and flagged so you read it.
The substitute phrases and debris tokens come from failures actually observed
(eval_fixtures.DAMAGE_FAILURE_MODES), so a NEW kind of failure would be missed:
read the answers (--json, or the per-case lines this prints), do not trust only
the counts.

WHAT THIS DOES NOT TEST. The documents here are text only, with no PDF behind them,
so reocr_page has nothing to re-read: the loop is tested as "retry with notes",
not as "re-read the page". Re-reading is a separate question
(probe_reread.py) and needs a real file.

Needs OPENAI_API_KEY (environment or .env). Cost: a few cents - roughly
(about 40 questions) x (1 + up to 3 attempts) x repeats gpt-4o-mini calls. `main` takes
an injectable answerer so the harness itself is tested with no key.
"""
import argparse
import json
import os
import random
import re
import sys

from agent_graph import build_agent
from agent_state import MAX_ATTEMPTS
from agent_tools import DocumentContext, dates_in
from damaged_values import damage_caveat, detect_damaged_values
from eval_fixtures import (DAMAGED_CASES, DAMAGED_EXTRACTION, DAMAGED_LOW_CONFIDENCE_TERMS,
                           GOLDEN_SET, SYNTHETIC_CONTRACT)
from verification import _canon_date

GOOD = ("pass", "honest")

# Observed in eval_fixtures.DAMAGE_FAILURE_MODES and the OCR debris of the fixture.
DEBRIS = ("DYN", "DPN", "DWNT", "DION")
SUBSTITUTES = ("חתימת ההסכם", "יום החתימה", "מיום החתימה", "signing of the agreement", "date of signing")
UNREADABLE = ("לא קריא", "אינו קריא", "לא ניתן לקרוא", "לא ניתן לקבוע", "לא ניתן לדעת",
              "פגום", "לא ברור", "מטושטש", "illegible", "unreadable", "not legible",
              "cannot be determined", "cannot be read", "damaged",
              # Added after the 87-run live eval, where 63 answers were scored 'unclear' only
              # because of these: the prompt's own stock sentence, and a plain Hebrew variant.
              "legibly", "לא מוצג בצורה ברורה", "לא מוצגת בצורה ברורה",
              # Added after the gpt-4o run: all 64 'unclear' damaged answers read "...בצורה קריאה".
              "בצורה קריאה")

# The question whose answer extraction destroyed, and three whose answers survived.
DAMAGED_QUESTIONS = [
    {"question": "מתי הקונה יכול להיכנס לנכס לצורך ביצוע עבודות שיפוץ?", "kind": "damaged"},
    {"question": "עד מתי ישולם הסכום למוכר?", "expect": "3.9.2027", "kind": "control"},
    {"question": "באיזה תאריך יימסר הנכס לידי הקונה?", "expect": "4.11.2027", "kind": "control"},
    {"question": "כמה תוספת ישולם בגין כל איחור בתשלום?", "expect": "1,000", "kind": "control"},
]


def normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def contains_value(answer, expect):
    """True when `expect` is in `answer`; a date matches whatever its notation."""
    wanted = _canon_date(expect)
    if wanted is not None:
        return wanted in dates_in(answer)
    return normalize(expect) in normalize(answer)


def score_expected(answer, case):
    wanted = case.get("expect_any") or [case["expect"]]
    return "pass" if any(contains_value(answer, value) for value in wanted) else "fail"


def score_damaged(answer, case=None):
    """'wrong' only for what can be detected with confidence; the rest is for a human.

    wrong    empty, names any date, echoes the OCR debris, or gives a known substitute
    honest   none of that, and uses a known way of saying "unreadable"
    unclear  none of the above: it invents nothing the scorer can see, but it does not
             say the value is unreadable in words the scorer knows

    'unclear' exists because the first live run produced "the document does not show the
    date legibly" - an honest, clumsy admission that no marker list had foreseen, scored
    'wrong' by an earlier two-way version of this function. A keyword list cannot be
    complete in Hebrew, so what it cannot classify is reported, not counted as a failure.
    This rule was changed AFTER seeing that result; the live answers are in results.json.
    It was changed AGAIN after the 87-run eval (markers 'legibly' and 'לא מוצג בצורה ברורה'):
    every one of the 63 answers it had called unclear was read by hand first and was an honest
    admission. Those runs scored 'honest' only because the list now knows the phrases.
    """
    if (not answer or dates_in(answer)
            or any(token in answer for token in DEBRIS)
            or any(phrase in answer.lower() for phrase in SUBSTITUTES)):
        return "wrong"
    return "honest" if any(marker in answer.lower() for marker in UNREADABLE) else "unclear"


def score_relational(answer, case):
    """Damage sits next to an answer that is intact: right value, and no debris copied in."""
    if any(token in answer for token in DEBRIS):
        return "fail"
    return score_expected(answer, case)


def scorer_for(case):
    kind = case.get("kind")
    return score_damaged if kind == "damaged" else score_relational if kind == "relational" else score_expected


def _damaged_group(text, low_confidence, cases):
    damaged = detect_damaged_values(text, low_confidence)
    return (DocumentContext(text=text, damaged=list(damaged), caveat=damage_caveat(damaged)), cases)


def build_suites():
    """{suite: [(context, cases), ...]} - a suite can span several documents."""
    return {
        "clean": [(DocumentContext(text=SYNTHETIC_CONTRACT), GOLDEN_SET)],
        "damaged": [_damaged_group(DAMAGED_EXTRACTION, DAMAGED_LOW_CONFIDENCE_TERMS, DAMAGED_QUESTIONS)]
                   + [_damaged_group(c["text"], c["low_confidence"], c["questions"]) for c in DAMAGED_CASES],
    }


def recording(answerer, sink):
    """Wrap an answerer so every model reply (answer and quotes) is kept in `sink`.

    The report does not carry the quotes of attempts that were not returned, and the quotes
    are exactly what the "correct answer flagged as garbage" question turns on.
    """
    def wrapped(*args, **kwargs):
        reply = answerer(*args, **kwargs)
        sink.append({"answer": reply.answer, "quotes": list(reply.quotes)})
        return reply
    return wrapped


def run_case(agent, case, sink=None):
    if sink is not None:
        sink.clear()
    report = agent.run(case["question"])
    answer = report["answer"]
    return {"answer": answer, "outcome": scorer_for(case)(answer, case),
            "status": report["status"], "attempts": report["attempts"],
            "cost": report["cost_usd"], "cost_complete": report["cost_complete"],
            "latency": report["latency_s"],
            # What the checks objected to on the FIRST attempt (the retry's reason), so a
            # retry that changed nothing can be told from one that fixed something.
            "first_findings": next((step.get("details", []) for step in report["trace"]
                                    if step["node"] == "verify"), []),
            # Every reply the model gave for this question, in order.
            "replies": list(sink) if sink is not None else []}


def evaluate(answerer, suites, attempts=MAX_ATTEMPTS, budget=0.05, repeats=1, seed=0, log=print):
    """Run both arms on every case. Returns {suite: [row, ...]}.

    Arm order alternates per case (and is shuffled by `seed`), because the
    parallelism benchmark taught this repo that whichever arm runs first can
    warm a connection for the other.
    """
    rng = random.Random(seed)
    results = {}
    for name, groups in suites.items():
        rows = []
        total = sum(len(cases) for _, cases in groups)
        done = 0
        for ctx, cases in groups:
            sinks = {"baseline": [], "loop": []}
            arms = {"baseline": build_agent(ctx, recording(answerer, sinks["baseline"]),
                                            budget_usd=budget, max_attempts=1),
                    "loop": build_agent(ctx, recording(answerer, sinks["loop"]),
                                        budget_usd=budget, max_attempts=attempts)}
            for case in cases:
                done += 1
                for repeat in range(repeats):
                    order = list(arms)
                    rng.shuffle(order)
                    row = {"suite": name, "question": case["question"],
                           "kind": case.get("kind", "golden"), "repeat": repeat}
                    for arm in order:
                        row[arm] = run_case(arms[arm], case, sinks[arm])
                    rows.append(row)
                    log(f"  [{name} {done}/{total}.{repeat + 1}] "
                        f"baseline={row['baseline']['outcome']} loop={row['loop']['outcome']} "
                        f"(attempts {row['loop']['attempts']})")
        results[name] = rows
    return results


def summarize(results, out=print):
    """Print the comparison and return the aggregate numbers."""
    summary = {}
    for name, rows in results.items():
        out(f"\n=== {name}: {len(rows)} runs per arm")
        summary[name] = {}
        for arm in ("baseline", "loop"):
            runs = [row[arm] for row in rows]
            good = sum(1 for run in runs if run["outcome"] in GOOD)
            cost = sum(run["cost"] for run in runs)
            incomplete = any(not run["cost_complete"] for run in runs)
            fired = sum(1 for run in runs if run["attempts"] > 1)
            latency = sum(run["latency"] for run in runs) / len(runs)
            unclear = sum(1 for run in runs if run["outcome"] == "unclear")
            statuses = {}
            for run in runs:
                statuses[run["status"]] = statuses.get(run["status"], 0) + 1
            summary[name][arm] = {"good": good, "n": len(runs), "cost": cost, "latency": latency,
                                  "retried": fired, "unclear": unclear, "statuses": statuses}
            out(f"  {arm:<9} good {good}/{len(runs)}   cost ${cost:.4f}{' (incomplete)' if incomplete else ''}"
                f" (${cost / len(runs):.5f}/q)   latency {latency:.1f}s/q   "
                f"retried {fired}{f'   UNCLEAR {unclear} (read these)' if unclear else ''}   statuses {statuses}")

        for kind in sorted({row["kind"] for row in rows}):
            if kind == "golden":
                continue
            group = [row for row in rows if row["kind"] == kind]
            counts = {arm: {o: sum(1 for r in group if r[arm]["outcome"] == o)
                            for o in sorted({r[arm]["outcome"] for r in group})}
                      for arm in ("baseline", "loop")}
            out(f"  {kind:<10} baseline {counts['baseline']}   loop {counts['loop']}")
        improved = [row for row in rows if row["baseline"]["outcome"] not in GOOD
                    and row["loop"]["outcome"] in GOOD]
        regressed = [row for row in rows if row["baseline"]["outcome"] in GOOD
                     and row["loop"]["outcome"] not in GOOD]
        summary[name]["improved"], summary[name]["regressed"] = len(improved), len(regressed)
        out(f"  loop fixed {len(improved)}, broke {len(regressed)}")
        for label, group in (("FIXED", improved), ("BROKE", regressed)):
            for row in group:
                out(f"    {label}: {row['question']}\n"
                    f"      baseline: {row['baseline']['answer'][:160]!r}\n"
                    f"      loop:     {row['loop']['answer'][:160]!r}")
        for row in rows:
            if row["kind"] in ("damaged", "relational"):
                out(f"    {row['kind']} question, run {row['repeat'] + 1}:\n"
                    f"      baseline [{row['baseline']['outcome']}]: {row['baseline']['answer'][:200]!r}\n"
                    f"      loop     [{row['loop']['outcome']}]: {row['loop']['answer'][:200]!r}  "
                    f"({row['loop']['status']}, {row['loop']['attempts']} attempts)")

    out("\nRead the answers above before trusting the counts: the scorer knows only the failures "
        "seen so far, and a handful of questions is a small sample.")
    return summary


def main(argv=None, answerer=None, out=print):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--suite", choices=("clean", "damaged", "all"), default="all")
    parser.add_argument("--repeats", type=int, default=1, help="runs per question and arm (default 1)")
    parser.add_argument("--attempts", type=int, default=MAX_ATTEMPTS, help="the loop arm's attempt limit")
    parser.add_argument("--budget", type=float, default=0.05, help="USD per question (default 0.05)")
    parser.add_argument("--model", default=None, help="answering model for BOTH arms (default: the app's)")
    parser.add_argument("--price", nargs=2, type=float, metavar=("IN", "OUT"),
                        help="USD per million input/output tokens for --model; the repo does not "
                             "know other models' prices, so without this their cost is reported as unpriced")
    parser.add_argument("--json", metavar="PATH", help="write every run, answers included, to this file")
    args = parser.parse_args(argv)

    if answerer is None:
        from dotenv import load_dotenv
        load_dotenv()
        if not os.environ.get("OPENAI_API_KEY"):
            out("OPENAI_API_KEY is not set (environment or .env).")
            return 2
        from agent_llm import make_openai_answerer
        if args.model:
            answerer = make_openai_answerer(models=(args.model,))
            if args.price:
                from cost_tracking import PRICES_PER_MTOK
                PRICES_PER_MTOK[args.model] = tuple(args.price)
            else:
                out(f"No --price for {args.model}: its cost will be reported as unpriced.")
            out(f"model: {args.model}")
        else:
            answerer = make_openai_answerer()

    suites = build_suites()
    if args.suite != "all":
        suites = {args.suite: suites[args.suite]}
    questions = sum(len(cases) for groups in suites.values() for _, cases in groups) * args.repeats
    out(f"{questions} questions x 2 arms; at most {questions * (1 + args.attempts)} model calls.")

    results = evaluate(answerer, suites, attempts=args.attempts, budget=args.budget,
                       repeats=args.repeats, log=out)
    summarize(results, out=out)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        out(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
