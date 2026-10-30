"""Ask one question of one PDF through the verification agent.

    python run_agent.py contract.pdf "מתי הפינוי?"
    python run_agent.py contract.pdf "..." --budget 0.02 --attempts 2
    python run_agent.py contract.pdf "..." --checkpoint-db agent.sqlite --thread q1
    python run_agent.py contract.pdf --resume q1 --checkpoint-db agent.sqlite

Needs OPENAI_API_KEY (from the environment or .env), like the Streamlit app.
`main` takes an injectable answerer so the whole path - real extraction, real
verification, real tools, real graph - runs in tests with no key.
"""
import argparse
import json
import os
import sys

from agent_graph import Agent, build_agent, purge_thread, sqlite_checkpointer
from agent_state import MAX_ATTEMPTS
from agent_tools import DocumentContext
from extraction import extract_document


def _print_report(report):
    print(report["answer"] or "(no answer)")
    print(f"\n[{report['status']}] attempts={report['attempts']} "
          f"cost=${report['cost_usd']:.4f}{'' if report['cost_complete'] else ' (incomplete: unpriced call)'} "
          f"latency={report['latency_s']}s tokens={report['tokens_in']}/{report['tokens_out']}")
    for warning in report["warnings"]:
        print(f"  ! {warning}")
    for finding in report["findings"]:
        print(f"  - {finding['kind']}: {finding['detail'][:80]!r}")
    print("trace:")
    for entry in report["trace"]:
        extras = {k: v for k, v in entry.items() if k not in ("node", "seconds")}
        print(f"  {entry['node']:<12} {entry['seconds']:>7.3f}s  {extras}")


def main(argv=None, answerer=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pdf")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--budget", type=float, default=0.05, help="USD per question (default 0.05)")
    parser.add_argument("--attempts", type=int, default=MAX_ATTEMPTS)
    parser.add_argument("--checkpoint-db", help="SQLite file; enables checkpointing and --resume")
    parser.add_argument("--thread", help="thread id for this question (default: random)")
    parser.add_argument("--resume", metavar="THREAD", help="continue an interrupted run")
    parser.add_argument("--purge", metavar="THREAD", help="delete a thread's checkpoints and exit")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)

    checkpointer = sqlite_checkpointer(args.checkpoint_db) if args.checkpoint_db else None
    if args.purge:
        if checkpointer is None:
            parser.error("--purge needs --checkpoint-db")
        purge_thread(checkpointer, args.purge)
        print(f"purged thread {args.purge}")
        return 0
    if not args.question and not args.resume:
        parser.error("give a question, or --resume THREAD")

    if answerer is None:
        from dotenv import load_dotenv
        load_dotenv()
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY is not set (environment or .env).", file=sys.stderr)
            return 2
        from agent_llm import make_openai_answerer
        answerer = make_openai_answerer()

    with open(args.pdf, "rb") as handle:
        file_bytes = handle.read()
    extraction = extract_document(file_bytes)
    if not extraction.usable:
        print(f"Could not use this PDF: {extraction.reason}.", file=sys.stderr)
        return 2

    ctx = DocumentContext.from_extraction(extraction, file_bytes=file_bytes)
    agent: Agent = build_agent(ctx, answerer, budget_usd=args.budget,
                               max_attempts=args.attempts, checkpointer=checkpointer)
    report = agent.resume(args.resume) if args.resume else agent.run(args.question, thread_id=args.thread)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        _print_report(report)
    return 0 if report["status"] in ("verified",) else 1


if __name__ == "__main__":
    sys.exit(main())
