"""Execute the résumé path once, outside the UI.

This branch had never been run. Not "was lightly tested" — never executed, in a
project that has a benchmark harness and an eval harness, because it sat behind
a file upload and a button click. The first execution of a code path should not
happen while someone is watching you demo it.

Costs one API call. It asserts the shape of the result rather than its content,
because the content is a model judgement and asserting on it would make the
test flaky for no gain.

    python smoke_resume.py
"""
import sys

from dotenv import load_dotenv

load_dotenv()
from resume_analysis import run_resume_analysis

JOB_DESCRIPTION = """
Senior Python Engineer — Applied AI. You will build retrieval-augmented
generation pipelines over customer documents. Required: Python, LangChain or
LlamaIndex, vector databases (Chroma, Pinecone or pgvector), async concurrency,
prompt evaluation. Nice to have: Kubernetes, Terraform, Rust.
"""

RESUME = """
Dana Levi — Backend Engineer, Tel Aviv
EXPERIENCE
Backend Engineer, Fintech Co (2021-2026). Built document-processing services in
Python with FastAPI and PostgreSQL. Designed an OCR pipeline over scanned
statements using Tesseract. Introduced asyncio-based fan-out that cut batch
processing time by roughly half. Wrote the evaluation harness the team used to
compare extraction backends.
Junior Developer, Agency (2019-2021). Django web applications, REST APIs.
SKILLS Python, FastAPI, PostgreSQL, Docker, asyncio, pytest, Tesseract, Redis.
EDUCATION BSc Computer Science, Tel Aviv University.
"""

result = run_resume_analysis(JOB_DESCRIPTION, RESUME)

failures = []
if not isinstance(result.match_score, int) or not 0 <= result.match_score <= 100:
    failures.append(f"match_score out of range: {result.match_score!r}")
if not result.rationale.strip():
    failures.append("rationale is empty")
if not result.action_items:
    failures.append("no action items returned")
if len(result.missing_keywords) > 5:
    failures.append(f"missing_keywords exceeds the documented cap: {len(result.missing_keywords)}")

print(f"match_score      : {result.match_score}")
print(f"rationale        : {result.rationale[:160]}")
print(f"action_items     : {len(result.action_items)}")
for item in result.action_items:
    print(f"  - {item[:110]}")
print(f"missing_keywords : {result.missing_keywords}")

# The résumé never mentions a vector database, which the job description
# requires — so an analysis that sees the whole document should notice. This is
# a sanity check on the OUTPUT being about the input, not a correctness
# assertion; it prints rather than fails, because it is a model judgement.
text = " ".join(result.missing_keywords + [result.rationale]).lower()
expected = [k for k in ("vector", "chroma", "pinecone", "pgvector", "langchain",
                        "llamaindex", "rag", "retrieval") if k in text]
print(f"\nnoticed the retrieval/vector gap: {expected or 'NO — worth a look'}")

if failures:
    print("\nFAILURES:")
    for line in failures:
        print(f"  - {line}")
    sys.exit(1)
print("\nrésumé path executes and returns a well-formed result")
