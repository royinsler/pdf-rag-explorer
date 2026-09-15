"""
RAG evaluation harness.

Measures retrieval and answer quality across indexing configurations, so
chunk-size decisions can be made from data instead of intuition.

WHY IT'S BUILT THIS WAY
-----------------------
* Synthetic corpus (eval_fixtures.py). No private data leaves the machine, the
  fixture can be committed, and — critically — we know exactly which sentence
  answers each question, which is what makes precise retrieval measurement
  possible at all.

* Retrieval and generation are scored SEPARATELY. A single blended "is the
  answer good" score can't tell you whether to fix your chunking or your
  prompt. "Retrieval hit" asks whether the answer-bearing text was even put in
  front of the model; "answer correct" asks what the model did with it. A
  config with high retrieval and low correctness is a generation problem; the
  reverse is a retrieval problem.

* Two of the three metrics need no LLM. Substring checks against known ground
  truth are deterministic, free, and can't themselves hallucinate. Only
  groundedness needs a judge, because "is this claim supported by that text"
  has no cheap mechanical answer.

* A deliberately bad config is included as a control. If a 200-char/k=2 index
  scores the same as a tuned one, the metric has no signal and the numbers
  mean nothing. Validating the instrument matters before trusting its readings.

COST / RUNTIME
--------------
5 configs x 12 questions x 2 calls ~= 120 gpt-4o-mini calls plus 5 embedding
passes. A few cents, a couple of minutes. Set LANGSMITH_TRACING=true beforehand
to capture every run in LangSmith (traces will include the synthetic document).

USAGE
-----
    .venv/bin/python eval_rag.py
"""
import asyncio
import os
import re
import sys
import time
import uuid

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from eval_fixtures import SYNTHETIC_CONTRACT, GOLDEN_SET

load_dotenv()

# (label, chunk_size, overlap, k) - the four presets app.py actually ships,
# plus a deliberately poor control to prove the metrics discriminate.
CONFIGS = [
    ("control-tiny",      200,   0,  2),
    ("Technical/Code",    600,  50, 10),
    ("Legal/Book",        800, 200,  7),
    ("General",          1000, 100,  5),
    ("Resume",           1500, 100,  3),
]

MAX_CONCURRENT_CALLS = 5
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"


def normalize(text: str) -> str:
    """Collapse whitespace so matches survive chunking and line wrapping."""
    return re.sub(r"\s+", " ", text).strip().lower()


def build_chain(chunk_size: int, overlap: int, k: int):
    """Index the corpus at one configuration and return (chain, vectorstore)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    splits = splitter.split_documents([Document(page_content=SYNTHETIC_CONTRACT)])

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        collection_name=f"eval_{uuid.uuid4().hex[:12]}",
    )

    llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer the question using ONLY the context below. If the context does "
         "not contain the answer, say you don't know. Be concise and quote exact "
         "figures where present.\n\n{context}"),
        ("human", "{input}"),
    ])
    chain = create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={"k": k}),
        create_stuff_documents_chain(llm, qa_prompt),
    )
    return chain, vectorstore, len(splits)


async def judge_grounded(judge_llm, question: str, answer: str, context: str) -> bool:
    """LLM-as-judge: is every claim in the answer supported by the context?

    Deliberately narrow. It is NOT asked whether the answer is correct - only
    whether it is supported by what was retrieved, which is the one question a
    judge can answer without ground truth of its own.
    """
    verdict = await judge_llm.ainvoke(
        "You are grading whether an answer is grounded in the provided context.\n"
        "Reply with exactly one word: GROUNDED or UNSUPPORTED.\n"
        "Reply GROUNDED only if every factual claim in the answer appears in the "
        "context. An answer that correctly says it does not know is GROUNDED.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER: {answer}"
    )
    return "GROUNDED" in verdict.content.upper()


async def evaluate_config(label: str, chunk_size: int, overlap: int, k: int):
    chain, vectorstore, n_chunks = build_chain(chunk_size, overlap, k)
    judge_llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)

    async def run_case(case):
        async with semaphore:
            result = await chain.ainvoke({"input": case["question"]})
            answer = result["answer"]
            context = "\n".join(d.page_content for d in result.get("context", []))

            expects = case["expect"] if isinstance(case["expect"], list) else [case["expect"]]
            retrieval_hit = normalize(case["needle"]) in normalize(context)
            answer_correct = any(normalize(e) in normalize(answer) for e in expects)
            grounded = await judge_grounded(judge_llm, case["question"], answer, context)

            return {
                "question": case["question"],
                "kind": case.get("kind", "lookup"),
                "retrieval_hit": retrieval_hit,
                "answer_correct": answer_correct,
                "grounded": grounded,
            }

    start = time.time()
    rows = await asyncio.gather(*(run_case(c) for c in GOLDEN_SET))
    elapsed = time.time() - start

    try:
        vectorstore.delete_collection()
    except Exception:
        pass

    n = len(rows)
    return {
        "config": label,
        "chunk": chunk_size,
        "overlap": overlap,
        "k": k,
        "chunks_indexed": n_chunks,
        "ctx_chars": chunk_size * k,
        "retrieval_hit": sum(r["retrieval_hit"] for r in rows) / n,
        "answer_correct": sum(r["answer_correct"] for r in rows) / n,
        "grounded": sum(r["grounded"] for r in rows) / n,
        "seconds": elapsed,
        "by_kind": {
            kind: sum(r["answer_correct"] for r in rows if r["kind"] == kind) /
                  max(1, sum(1 for r in rows if r["kind"] == kind))
            for kind in sorted({r["kind"] for r in rows})
        },
        "failures": [f"[{r['kind']}] {r['question']}" for r in rows if not r["answer_correct"]],
    }


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)

    print(f"Corpus: {len(SYNTHETIC_CONTRACT):,} chars | {len(GOLDEN_SET)} golden questions")
    print(f"Tracing: {'ON - runs sent to LangSmith' if os.getenv('LANGSMITH_TRACING', '').lower() == 'true' else 'off'}\n")

    results = []
    for label, chunk_size, overlap, k in CONFIGS:
        print(f"  running {label:<16} (chunk {chunk_size}, k {k}) ...", flush=True)
        results.append(await evaluate_config(label, chunk_size, overlap, k))

    header = (
        f"\n| {'Config':<16} | {'Chunk':>5} | {'k':>2} | {'Idx':>3} | {'Ctx':>6} | "
        f"{'Retrieval':>9} | {'Correct':>7} | {'Grounded':>8} | {'Secs':>5} |"
    )
    print(header)
    print("|" + "-" * (len(header) - 3) + "|")
    for r in results:
        print(
            f"| {r['config']:<16} | {r['chunk']:>5} | {r['k']:>2} | {r['chunks_indexed']:>3} | "
            f"{r['ctx_chars']:>6} | {r['retrieval_hit']:>8.0%} | {r['answer_correct']:>6.0%} | "
            f"{r['grounded']:>7.0%} | {r['seconds']:>5.1f} |"
        )

    kinds = sorted({k for r in results for k in r["by_kind"]})
    print(f"\n| {'Config':<16} |" + "".join(f" {k:>10} |" for k in kinds))
    print("|" + "-" * (18 + 13 * len(kinds)) + "|")
    for r in results:
        print(f"| {r['config']:<16} |" + "".join(f" {r['by_kind'].get(k, 0):>9.0%} |" for k in kinds))
    print("\nCorrectness split by question type. Lookups are single-fact; distractors")
    print("have a near-identical competing clause elsewhere in the document; synthesis")
    print("needs arithmetic on the retrieved clause.")

    print("\nRetrieval = answer-bearing text was retrieved | Correct = expected fact in answer")
    print("Grounded  = answer supported by retrieved context (LLM judge)")
    print("Ctx = chunk x k, the upper bound on characters the model can see\n")

    for r in results:
        if r["failures"]:
            print(f"{r['config']} missed: {'; '.join(r['failures'])}")

    out = "eval_results.csv"
    with open(out, "w") as f:
        f.write("config,chunk,overlap,k,chunks_indexed,ctx_chars,retrieval_hit,answer_correct,grounded,seconds\n")
        for r in results:
            f.write(
                f"{r['config']},{r['chunk']},{r['overlap']},{r['k']},{r['chunks_indexed']},"
                f"{r['ctx_chars']},{r['retrieval_hit']:.3f},{r['answer_correct']:.3f},"
                f"{r['grounded']:.3f},{r['seconds']:.2f}\n"
            )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
