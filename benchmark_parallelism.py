"""
Parallel vs sequential benchmark.

Measures what concurrency actually buys on this pipeline, and where it stops
buying anything.

WHY IT'S BUILT THIS WAY
-----------------------
* IDENTICAL WORKLOAD, ONE VARIABLE. Both arms run the same questions against
  the same index; only execution strategy changes. An earlier version of this
  project compared a single chat query against a two-chain analysis run, which
  changed workload size AND strategy at once and therefore measured neither.

* REPEATED RUNS, MEDIAN REPORTED. A single timing is noise - network latency to
  the API varies run to run. Median of N with the observed range shown is the
  minimum honest reporting.

* FAN-OUT IS SWEPT, NOT FIXED. The interesting result isn't "parallel is
  faster" - it's the shape of the curve, which has TWO distinct regimes:

  Below the concurrency limit, every call is genuinely in flight at once, so
  wall-clock time equals the SLOWEST call in the batch, not the average one.
  The expected maximum of N samples grows with N, so efficiency decays even
  with unlimited concurrency. This is tail latency, not a concurrency cap.

  At and above the limit, the semaphore forces the work into ceil(N / limit)
  waves, so time grows stepwise and efficiency falls off faster. That knee is
  where the constraint changes from tail latency to throughput.

  The sweep crosses the limit deliberately, so both regimes are visible. Use
  --concurrency to move the knee and confirm the mechanism rather than
  asserting it.

* WARM-UP RUN DISCARDED. The first call pays connection setup and cold caches.

USAGE
-----
    .venv/bin/python benchmark_parallelism.py            # 3 reps (default)
    .venv/bin/python benchmark_parallelism.py --reps 5
"""
import argparse
import asyncio
import os
import statistics
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

from eval_fixtures import SYNTHETIC_CONTRACT, EXTENDED_BENCHMARK_QUESTIONS
from analysis_sets import ANALYSIS_QUESTION_SETS

load_dotenv()

# Matches app.py's Legal/Book preset, so the benchmark measures the real thing.
CHUNK_SIZE, OVERLAP, K = 800, 200, 7
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# The app's real shipped set first (imported, never copied, so it can't
# drift), then extensions that push the sweep past the concurrency limit.
SHIPPED = ANALYSIS_QUESTION_SETS["Legal/Book"]
QUESTIONS = SHIPPED + EXTENDED_BENCHMARK_QUESTIONS


def build_chain():
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
    splits = splitter.split_documents([Document(page_content=SYNTHETIC_CONTRACT)])
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        collection_name=f"bench_{uuid.uuid4().hex[:12]}",
    )
    llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using ONLY the context below. Be concise.\n\n{context}"),
        ("human", "{input}"),
    ])
    chain = create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={"k": K}),
        create_stuff_documents_chain(llm, prompt),
    )
    return chain, vectorstore


async def run_parallel(chain, questions, concurrency):
    semaphore = asyncio.Semaphore(concurrency)

    async def ask(question):
        async with semaphore:
            return await chain.ainvoke({"input": question})

    return await asyncio.gather(*(ask(q) for _, q in questions))


async def run_sequential(chain, questions):
    out = []
    for _, question in questions:
        out.append(await chain.ainvoke({"input": question}))
    return out


async def timed(coro_fn, chain, questions, *args):
    start = time.perf_counter()
    await coro_fn(chain, questions, *args)
    return time.perf_counter() - start


async def main(reps: int, concurrency: int):
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)

    print(f"Corpus {len(SYNTHETIC_CONTRACT):,} chars | chunk {CHUNK_SIZE} | k {K} | "
          f"concurrency {concurrency} | {reps} reps | {len(QUESTIONS)} questions\n")
    print(f"Fan-out 2-{len(SHIPPED)} is the app's real Legal/Book set; "
          f"{len(SHIPPED) + 1}-{len(QUESTIONS)} extends past it.\n")

    chain, vectorstore = build_chain()

    print("  warm-up (discarded) ...", flush=True)
    await run_parallel(chain, QUESTIONS[:1], concurrency)

    rows = []
    for n in range(2, len(QUESTIONS) + 1):
        subset = QUESTIONS[:n]
        seq, par = [], []

        # Warm up at THIS fan-out before measuring. A single warm-up at the
        # start of the whole sweep was not enough: cold-start cost (TLS,
        # connection pool) landed inside the first measured rep.
        await run_parallel(chain, subset, concurrency)

        for rep in range(reps):
            print(f"  fan-out {n}, rep {rep + 1}/{reps} ...", flush=True)
            # ALTERNATE ARM ORDER. Running sequential first every time let the
            # parallel arm inherit a connection the sequential arm had just
            # warmed - a systematic bias that produced an impossible 3x speedup
            # on a 2-call batch (151% efficiency) before this was fixed.
            if rep % 2 == 0:
                seq.append(await timed(run_sequential, chain, subset))
                par.append(await timed(run_parallel, chain, subset, concurrency))
            else:
                par.append(await timed(run_parallel, chain, subset, concurrency))
                seq.append(await timed(run_sequential, chain, subset))

        seq_med, par_med = statistics.median(seq), statistics.median(par)
        rows.append({
            "n": n,
            "seq_med": seq_med, "seq_min": min(seq), "seq_max": max(seq),
            "par_med": par_med, "par_min": min(par), "par_max": max(par),
            "speedup": seq_med / par_med if par_med else 0,
            "efficiency": (seq_med / par_med) / n if par_med else 0,
        })

    try:
        vectorstore.delete_collection()
    except Exception:
        pass

    print(f"\n| {'Fan-out':>7} | {'Sequential':>18} | {'Parallel':>18} | {'Speedup':>7} | {'Efficiency':>10} |")
    print("|" + "-" * 79 + "|")
    for r in rows:
        seq_s = f"{r['seq_med']:.2f}s ({r['seq_min']:.1f}-{r['seq_max']:.1f})"
        par_s = f"{r['par_med']:.2f}s ({r['par_min']:.1f}-{r['par_max']:.1f})"
        marker = "  <- concurrency limit" if r["n"] == concurrency else ""
        if r["efficiency"] > 1.0:
            marker = "  <- INVALID: efficiency cannot exceed 100%"
        elif r["speedup"] > min(r["n"], concurrency) + 0.05:
            marker = f"  <- INVALID: speedup cannot exceed {min(r['n'], concurrency)}x here"
        print(f"| {r['n']:>7} | {seq_s:>18} | {par_s:>18} | {r['speedup']:>6.2f}x | "
              f"{r['efficiency']:>9.0%} |{marker}")

    print(f"\nMedian of {reps} reps; observed range in parentheses. If the ranges for")
    print("sequential and parallel overlap at a given fan-out, that row is noise -")
    print("raise --reps until they separate.\n")
    print("Speedup    = sequential / parallel. Ceiling is min(fan-out, concurrency):")
    print("             you cannot go faster than running every call at once.")
    print("Efficiency = speedup / fan-out. 100% is perfect scaling; ABOVE 100% is")
    print("             impossible and means the measurement is biased, not fast.")
    print(f"\nTwo regimes, with the knee at fan-out {concurrency}:")
    print(f"  Below {concurrency}: all calls are in flight at once, so wall-clock equals the")
    print("            SLOWEST call. The expected max of N samples grows with N, so")
    print("            efficiency decays from tail latency alone - no cap involved.")
    print(f"  Above {concurrency}: the semaphore forces ceil(N/{concurrency}) waves, so time grows")
    print("            stepwise and efficiency falls off faster.")
    print("\nRe-run with a different --concurrency to move the knee and confirm this.")

    with open("benchmark_results.csv", "w") as f:
        f.write("fan_out,seq_median,seq_min,seq_max,par_median,par_min,par_max,speedup,efficiency\n")
        for r in rows:
            f.write(f"{r['n']},{r['seq_med']:.3f},{r['seq_min']:.3f},{r['seq_max']:.3f},"
                    f"{r['par_med']:.3f},{r['par_min']:.3f},{r['par_max']:.3f},"
                    f"{r['speedup']:.3f},{r['efficiency']:.3f}\n")
    print("\nWrote benchmark_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=3, help="repetitions per fan-out (default 3)")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="max simultaneous calls - where the knee should appear (default 5)")
    args = parser.parse_args()
    asyncio.run(main(args.reps, args.concurrency))
