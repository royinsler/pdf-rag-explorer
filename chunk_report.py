"""
Chunk-quality report for a real document.

Exists because eval_rag.py cannot measure the tiny-chunk problem: its corpus is
clean text fed straight to the splitter, so it never produces the page-header
fragments that OCR does on scanned PDFs. This runs the real extraction pipeline
over a real file and reports what the size filter actually removes, so the
decision rests on the document rather than on my assumption about it.

    .venv/bin/python chunk_report.py /path/to/file.pdf
    .venv/bin/python chunk_report.py /path/to/file.pdf --chunk 800 --overlap 200
"""
import argparse
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter

from extraction import extract_document, drop_tiny_chunks, MIN_CHUNK_CHARS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--chunk", type=int, default=800, help="chunk size (default: Legal/Book preset)")
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--min-chars", type=int, default=MIN_CHUNK_CHARS)
    args = ap.parse_args()

    with open(args.pdf, "rb") as f:
        data = f.read()

    result = extract_document(data)
    if not result.usable:
        print(f"Extraction failed: {result.reason}")
        sys.exit(1)

    print(f"{args.pdf}")
    print(f"extracted {len(result.text):,} chars from {len(result.docs)} page(s)"
          f"{' via OCR' if result.used_ocr else ''}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk, chunk_overlap=args.overlap)
    splits = splitter.split_documents(result.docs)
    kept, dropped = drop_tiny_chunks(splits, args.min_chars)

    sizes = sorted(len(s.page_content.strip()) for s in splits)
    print(f"\nchunk size {args.chunk}, overlap {args.overlap} -> {len(splits)} chunks")
    print(f"  smallest {sizes[0]}  median {sizes[len(sizes)//2]}  largest {sizes[-1]}")

    buckets = [(0, 40), (40, 80), (80, 200), (200, 500), (500, 10**9)]
    print("\n  size distribution")
    for lo, hi in buckets:
        n = sum(1 for s in sizes if lo <= s < hi)
        label = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
        bar = "#" * n
        print(f"    {label:>8} chars  {n:>3}  {bar}")

    print(f"\nfilter at {args.min_chars} chars: keeps {len(kept)}, drops {len(dropped)}"
          f"  ({len(dropped) / max(1, len(splits)):.0%} of chunks)")

    if dropped:
        print("\ndropped fragments (these were competing for retrieval slots):")
        for chunk in dropped:
            body = " ".join(chunk.page_content.split())
            print(f"  [{len(body):>3} chars] {body[:90]}")
    else:
        print("\nNothing dropped - this document produces no undersized chunks.")


if __name__ == "__main__":
    main()
