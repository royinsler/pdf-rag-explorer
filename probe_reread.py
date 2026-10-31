"""Does re-reading a damaged page at higher resolution recover the lost value?

The agent's reocr_page tool assumes it can. This is the experiment that checks,
on a real file, before anything is built on the assumption. It is a diagnostic
in the same family as damage_report.py and ocr_text_diff.py: it exists because a
question needed evidence rather than an argument.

    python probe_reread.py contract.debug.pdf                    # pages with detected damage, 300/400/600 DPI
    python probe_reread.py contract.debug.pdf --page 2           # a specific page, damaged or not
    python probe_reread.py contract.debug.pdf --force-ocr        # a text-layer PDF has no page map; this builds one
    python probe_reread.py contract.debug.pdf --dpi 400 500 600  # other resolutions

For each page and resolution it reports the dates that appear ONLY in the
re-read (recovered) and ONLY in the original extraction (lost), and the text
around each damaged connector in the re-read. Everything prints to the terminal;
nothing is written to disk and nothing is sent anywhere. No API key is needed.

How to read the result: "dates only in the re-read" naming the date you know is
printed there means a second pass can help, and the agent should try one.
"none" at every resolution means it cannot, on this file, and a second read of
the same pixels by the same engine is not the fix - which is the case for a
second engine, the roadmap item in the README.
"""
import argparse
import re
import sys

from agent_tools import DocumentContext, build_tools, excerpt_around, run_tool
from extraction import OCR_LANG, extract_document
from verification import _DATE_RE, _canon_date

DEFAULT_DPIS = (300, 400, 600)


def dates_in(text):
    """The set of (day, month, year) dates written in `text`, format-independent."""
    return {d for d in (_canon_date(m) for m in re.findall(_DATE_RE, text)) if d}


def fmt(date):
    day, month, year = date
    return f"{day:02d}.{month:02d}.{year}"


def _dates(dates):
    return ", ".join(fmt(d) for d in sorted(dates)) or "none"


def probe(ctx, pages, dpis=DEFAULT_DPIS, out=print):
    """Re-read each page at each resolution and report what changed.

    Goes through the agent's own reocr_page tool rather than calling OCR
    directly, so what is measured here is what the agent would actually get.
    """
    tools = build_tools(ctx)
    for page in pages:
        original = ctx.pages.get(page, "")
        connectors = [d.connector for d in ctx.damaged if ctx.page_containing(d.snippet) == page]
        before = dates_in(original)
        out(f"\n=== page {page}: {len(original)} characters in the original extraction; "
            f"dates there: {_dates(before)}")
        for dpi in dpis:
            result = run_tool(tools, "reocr_page", page=page, dpi=dpi)
            if not result.ok:
                out(f"  {dpi} DPI: failed - {result.error}")
                continue
            data = result.data
            after = dates_in(data["text"])
            out(f"  {dpi} DPI: mean confidence {data['mean_confidence']}")
            out(f"    dates only in the re-read:   {_dates(after - before)}")
            out(f"    dates only in the original:  {_dates(before - after)}")
            if data["low_confidence_terms"]:
                out(f"    low-confidence terms: {', '.join(data['low_confidence_terms'])}")
            for connector in dict.fromkeys(connectors):
                for excerpt in excerpt_around(data["text"], connector):
                    out(f"    near {connector!r}: {excerpt}")


def main(argv=None, out=print):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pdf")
    parser.add_argument("--page", type=int, action="append", help="1-based page; repeatable")
    parser.add_argument("--dpi", type=int, nargs="+", default=list(DEFAULT_DPIS))
    parser.add_argument("--lang", default=OCR_LANG, help=f"Tesseract languages (default {OCR_LANG})")
    parser.add_argument("--force-ocr", action="store_true",
                        help="OCR every page, so a text-layer PDF gets page numbers")
    args = parser.parse_args(argv)

    with open(args.pdf, "rb") as handle:
        file_bytes = handle.read()
    extraction = extract_document(file_bytes, force_ocr=args.force_ocr, lang=args.lang)
    if not extraction.usable:
        out(f"Could not use this PDF: {extraction.reason}.")
        return 2

    ctx = DocumentContext.from_extraction(extraction, file_bytes=file_bytes, ocr_lang=args.lang)
    out(f"Read by {'OCR' if extraction.used_ocr else 'the PDF text layer'}; "
        f"{len(ctx.damaged)} damaged passage(s) detected; "
        f"page map: {sorted(ctx.pages) if ctx.pages else 'none'}")

    if not ctx.pages:
        out("No page numbers are available, because the text layer comes back as one "
            "undifferentiated document. Re-run with --force-ocr to OCR it page by page.")
        return 1

    for number, damaged in enumerate(ctx.damaged, 1):
        page = ctx.page_containing(damaged.snippet)
        out(f"  damaged passage {number}: page {page if page else '? (spans pages or not found)'}, "
            f"connector {damaged.connector!r}, signals: {', '.join(damaged.reasons)}")

    pages = args.page or sorted({p for p in (ctx.page_containing(d.snippet) for d in ctx.damaged) if p})
    if not pages:
        out("No damaged passage maps to a single page. Pass --page N to probe a page directly.")
        return 1

    probe(ctx, pages, args.dpi, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
