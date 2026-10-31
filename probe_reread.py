"""Does reading a damaged page differently recover the lost value?

The agent's reocr_page tool assumes a second pass can. This is the experiment
that checks, on a real file, before anything is built on the assumption. It is
a diagnostic in the same family as damage_report.py and ocr_text_diff.py: it
exists because a question needed evidence rather than an argument.

    python probe_reread.py contract.debug.pdf --force-ocr                    # damaged pages, 300/400/600 DPI
    python probe_reread.py contract.debug.pdf --force-ocr --expect 11.9.27   # ...and say whether that date turns up
    python probe_reread.py contract.debug.pdf --force-ocr --page 4 --dpi 300 --psm 6 11 12
    python probe_reread.py contract.debug.pdf --force-ocr --page 4 --dpi 300 --threshold 140 180 220

Options that change HOW the pixels are read, each tried alone and in
combination with the others you list:
  --dpi        resolution to rasterise at (default 300 400 600)
  --psm        Tesseract page-segmentation mode; 6 = one block of text, 11 and 12
               = sparse text, which can find a lone token the default layout
               analysis skips
  --threshold  convert to black and white at this grey level (1-254); can lift a
               faint or coloured value off its background
Every combination is one OCR run of the page (a few seconds each), so list a
few values, not many.

--expect is the date you KNOW is printed there. The report then says, per
configuration, whether it was found, and ends with the configurations that did
find it - the answer to the question this script exists for.

For each configuration it reports the dates that appear ONLY in the re-read
(recovered) and ONLY in the original extraction (lost), and the text around each
damaged connector. It ends with the values that two or more configurations
agree on and the original lacked: that is exactly what the agent will accept
(see DocumentContext.corroborated_new), because a single re-read is not
evidence - the same printed date has been read correctly at 300 and 600 DPI and
as a different date at 400. Everything prints to the terminal; nothing is written
to disk and nothing is sent anywhere. No API key is needed.
"""
import argparse
import itertools
import sys

from agent_tools import (DocumentContext, build_tools, dates_in, excerpt_around,
                         fmt_date, run_tool)
from extraction import OCR_LANG, extract_document
from verification import _canon_date

DEFAULT_DPIS = (300, 400, 600)

# Kept under the old names: the tests and other callers import them from here.
fmt = fmt_date


def _dates(dates):
    return ", ".join(fmt_date(d) for d in sorted(dates)) or "none"


def label(dpi, psm=None, threshold=None):
    parts = [f"{dpi} DPI"]
    if psm is not None:
        parts.append(f"psm {psm}")
    if threshold is not None:
        parts.append(f"threshold {threshold}")
    return ", ".join(parts)


def parse_date(text):
    """'11.9.27', '15/05/2026' -> (15, 5, 2026), or None if it is not a date."""
    return _canon_date(text.strip())


def probe(ctx, pages, dpis=DEFAULT_DPIS, out=print, psms=(None,), thresholds=(None,), expect=None):
    """Re-read each page in every configuration and report what changed.

    Goes through the agent's own reocr_page tool rather than calling OCR
    directly, so what is measured here is what the agent would actually get.
    Returns {page: [labels of the configurations that found `expect`]}.
    """
    tools = build_tools(ctx)
    found_by = {}
    for page in pages:
        original = ctx.pages.get(page, "")
        connectors = [d.connector for d in ctx.damaged if ctx.page_containing(d.snippet) == page]
        before = dates_in(original)
        found_by[page] = []
        out(f"\n=== page {page}: {len(original)} characters in the original extraction; "
            f"dates there: {_dates(before)}")
        if expect is not None:
            out(f"    looking for {fmt_date(expect)}; "
                f"{'already in the original' if expect in before else 'NOT in the original'}")
        for dpi, psm, threshold in itertools.product(dpis, psms, thresholds):
            name = label(dpi, psm, threshold)
            result = run_tool(tools, "reocr_page", page=page, dpi=dpi, psm=psm, threshold=threshold)
            if not result.ok:
                out(f"  {name}: failed - {result.error}")
                continue
            data = result.data
            after = dates_in(data["text"])
            out(f"  {name}: mean confidence {data['mean_confidence']}")
            out(f"    dates only in the re-read:   {_dates(after - before)}")
            out(f"    dates only in the original:  {_dates(before - after)}")
            if expect is not None:
                hit = expect in after
                out(f"    {fmt_date(expect)}: {'FOUND' if hit else 'not found'}")
                if hit:
                    found_by[page].append(name)
            if data["low_confidence_terms"]:
                out(f"    low-confidence terms: {', '.join(data['low_confidence_terms'])}")
            for connector in dict.fromkeys(connectors):
                for excerpt in excerpt_around(data["text"], connector):
                    out(f"    near {connector!r}: {excerpt}")

        agreed = ctx.corroborated_new(page)
        out(f"  values two or more configurations agree on that the original lacked "
            f"(what the agent would accept): {', '.join(agreed) or 'none'}")
        if expect is not None:
            names = found_by[page]
            out(f"  {fmt_date(expect)} was found by {len(names)} configuration(s)"
                + (f": {'; '.join(names)}" if names else ""))
    return found_by


def main(argv=None, out=print):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pdf")
    parser.add_argument("--page", type=int, action="append", help="1-based page; repeatable")
    parser.add_argument("--dpi", type=int, nargs="+", default=list(DEFAULT_DPIS))
    parser.add_argument("--psm", type=int, nargs="+", default=[None],
                        help="Tesseract page-segmentation modes to try, e.g. 6 11 12")
    parser.add_argument("--threshold", type=int, nargs="+", default=[None],
                        help="black-and-white grey levels to try (1-254), e.g. 140 180")
    parser.add_argument("--expect", help="a date you know is printed there, e.g. 11.9.27")
    parser.add_argument("--lang", default=OCR_LANG, help=f"Tesseract languages (default {OCR_LANG})")
    parser.add_argument("--force-ocr", action="store_true",
                        help="OCR every page, so a text-layer PDF gets page numbers")
    args = parser.parse_args(argv)

    expect = None
    if args.expect:
        expect = parse_date(args.expect)
        if expect is None:
            out(f"--expect {args.expect!r} is not a date I can read; use a form like 11.9.27.")
            return 2

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

    runs = len(args.dpi) * len(args.psm) * len(args.threshold) * len(pages)
    out(f"{runs} OCR run(s) to do; each takes a few seconds.")
    probe(ctx, pages, args.dpi, out, psms=args.psm, thresholds=args.threshold, expect=expect)
    return 0


if __name__ == "__main__":
    sys.exit(main())
