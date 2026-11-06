"""
Answer verification — checking what the model said against what the document says.

Separate from app.py so it can be tested without starting Streamlit. Two
different checks, catching two different failures:

  verify_quotes       - did the model fabricate a citation?
  verify_answer_facts - did the model state a date or amount the document
                        never contained, inside an otherwise-honest answer?

The second is the more likely failure and, in a contract, the more expensive
one. A citation you can check is worth little if the number in the prose beside
it was invented.
"""
import re


def verify_quotes(quotes, source_text):
    """Check each quoted passage really appears in the document.

    Whitespace is normalised before comparing, because extraction and OCR
    introduce line breaks that a faithful quote will not reproduce exactly.
    Returns [(quote, found)] so the UI can mark anything unverified rather
    than presenting a fabricated citation as evidence.
    """
    haystack = re.sub(r"\s+", " ", source_text).lower()
    checked = []
    for quote in quotes:
        needle = re.sub(r"\s+", " ", quote).strip().lower()
        checked.append((quote, bool(needle) and needle in haystack))
    return checked


# Values that carry legal weight in a contract, and the ones a model is most
# likely to construct when extraction has destroyed the real one.
_DATE_RE = r"\b\d{1,4}[./-]\d{1,2}[./-]\d{1,4}\b"
_AMOUNT_RE = r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b"


def _canon_date(text):
    """Normalise a date to (day, month, year) so formats compare equal.

    Written after a first version compared separator-stripped strings and
    decided 4/11/2027 did not match 4.11.2027 - a zero-padding difference. A
    check that flags CORRECT values is worse than no check, because it teaches
    the reader to dismiss the warning.
    """
    parts = re.split(r"[./-]", text)
    if len(parts) != 3:
        return None
    try:
        nums = [int(part) for part in parts]
    except ValueError:
        return None
    if nums[0] > 31:                     # 2027-11-04
        year, month, day = nums
    else:                                # 04.11.2027
        day, month, year = nums
    if year < 100:
        year += 2000
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return (day, month, year)


def verify_answer_facts(answer, source_text):
    """Check every date and amount stated in the answer appears in the source.

    Quote verification catches a fabricated citation. This catches the more
    likely and more expensive failure: an otherwise-honest answer containing a
    number the document never stated.

    Observed in this project: OCR destroyed a date, leaving "החל DYN" followed
    by the next clause's number. The model answered "8.8.2027" - assembled from
    that clause number and a year borrowed from elsewhere in the contract - and
    presented it as fact. Nothing flagged it, because the citation was real and
    only the figure inside the prose was invented.

    Limitation: only comma-formatted amounts are checked, so a bare "500000"
    passes unexamined. Deliberate - broadening it to any long number would
    sweep in years and clause numbers and produce false alarms, and this check
    is only worth having if a warning means something.
    """
    source_dates = {d for d in (_canon_date(m) for m in re.findall(_DATE_RE, source_text)) if d}
    source_amounts = {re.sub(r"[,\s]", "", m) for m in re.findall(_AMOUNT_RE, source_text)}

    seen, checked = set(), []
    for match in re.findall(_DATE_RE, answer):
        if match not in seen:
            seen.add(match)
            canon = _canon_date(match)
            checked.append((match, canon is not None and canon in source_dates))
    for match in re.findall(_AMOUNT_RE, answer):
        if match not in seen:
            seen.add(match)
            checked.append((match, re.sub(r"[,\s]", "", match) in source_amounts))
    return checked

# Latin text that legitimately appears in these Hebrew contracts. Everything
# else in Latin script, in a Hebrew document, is Tesseract's rendering of
# Hebrew it failed on.
_LATIN_ALLOWLIST = {"as", "is", "as-is", "asis", "nis", "ils"}
_LATIN_RUN_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")


def find_ocr_garbage_echoes(answer, source_text, allowlist=_LATIN_ALLOWLIST):
    """Find OCR gibberish the answer has copied out of the document.

    The third check in this module, and the one that exists because the other
    two are not enough. The pipeline can now tell the model exactly which
    passages are damaged and instruct it never to report the surrounding
    fragments as a value — and the model does it anyway, answering with
    "החל DYN 6" copied verbatim from the corrupted text.

    So this does not ask the model for anything. It reads the answer after the
    fact and asks a question with a definite answer: does this Hebrew answer
    contain Latin character runs that also appear in the document? In a Hebrew
    contract, such a run is not a word — it is what OCR produced where it could
    not read Hebrew, and an answer repeating it is presenting damage as content.

    Requiring the run to appear in the source is what keeps this specific: it
    distinguishes "the model echoed corrupted extraction" from "the model used
    an English word", which is the difference between a broken answer and a
    fine one.

    Limitation, stated rather than hidden: this is the wrong check for a
    document that is genuinely bilingual, where Latin runs are ordinary
    content. It is gated on the document being predominantly Hebrew.
    """
    hebrew = sum(1 for ch in source_text if "\u0590" <= ch <= "\u05ff")
    letters = sum(ch.isalpha() for ch in source_text)
    if not letters or hebrew / letters < 0.5:
        return []

    source_runs = {run.lower() for run in _LATIN_RUN_RE.findall(source_text)}
    echoes = []
    for run in _LATIN_RUN_RE.findall(answer):
        key = run.lower()
        if key in allowlist or key not in source_runs:
            continue
        if run not in echoes:
            echoes.append(run)
    return echoes

def quote_only_debris(answer, quotes, source_text):
    """OCR gibberish that appears in the quoted evidence but not in the answer itself.

    A quote must be copied exactly (verify_quotes rejects a "tidied" one), so quoting a
    sentence that sits next to damage copies the damage too. That is not the answer
    repeating garbage, and treating it as such marks correct answers as broken (the
    87-run live eval: "750 ש\"ח" flagged because its supporting quote contained DWNT).
    It is still worth showing - the evidence is damaged - so it is reported separately and
    does not by itself fail an answer or trigger a retry.

    KNOWN COST: a fluent invented answer with no numbers, no overlap with a damaged passage
    and debris only in its quote used to be caught here and no longer is. The prompt's damage
    caveat and the provenance check remain; see test_damage_regression.py.
    """
    in_answer = set(find_ocr_garbage_echoes(answer, source_text))
    return [run for run in find_ocr_garbage_echoes(" ".join(quotes), source_text) if run not in in_answer]


def answer_rests_on_damage(answer, damaged_snippets, min_overlap=25):
    """Report which damaged passages an answer was drawn from.

    This is the provenance check, and it exists because presence is not
    provenance. verify_answer_facts asks "does this value appear in the
    document?" — a question that says yes when the value appears anywhere, for
    any obligation. In the reference contract the date 3.9.2027 is printed for
    a payment; had the model guessed it for the renovation clause, the check
    would have confirmed it against text from an entirely different section.

    This asks the other question: is the passage this answer came from one we
    already know is damaged? It needs no cooperation from the model and no
    recognisable garbage in the output — which matters, because the failure that
    prompted it has neither. Told not to report damaged fragments, the model
    stopped copying "DYN" and instead wrote "החל מיום 6": clean Hebrew, correct
    grammar, and the 6 is debris from the clause heading below. Nothing in that
    sentence looks wrong. Its source does.

    Overlap is measured on the longest shared run of characters, so a passing
    mention of a common phrase does not trigger it — the answer has to actually
    reproduce the damaged passage.
    """
    from difflib import SequenceMatcher

    clean = lambda text: re.sub(r"\s+", " ", text).strip()
    answer_text = clean(answer)
    if not answer_text:
        return []

    resting_on = []
    for snippet in damaged_snippets:
        snippet_text = clean(snippet)
        match = SequenceMatcher(None, answer_text, snippet_text).find_longest_match(
            0, len(answer_text), 0, len(snippet_text))
        if match.size >= min_overlap:
            resting_on.append((snippet, answer_text[match.a:match.a + match.size]))
    return resting_on
