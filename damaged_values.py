"""Detecting values a document promises but extraction did not deliver intact.

Why this exists, precisely:

OCR confidence (see extraction.ocr_confidence) scores the tokens the engine
produced. It caught 68 damaged tokens in the reference contract at a mean
confidence of 87.7%. It could not catch the one that mattered, for a structural
reason rather than a tuning one: Tesseract did not misread the date
"03.09.2027" in the renovation clause — it emitted nothing at all for it. A
deleted value produces no token, a token that does not exist has no confidence
score, and a score that does not exist cannot fall below a threshold. No
calibration of OCR_MIN_CONFIDENCE reaches this failure.

What a deletion DOES leave is a hole in the grammar. The clause read

    ... לצורך ביצוע עבודות שיפוץ בלבד החל מיום 03.09.2027

and extraction produced

    ... לצורך ביצוע עבדות שיפוץ בלבד החל DYN 6.

"החל" ("commencing") survived at high confidence, and what follows it is Latin
gibberish and a stray number in an otherwise-Hebrew document. The connector
still promises a date. The date is gone. That mismatch is checkable.

The naive version of this check — "flag every date connector with no date after
it" — fires constantly on correct text, because Hebrew uses the same words
relationally: "מיום שנדרש" (from the day he was required), "החל ממועד הנ״ל"
(from the aforementioned date). Both are complete sentences with no date in
them and nothing wrong. A check that cries wolf on correct text is worse than
no check, which this project has already learned once (see
verification._canon_date).

So the rule requires TWO things to coincide: a connector that promises a value,
and evidence of extraction damage in the window where the value should be. The
damage evidence is what makes it specific — it is the difference between "this
sentence has no date in it" and "this sentence lost its date."

MEASURED TWICE, and the second measurement is the important one.

RECALL, on a second contract (test.pdf): the rule found nothing, and that is a
failure, not a clean pass. That document has a payment date, "ביום 1.7.2026",
which OCR deleted — and the damage evidence was right there, "סך" mangled into
the Latin "Jo" and a bare "6" left where the date had been. The rule still said
nothing, because it never looked: "ביום", "מיום", "ליום" and "בתאריך" occur ZERO
times in the entire extracted document, and "החל" once. OCR destroyed the
connectors themselves.

So this check assumes the grammar survives even when the value does not. That
held on one contract and failed on the next, and the assumption is a bad bet on
reflection: the anchor words are short, common function words, which are
precisely the tokens an OCR engine is most likely to misread. A rule anchored on
vocabulary inherits the fragility of that vocabulary.

Widening the connector list does not fix this — the failure is not a missing
word, it is that no word survived. Fuzzy-matching the connectors would trade
away the precision below for unmeasured recall. The honest fix is at extraction:
OCR the page with a second engine and diff the outputs, where disagreement
detects absence directly and needs no anchor at all.

PRECISION, on the contract this was built from: Run over the whole
reference contract (10,014 chars) the rule fires three times:

  1. "...בגין כל איחור החל מהיום העשירי לפיגור..." — with DWNT (confidence 3.0)
     in the window. No date was ever printed here: "from the tenth day" is
     relational. Damaged region, but nothing was deleted.
  2. "...שיפוץ בלבד החל DYN 6,..." — the renovation date, genuinely gone.
  3. "...החל ממועד הנ״ל..." where הנ״ל came out as NW". Again damaged, again
     nothing deleted.

As a DELETION detector that is 1 hit in 3. But all three windows do contain
corrupted text, and in all three the value the sentence points at cannot be
trusted — which is the question a reader actually has. So the concept was
widened to match what the rule detects rather than the rule narrowed to match
the original name: this finds values that are missing OR corrupted, and says
which signals fired so the distinction stays visible.
"""
import re
from dataclasses import dataclass
from typing import List, Sequence

# Words that promise a date immediately after them. Bounded by Hebrew-letter
# lookaround rather than \b, because Python's \b does not treat Hebrew prefix
# letters (ב/ל/מ/ה/ו) as word boundaries the way the grammar does.
_CONNECTORS = ["מיום", "ביום", "ליום", "החל", "בתאריך", "מתאריך", "עד ליום", "לא יאוחר"]
_CONNECTOR_RE = re.compile(
    "(?<![א-ת])(" + "|".join(_CONNECTORS) + ")(?![א-ת])"
)

# Any recognisable date form, including the Hebrew "9 לחודש נובמבר שנת 2027"
# spelled-out style that appears in contract preambles.
_DATE_RE = re.compile(
    r"\d{1,4}[./-]\d{1,2}[./-]\d{1,4}"
    r"|\d{1,2}\s+(?:לחודש|ב)?\s*(?:ינואר|פברואר|מרץ|אפריל|מאי|יוני|יולי|אוגוסט|ספטמבר|אוקטובר|נובמבר|דצמבר)"
)

# A run of Latin letters. In a document that is overwhelmingly Hebrew, this is
# how Tesseract renders Hebrew it has failed on — "DYN", "DWNT", "DION". Real
# Latin text does occur in these contracts ("As-Is", "AS IS"), which is why a
# Latin run alone is not enough to flag anything.
_LATIN_RE = re.compile(r"[A-Za-z]{2,}")

# A number standing alone with no units, currency or date structure — the debris
# left when a neighbouring line's clause number bleeds into the hole.
_BARE_NUMBER_RE = re.compile(r"(?<![\d./-])\d{1,2}\.?(?![\d./-])")

WINDOW_CHARS = 45


@dataclass
class DamagedValue:
    """A place where the text promises a value and extraction did not deliver it intact."""
    connector: str
    snippet: str
    reasons: List[str]

    def describe(self) -> str:
        return f"{self.snippet!r} ({', '.join(self.reasons)})"


def _is_hebrew_document(text: str) -> bool:
    letters = sum(ch.isalpha() for ch in text)
    hebrew = sum(1 for ch in text if "֐" <= ch <= "׿")
    return letters > 0 and hebrew / letters > 0.5


def detect_damaged_values(text: str,
                          low_confidence_terms: Sequence = (),
                          window_chars: int = WINDOW_CHARS) -> List[DamagedValue]:
    """Find connectors that promise a value where extraction left damage instead.

    Returns [] on clean text. Each DamagedValue carries the reasons it fired, so
    the UI and the prompt can say WHY a region is suspect rather than asserting
    it — and so a reader can tell a deletion from a corruption.
    """
    if not text:
        return []

    hebrew_doc = _is_hebrew_document(text)
    suspect_tokens = {token for token, _ in low_confidence_terms}
    found: List[DamagedValue] = []

    for match in _CONNECTOR_RE.finditer(text):
        window = text[match.end():match.end() + window_chars]
        if _DATE_RE.search(window):
            continue                      # the promise was kept

        reasons = []
        # Latin gibberish inside Hebrew prose is the strongest single signal.
        if hebrew_doc and _LATIN_RE.search(window):
            reasons.append("Latin characters where Hebrew is expected")
        # A token the OCR engine itself doubted, sitting in the gap.
        if any(token in window for token in suspect_tokens):
            reasons.append("a token OCR scored as low confidence")
        # A stray bare number — a fragment of the value, or of a clause number
        # that collapsed into the hole when the real value vanished.
        if _BARE_NUMBER_RE.search(window):
            reasons.append("a bare number with no date structure")

        # One signal alone is too weak: legitimate prose routinely contains a
        # connector with no date after it. Two independent signals in the same
        # window is the bar for calling it damage.
        if len(reasons) >= 2:
            found.append(DamagedValue(
                connector=match.group(1),
                snippet=_snippet(text, match.start() - 25,
                                 match.end() + window_chars),
                reasons=reasons,
            ))
    return found


def _snippet(text: str, start: int, end: int) -> str:
    """Quote a passage on whole-word boundaries.

    A snippet cut mid-word is shown to a human in the UI and handed to the model
    in the prompt; in both places a severed first word reads as further damage
    and obscures the damage actually being reported.
    """
    start = max(0, start)
    end = min(len(text), end)
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    while end < len(text) and not text[end].isspace():
        end += 1
    return " ".join(text[start:end].split())


def damage_caveat(damaged: Sequence[DamagedValue], limit: int = 5) -> str:
    """A prompt fragment naming the passages whose promised value is untrustworthy.

    Parallel to extraction.ocr_caveat and deliberately worded differently. That
    one says "these tokens are probably wrong"; this one says "the value this
    sentence points at is not reliably here." The wording avoids asserting
    deletion, because the measurement above showed the rule cannot tell a
    deletion from a corruption — and claiming the stronger thing would be the
    same overreach as an answer inventing a date.
    """
    if not damaged:
        return ""
    shown = "; ".join(f'"{d.snippet}"' for d in damaged[:limit])
    more = "" if len(damaged) <= limit else f" (and {len(damaged) - limit} more)"
    return (
        " In these passages the text promises a date or amount that extraction "
        f"did not capture intact — it is missing or corrupted: {shown}{more}. If a "
        "question depends on one of these values, say the document does not show "
        "it legibly and that extraction appears to have damaged it. Never report "
        "the surrounding fragments as if they were the value."
    )


def damage_caveat_narrow(damaged: Sequence[DamagedValue], limit: int = 5) -> str:
    """CANDIDATE replacement for damage_caveat; not used by the app. See eval_caveat.py.

    The current caveat says "if a question depends on one of these values, say the
    document does not show it legibly". Measured on a handful of questions it prevented
    one echo of OCR debris and caused one refusal of a question the text could answer
    (damage nearby, answer intact). This version limits the refusal to the case where the
    damaged value IS what is asked for, and says to answer from intact text otherwise
    without repeating the broken characters. Whether it does better is what the
    experiment is for; adopt it only if the numbers say so.
    """
    if not damaged:
        return ""
    shown = "; ".join(f'"{d.snippet}"' for d in damaged[:limit])
    more = "" if len(damaged) <= limit else f" (and {len(damaged) - limit} more)"
    return (
        " In these passages the text promises a date or amount that extraction "
        f"did not capture intact — it is missing or corrupted: {shown}{more}. Say the "
        "document does not show a value legibly ONLY when that damaged value is itself "
        "what the question asks for, and add that extraction appears to have damaged it. "
        "If the question can be answered from intact text, answer from that text and do "
        "not repeat the broken characters. Never report the surrounding fragments as if "
        "they were the value."
    )
