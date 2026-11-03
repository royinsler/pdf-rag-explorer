"""
Synthetic evaluation fixture.

A fabricated service agreement used as the eval corpus. It is synthetic on
purpose: it contains no real or private data, it can be committed to the repo,
and — most importantly — we know exactly which sentence answers each question,
which is what makes precise retrieval measurement possible. All names, figures
and addresses below are invented.

Facts are deliberately scattered across sections so that retrieval depth
actually matters: no single chunk contains them all.
"""

SYNTHETIC_CONTRACT = """
MASTER SERVICES AGREEMENT

1. PARTIES AND EFFECTIVE DATE

This Master Services Agreement is entered into between Meridian Analytics Ltd.,
a company registered in Dublin, Ireland ("the Provider"), and Northwind Logistics
GmbH, a company registered in Hamburg, Germany ("the Client"). This Agreement takes
effect on 14 March 2025 (the "Effective Date") and governs all services delivered
by the Provider to the Client from that date forward, superseding any prior
arrangements, proposals or correspondence between the parties on the same subject.

2. SCOPE OF SERVICES

The Provider shall deliver freight analytics services, comprising route optimisation
modelling, demand forecasting, and a quarterly performance review delivered in
writing. The Provider shall assign a named account manager to the Client within ten
business days of the Effective Date. Services are delivered remotely unless the
parties agree otherwise in writing. Any expansion of scope requires a written change
order signed by authorised representatives of both parties, and no verbal instruction
shall be treated as amending the scope of this Agreement.

3. TERM AND RENEWAL

The initial term of this Agreement is 36 months from the Effective Date. Upon
expiry, the Agreement renews automatically for successive periods of 12 months
unless either party gives written notice of non-renewal at least 60 days before the
end of the then-current term. Renewal does not reset any accrued obligations or
outstanding balances.

4. FEES AND PAYMENT

The Client shall pay the Provider a fixed fee of EUR 12,500 per month, invoiced
monthly in arrears. Invoices are due within 30 days of the invoice date. Amounts
remaining unpaid after the due date accrue interest at 1.5% per month, calculated
daily from the due date until payment is received in full. Fees are exclusive of
value added tax, which shall be added where applicable. The Provider may increase
the monthly fee once per calendar year, by no more than 4%, on 90 days written
notice to the Client.

5. SERVICE LEVELS

The Provider warrants platform availability of 99.5% measured monthly, excluding
scheduled maintenance notified at least 48 hours in advance. Where availability
falls below the warranted level, the Client is entitled to a service credit of 5%
of that month's fee for each 0.1 percentage point below 99.5%, capped at 50% of the
monthly fee. Service credits are the Client's sole remedy for availability
shortfalls and must be claimed within 30 days of the end of the affected month.

6. CONFIDENTIALITY

Each party shall keep confidential all non-public information received from the
other and shall not disclose it to any third party without prior written consent,
except where disclosure is required by law or by a competent regulatory authority.
This obligation survives termination of this Agreement for a period of five years.

7. DATA PROTECTION AND RETENTION

The Provider processes Client data solely to deliver the services described in
Section 2. Upon termination or expiry of this Agreement, the Provider shall delete
all Client data within 60 days, and shall certify such deletion in writing to the
Client on request. The Provider shall not transfer Client data outside the European
Economic Area without the Client's prior written consent.

8. LIMITATION OF LIABILITY

The aggregate liability of either party arising out of or in connection with this
Agreement, whether in contract, tort or otherwise, shall not exceed EUR 450,000.
Neither party shall be liable for indirect or consequential loss, loss of profit,
or loss of anticipated savings. Nothing in this clause limits liability for death
or personal injury caused by negligence, or for fraud or fraudulent
misrepresentation.

9. TERMINATION

Either party may terminate this Agreement for convenience by giving 90 days written
notice to the other. Either party may terminate immediately by written notice where
the other party commits a material breach that remains unremedied 30 days after
written notice of that breach, or where the other party becomes insolvent, enters
administration, or ceases to carry on business. On termination the Client shall pay
all fees accrued up to the effective date of termination.

10. GOVERNING LAW AND DISPUTES

This Agreement is governed by the laws of Ireland. The parties submit to the
exclusive jurisdiction of the courts of Ireland. Before commencing proceedings, the
parties shall attempt in good faith to resolve any dispute through senior executive
negotiation for a period of no less than 30 days.

11. NOTICES

Notices under this Agreement shall be sent in writing to legal@meridian-analytics.example
for the Provider and to contracts@northwind-logistics.example for the Client. Notice
is deemed given on the second business day after sending. A change of notice address
must itself be notified in accordance with this clause.

12. INTELLECTUAL PROPERTY

All intellectual property rights in the Provider's platform, models, methodologies
and documentation remain vested in the Provider. The Client is granted a
non-exclusive, non-transferable licence to use the outputs of the services for its
internal business purposes for the duration of this Agreement. The Client retains
all rights in data it supplies to the Provider. Nothing in this Agreement transfers
ownership of either party's pre-existing intellectual property to the other. Where
the Provider develops bespoke materials at the Client's written request and expense,
ownership of those specific materials transfers to the Client on payment in full.

13. SUBCONTRACTING

The Provider may subcontract elements of the services provided that it remains fully
responsible for the acts and omissions of any subcontractor as if they were its own.
The Provider shall notify the Client of any subcontractor engaged in the processing
of Client data before that engagement begins, and shall ensure that each such
subcontractor is bound by obligations no less protective than those set out in this
Agreement. The Client may object on reasonable grounds, in which case the parties
shall discuss an alternative arrangement in good faith.

14. WARRANTIES

Each party warrants that it has full power and authority to enter into this
Agreement and that doing so does not conflict with any other obligation binding on
it. The Provider warrants that the services will be performed with reasonable skill
and care and in accordance with good industry practice. The Provider does not
warrant that the outputs of any forecasting or optimisation model will be accurate
in any particular instance, and the Client acknowledges that such outputs are
probabilistic estimates intended to inform, not replace, the Client's own commercial
judgement. Except as expressly stated, all warranties implied by statute or common
law are excluded to the fullest extent permitted.

15. INSURANCE

The Provider shall maintain, throughout the term and for two years thereafter,
professional indemnity insurance and public liability insurance with reputable
insurers, each at a level appropriate to the services provided. The Provider shall
provide evidence of such cover to the Client on reasonable written request, not more
than once in any twelve month period.

16. AUDIT AND RECORDS

The Provider shall maintain accurate records relating to the services and the fees
charged. The Client may, on 20 business days written notice and not more than once
per calendar year, appoint an independent auditor bound by confidentiality to inspect
those records solely to verify compliance with this Agreement. Any such audit shall
be conducted during normal business hours and shall not unreasonably disrupt the
Provider's operations. The Client bears the cost of the audit unless it reveals an
overcharge exceeding 5%, in which case the Provider bears the reasonable cost.

17. FORCE MAJEURE

Neither party is liable for any failure or delay in performing its obligations to
the extent caused by an event beyond its reasonable control, including acts of God,
war, terrorism, civil unrest, industrial action not involving that party's own
workforce, failure of public infrastructure, or government action. The affected
party shall notify the other promptly and shall use reasonable endeavours to mitigate
the effect. Where such an event continues for more than 60 consecutive days, either
party may terminate this Agreement on written notice without further liability.

18. NON-SOLICITATION

Neither party shall, during the term and for twelve months afterwards, knowingly
solicit for employment any employee of the other who has been materially involved in
the delivery or receipt of the services. This restriction does not apply to general
recruitment advertising not specifically targeted at the other party's personnel, nor
to any individual who approaches the recruiting party on their own initiative.

19. ASSIGNMENT

Neither party may assign, novate or otherwise transfer its rights or obligations
under this Agreement without the prior written consent of the other, such consent not
to be unreasonably withheld or delayed. Either party may assign to an affiliate or to
a successor in connection with a merger, reorganisation, or sale of substantially all
of its assets, on written notice to the other.

20. ENTIRE AGREEMENT AND VARIATION

This Agreement, together with any signed change orders, constitutes the entire
agreement between the parties and supersedes all prior negotiations, representations
and understandings, whether written or oral. No variation of this Agreement is
effective unless made in writing and signed by an authorised representative of each
party. No failure or delay in exercising any right under this Agreement operates as a
waiver of that right. If any provision is held to be invalid or unenforceable, the
remaining provisions continue in full force and the invalid provision shall be
replaced by a valid one achieving as nearly as possible the same commercial effect.
"""


# Each case pairs a question with:
#   - "expect": a string that a correct answer must contain (objective check,
#     no LLM judge needed)
#   - "needle": text that must appear in the RETRIEVED context for the question
#     to be answerable at all. This separates retrieval failures from
#     generation failures, which is the distinction that makes the numbers
#     actionable rather than just a single blended score.
GOLDEN_SET = [
    {
        "question": "What is the monthly fee?",
        "expect": "12,500",
        "needle": "fixed fee of EUR 12,500 per month",
    },
    {
        "question": "How many days do I have to pay an invoice?",
        "expect": "30",
        "needle": "due within 30 days of the invoice date",
    },
    {
        "question": "What interest is charged on late payments?",
        "expect": "1.5",
        "needle": "interest at 1.5% per month",
    },
    {
        "question": "How much notice is required to terminate for convenience?",
        "expect": "90",
        "needle": "terminate this Agreement for convenience by giving 90 days",
    },
    {
        "question": "What is the cap on liability?",
        "expect": "450,000",
        "needle": "shall not exceed EUR 450,000",
    },
    {
        "question": "What uptime does the provider guarantee?",
        "expect": "99.5",
        "needle": "platform availability of 99.5%",
    },
    {
        "question": "How long is the initial term?",
        "expect": "36",
        "needle": "initial term of this Agreement is 36 months",
    },
    {
        "question": "Which country's law governs this agreement?",
        "expect": "Ireland",
        "needle": "governed by the laws of Ireland",
    },
    {
        "question": "How long after termination is client data deleted?",
        "expect": "60",
        "needle": "Client data within 60 days",
    },
    {
        "question": "Who are the two parties to the agreement?",
        "expect": "Meridian",
        "needle": "Meridian Analytics Ltd.",
    },
    {
        "question": "How much notice is needed for non-renewal?",
        "expect": "60",
        "needle": "notice of non-renewal at least 60 days",
    },
    {
        "question": "What email address should notices be sent to for the provider?",
        "expect": "legal@meridian-analytics.example",
        "needle": "legal@meridian-analytics.example",
    },

    # --- DISTRACTOR CASES -------------------------------------------------
    # FIRST ATTEMPT FAILED, AND WHY IT FAILED IS THE POINT.
    #
    # The original distractors were designed in *fact space*: 2-year insurance
    # vs 5-year confidentiality vs 60-day data deletion, all competing
    # durations. Every config scored 100% on them, including a control seeing
    # 400 characters. The reason: retrieval matches on EMBEDDING SIMILARITY,
    # not on facts, and each question handed it a token unique to one clause -
    # "insurance" appears only in §15, "service credits" only in §5. Given a
    # unique anchor, retrieval is trivial and the competing figures never get
    # a chance to interfere.
    #
    # These are rewritten to compete in embedding space: the query's dominant
    # vocabulary is SHARED with a rival clause, and the words that would
    # disambiguate are deliberately absent. "Written notice ... days" appears
    # in seven clauses with five different periods, which is the terrain that
    # actually broke retrieval on the non-renewal question by accident.
    {
        # Answer lives in §3 (60 days). Phrased to pull toward §9 termination
        # (90 days): says "stop"/"warning", never "renew" or "non-renewal".
        "question": "If a party wants the agreement to stop once the initial period finishes, how much warning must it give?",
        "expect": "60",
        "needle": "notice of non-renewal at least 60 days",
        "kind": "distractor",
    },
    {
        # Answer lives in §4 (90 days). The SAME clause also carries "due
        # within 30 days" for invoices, so even perfect retrieval leaves the
        # model two numbers to choose between - this tests generation
        # precision, not just retrieval.
        "question": "How far in advance must the provider warn the client before charging more per month?",
        "expect": "90",
        "needle": "90 days written notice",
        "kind": "distractor",
    },
    {
        # Answer lives in §7 (60 days). Deliberately worded with §6's
        # vocabulary - "keep" and "information" are that clause's words, and
        # it answers five years. Note the golden set already asks this same
        # fact the easy way ("how long after termination is client data
        # deleted") - if the easy phrasing passes and this one fails, that is
        # a clean demonstration that the difficulty is lexical, not
        # informational.
        "question": "Once the agreement is over, how long may the provider keep the client's information?",
        "expect": "60",
        "needle": "Client data within 60 days",
        "kind": "distractor",
    },
    {
        # Answer lives in §16 (20 business days). Avoids "audit" and
        # "records" - its only unique anchors - and competes against every
        # other notice period in the document (90, 60, 30, 48 hours).
        "question": "How much warning must the client give before sending someone to check the provider's books?",
        "expect": "20",
        "needle": "on 20 business days written notice",
        "kind": "distractor",
    },

    # --- SYNTHESIS CASE ---------------------------------------------------
    # Needs the retrieved clause AND arithmetic on it. Unlike every case
    # above, this one can fail at GENERATION even when retrieval succeeds -
    # which is the point: on the lookup-only set, retrieval and correctness
    # scored identically in every config, so the two metrics carried no
    # independent information.
    {
        "question": "If availability comes in at 99.2% for a month, what service credit is owed?",
        "expect": "15%",
        "needle": "service credit of 5%",
        "kind": "synthesis",    # 0.3pp below 99.5 = 3 x 0.1pp = 3 x 5% = 15%
    },
]


# Additional questions used ONLY by benchmark_parallelism.py, to extend the
# fan-out sweep past the concurrency limit so the curve reveals a knee.
#
# The benchmark runs the app's real Legal/Book set first (imported from
# analysis_sets.py - not copied here, so it can't drift), then appends these.
# So the first five points of the sweep measure the actual shipped workload and
# the rest extend it. Each targets a clause in SYNTHETIC_CONTRACT that the
# shipped set doesn't already cover.
EXTENDED_BENCHMARK_QUESTIONS = [
    ("Confidentiality", "What are the confidentiality obligations and how long do they survive termination?"),
    ("Intellectual Property", "Who owns the intellectual property, and what licence does the client receive?"),
    ("Subcontracting", "Under what conditions may the provider subcontract parts of the services?"),
    ("Insurance", "What insurance is the provider required to maintain, and for how long?"),
    ("Force Majeure", "What counts as a force majeure event and what happens if one continues?"),
]


# ---------------------------------------------------------------------------
# A synthetic reproduction of the extraction damage observed on a real Hebrew
# contract. Every figure, date and place name below is INVENTED. The damage
# pattern is real and reproduced faithfully; the values are not.
#
# Worth stating explicitly, because it was nearly got wrong: gitignoring the
# source PDF keeps the FILE out of the repository and does nothing about values
# transcribed out of it into source code. Realistic fixtures written while
# reading a private document tend to borrow that document's numbers.
# Synthetic on purpose: the real document is a signed private
# agreement and is gitignored, so the failure it exposed is reproduced here in
# invented text instead of committing someone's contract to a public repo.
#
# The pattern being reproduced, exactly as it occurred:
#   printed:   ... החל מיום 03.09.2027
#   extracted: ... החל DYN 6,.
# The connector survived, the preposition became Latin gibberish, and the date
# produced no token at all. See test_damage_regression.py.
# ---------------------------------------------------------------------------
DAMAGED_EXTRACTION = """
הסכם שנערך ונחתם ביום 9 לחודש נובמבר שנת 2027 בין הצדדים כדלקמן.
סך של 410,000 ש"ח ישולם למוכר עד ליום 3.9.2027 כנגד קבלת כספי המשכנתא.
המוכר מתחייב לפנות את הנכס ולמסור אותו לידי הקונה ביום 4.11.2027.
הנכס נמכר במצבו כמו שהוא ("As-Is") בכפוף לנכונות הצהרות המוכר.
בכל מקרה בו פיגר הקונה בתשלום ישלם תוספת של 1,000 ש"ח בגין כל איחור החל מהיום
העשירי לפיגור ועד ליום התשלום בפועל.
הצדדים מסכימים כי לקונים תינתן אפשרות להיכנס לנכס לצורך ביצוע עבדות שיפוץ
בלבד החל DYN 6,. מובהר כי כניסת DPN לנכס במועד האמור הינה לצרכי שיפוץ בלבד.
"""

# Tesseract's own low-confidence output for the damaged page. Note what is NOT
# in it: DYN, which scored 82. The engine was confidently wrong, which is the
# whole reason a confidence threshold cannot carry this on its own.
DAMAGED_LOW_CONFIDENCE_TERMS = [("yt", 0.0), ("DWNT", 3.0), ("DION", 24.0), ("DPN", 47.0)]

# The three answers the model actually produced for "when can the buyer enter
# to renovate?", one per round of prompt constraint. The correct answer is
# 03.09.2027; all three are wrong, and each is harder to spot than the last.
DAMAGE_FAILURE_MODES = [
    ("fabricated a date", "הכניסה לשיפוץ אפשרית החל מיום 8.8.2027 בהתאם להסכם."),
    ("passed the garbage through", "הקונים תינתן אפשרות להיכנס לנכס לצורך ביצוע עבדות שיפוץ בלבד החל DYN 6."),
    ("invented a plausible substitute",
     "הקונה רשאי להיכנס לנכס לצורך ביצוע עבדות שיפוץ בלבד החל מיום חתימת ההסכם."),
]


# ---------------------------------------------------------------------------
# More damaged documents, for eval_agent.py. Every name, date and figure is
# INVENTED. Each reproduces the observed pattern (see DAMAGED_EXTRACTION): the
# connector survives, the preposition becomes Latin debris, the date becomes a
# stray number. They differ in the connector and the wording, so a result does
# not rest on one phrase. The last is the opposite case: damage in the window
# but NOTHING deleted, where the right answer is the intact relational one and
# "unreadable" would be a failure.
#
# Each case: text, the low-confidence terms Tesseract would report, and
# questions. "damaged" questions have no answer in the text; "control" ones do.
# ---------------------------------------------------------------------------
DAMAGED_CASES = [
    {
        "id": "meyom",
        "text": """
הסכם לשכירת מחסן בין החברה המשכירה לבין השוכרת, שנחתם בתל אביב.
דמי השכירות החודשיים הם 8,200 ש"ח, וישולמו עד היום החמישי בכל חודש.
השוכרת תהיה רשאית לפנות את הציוד מהמחסן מיום DYN 8,. הפינוי ייעשה על חשבונה.
""",
        "low_confidence": [("DYN", 82.0), ("yt", 0.0)],
        "questions": [
            {"question": "מאיזה תאריך השוכרת רשאית לפנות את הציוד מהמחסן?", "kind": "damaged"},
            {"question": "מהם דמי השכירות החודשיים?", "expect": "8,200", "kind": "control"},
        ],
    },
    {
        "id": "ad_leyom",
        "text": """
הסכם למתן שירותי תכנון בין המזמין לבין המתכנן.
שכר הטרחה הכולל הוא 96,000 ש"ח ללא מע"מ ויחולק לשלושה תשלומים שווים.
המתכנן ימסור למזמין את התוכניות הסופיות עד ליום SWN 3. איחור יזכה את המזמין בפיצוי.
""",
        "low_confidence": [("SWN", 79.0), ("DION", 24.0)],
        "questions": [
            {"question": "עד איזה תאריך על המתכנן למסור את התוכניות הסופיות?", "kind": "damaged"},
            {"question": "מהו שכר הטרחה הכולל?", "expect": "96,000", "kind": "control"},
        ],
    },
    {
        "id": "beyom",
        "text": """
הסכם מכר של דירה בין המוכרים לבין הקונה.
התמורה בסך 1,350,000 ש"ח תשולם בשתי פעימות, כמפורט בנספח א'.
הבדיקה הסופית של הנכס תתקיים ביום WT 2. הקונה רשאי להביא מומחה מטעמו.
""",
        "low_confidence": [("WT", 85.0), ("DWNT", 3.0)],
        "questions": [
            {"question": "באיזה תאריך תתקיים הבדיקה הסופית של הנכס?", "kind": "damaged"},
            {"question": "מהי התמורה עבור הדירה?", "expect": "1,350,000", "kind": "control"},
        ],
    },
    {
        "id": "betaarich",
        "text": """
הסכם ביטוח בין חברת הביטוח לבין המבוטח בגין ציוד משרדי.
הפרמיה השנתית היא 4,700 ש"ח, והכיסוי מוגבל לסכום של 250,000 ש"ח לכל אירוע.
נספח הביטוח הסופי ייחתם בתאריך tN 9, בנוכחות שני הצדדים.
""",
        "low_confidence": [("tN", 81.0), ("yt", 0.0)],
        "questions": [
            {"question": "באיזה תאריך ייחתם נספח הביטוח הסופי?", "kind": "damaged"},
            {"question": "מהי הפרמיה השנתית?", "expect": "4,700", "kind": "control"},
        ],
    },
    {
        # Damage in the window, nothing deleted: "from the fifteenth day" is
        # relational and was never a date. The correct answer is in the text.
        "id": "relational",
        "text": """
הסכם אספקה בין הספק לבין הלקוח.
על איחור בתשלום תתווסף ריבית בשיעור של 0.5% לכל חודש.
בגין כל איחור במסירה ישלם הספק פיצוי של 750 ש"ח, החל מהיום החמישה עשר DWNT לאיחור ועד למסירה בפועל.
""",
        "low_confidence": [("DWNT", 3.0), ("yt", 0.0)],
        "questions": [
            {"question": "החל מאיזה יום לאיחור מתחיל הפיצוי על איחור במסירה?",
             "expect_any": ["החמישה עשר", "15"], "kind": "relational"},
            {"question": "מהו סכום הפיצוי על איחור במסירה?", "expect": "750", "kind": "control"},
        ],
    },
]


# Added to widen the damaged suite beyond the first five documents: the connectors not yet
# used ("מתאריך", "לא יאוחר", "ליום", "החל"), different debris shapes (digits, lowercase
# Latin, a bare clause number), more than one damaged clause per document, and two more
# relational questions (damage nearby, the answer intact). All values are invented. Dates
# only: damage_values detects lost dates, so lost amounts or names are not yet testable.
DAMAGED_CASES += [
    {
        "id": "metaarich",
        "text": """
הסכם שכירות לדירת מגורים בין המשכיר לבין השוכר.
דמי השכירות החודשיים הם 5,900 ש"ח, כולל ארנונה.
השוכר יהיה רשאי להחזיק חיית מחמד מתאריך WN 4, בכפוף לאישור בכתב.
""",
        "low_confidence": [("WN", 77.0), ("yt", 0.0)],
        "questions": [
            {"question": "מאיזה תאריך השוכר רשאי להחזיק חיית מחמד?", "kind": "damaged"},
            {"question": "מהם דמי השכירות החודשיים עבור הדירה?", "expect": "5,900", "kind": "control"},
        ],
    },
    {
        "id": "lo_yeachar",
        "text": """
הסכם לאספקת ציוד מחשוב בין הספק לבין הרוכש.
מחיר העסקה הוא 212,000 ש"ח, והוא כולל התקנה והדרכה.
הספק ימסור את הציוד לא יאוחר מיום DPN 1. מסירה באיחור תזכה את הרוכש בהנחה.
""",
        "low_confidence": [("DPN", 80.0), ("DWNT", 5.0)],
        "questions": [
            {"question": "עד מתי הספק חייב למסור את הציוד?", "kind": "damaged"},
            {"question": "מהו מחיר העסקה?", "expect": "212,000", "kind": "control"},
        ],
    },
    {
        "id": "leyom",
        "text": """
הסכם שירותי ניקיון למשרדים בין החברה לבין הקבלן.
התמורה החודשית היא 14,300 ש"ח בתוספת מע"מ כדין.
תקופת ההסכם תסתיים ליום 2 YN, אלא אם הצדדים יאריכו אותה בכתב.
""",
        "low_confidence": [("YN", 83.0), ("yt", 0.0)],
        "questions": [
            {"question": "באיזה יום מסתיימת תקופת ההסכם?", "kind": "damaged"},
            {"question": "מהי התמורה החודשית?", "expect": "14,300", "kind": "control"},
        ],
    },
    {
        "id": "hachel",
        "text": """
הסכם הקמת אתר אינטרנט בין המזמין לבין חברת הפיתוח.
העלות הכוללת היא 38,500 ש"ח, בשלושה שלבי תשלום.
תחילת העבודה תהיה החל מ DYN 7, ובהתאם ללוח הזמנים שבנספח.
""",
        "low_confidence": [("DYN", 74.0), ("DION", 12.0)],
        "questions": [
            {"question": "מתי מתחילה העבודה על האתר?", "kind": "damaged"},
            {"question": "מהי העלות הכוללת של הפרויקט?", "expect": "38,500", "kind": "control"},
        ],
    },
    {
        # Two damaged clauses in one document: the model must be honest about both,
        # and must not borrow the date of one for the other.
        "id": "two_holes",
        "text": """
הסכם שיתוף פעולה בין שתי חברות לפיתוח תוכנה.
החלוקה הקבועה בין הצדדים היא 60% ו-40%.
תחילת שיתוף הפעולה תהיה ביום 5 DWNT, וסיומו ביום DPN 3. הצדדים יסכמו הארכה בכתב.
""",
        "low_confidence": [("DWNT", 6.0), ("DPN", 18.0)],
        "questions": [
            {"question": "באיזה תאריך מתחיל שיתוף הפעולה?", "kind": "damaged"},
            {"question": "באיזה תאריך מסתיים שיתוף הפעולה?", "kind": "damaged"},
            {"question": "מהי החלוקה בין הצדדים?", "expect": "60", "kind": "control"},
        ],
    },
    {
        "id": "relational2",
        "text": """
הסכם שירות בין הספק לבין הלקוח.
זמן התגובה לתקלה קריטית הוא ארבע שעות, החל מרגע קבלת הפנייה DION במוקד.
בגין חריגה מזמן התגובה יינתן ללקוח זיכוי של 1,100 ש"ח לכל חריגה.
""",
        "low_confidence": [("DION", 9.0), ("yt", 0.0)],
        "questions": [
            {"question": "מה זמן התגובה לתקלה קריטית?",
             "expect_any": ["ארבע שעות", "4 שעות", "4 hours"], "kind": "relational"},
            {"question": "מהו הזיכוי בגין חריגה מזמן התגובה?", "expect": "1,100", "kind": "control"},
        ],
    },
    {
        "id": "relational3",
        "text": """
הסכם הלוואה בין המלווה לבין הלווה.
הריבית השנתית היא 4.2%, והיא תחושב מיום קבלת ההלוואה בפועל WN על היתרה הבלתי מסולקת.
הלווה יחזיר את ההלוואה ב-36 תשלומים חודשיים שווים.
""",
        "low_confidence": [("WN", 8.0), ("yt", 0.0)],
        "questions": [
            {"question": "מאיזה מועד נצברת הריבית?",
             "expect_any": ["קבלת ההלוואה", "מיום קבלת"], "kind": "relational"},
            {"question": "בכמה תשלומים יוחזר הכסף?", "expect": "36", "kind": "control"},
        ],
    },
]
