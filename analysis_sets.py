"""
Analysis question sets, kept separate from app.py so they can be imported
without starting Streamlit - which is what lets the benchmark and eval
scripts run the *real* question sets rather than a copy that drifts.
"""

# Each question is fired as its OWN call with its OWN retrieval. They are not
# merged into one prompt because they need different chunks: "payment terms"
# and "termination conditions" live in different parts of a long contract, so
# a single shared retrieval would serve all of them badly. That's what makes
# the concurrency here architectural rather than decorative.
ANALYSIS_QUESTION_SETS = {
    "Legal/Book": [
        ("Parties & Roles", "Who are the parties to this agreement, and what is each party's role and core obligations?"),
        ("Payment Terms", "What are the payment terms — amounts, schedule, currency, and any penalties or interest?"),
        ("Dates & Deadlines", "What are the key dates, deadlines, duration, and renewal terms?"),
        ("Termination", "Under what conditions can this agreement be terminated, and what notice is required?"),
        ("Risks & Unusual Clauses", "Identify any unusual, one-sided, or high-risk clauses a reviewer should pay attention to."),
    ],
    "Technical/Code": [
        ("Architecture", "What is the overall architecture, and how are the main components organized?"),
        ("Dependencies", "What external dependencies, libraries, or services does this rely on?"),
        ("Entry Points", "What are the main entry points, interfaces, or public APIs?"),
        ("Risks & Gaps", "What are the notable risks, limitations, TODOs, or known issues?"),
    ],
    "General": [
        ("Summary", "What is this document about — its purpose and main subject?"),
        ("Key Facts", "What are the most important facts, figures, and data points?"),
        ("Entities", "Who are the key people, organizations, and entities mentioned?"),
        ("Dates", "What dates, deadlines, or time-sensitive items appear?"),
    ],
}
