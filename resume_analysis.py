"""The résumé path: one call, whole document, structured output.

Lifted out of app.py for a reason worth stating. This was the only branch in
the application that had never once been executed — and the reason is visible
in the old import graph: it lived inside a Streamlit script, so running it at
all meant launching the UI, uploading a file and clicking a button. Code that
can only be reached through a browser does not get tested, and untested code in
a demo gets discovered by the person you are demoing to.

Everything else in this pipeline had already been pulled out for exactly this
reason (extraction, verification, damaged_values, analysis_sets); this was the
one that got missed.
"""
from typing import List

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class ResumeAnalysis(BaseModel):
    """Structured output schema for the résumé path.

    A schema rather than free text keeps the score a real field instead of
    prose to be parsed — though it is still a model estimate, not a calibrated
    metric, and the UI says so rather than presenting it as a measurement.
    """
    match_score: int = Field(description="0-100 estimate of how well the resume matches the job description")
    rationale: str = Field(description="Two or three sentences explaining the score")
    action_items: List[str] = Field(description="Three concrete, specific changes that would improve the match")
    missing_keywords: List[str] = Field(description="Up to five skills or technologies present in the job description but absent from the resume")


def run_resume_analysis(jd: str, resume_text: str) -> ResumeAnalysis:
    """Deliberately uses NEITHER retrieval NOR parallelism.

    A résumé fits comfortably in a context window, so chunking it and
    retrieving top-k actively discards information — and "which skills are
    missing from this résumé?" is unanswerable from partial text, since the
    model cannot distinguish "absent from the document" from "absent from the
    three chunks I happened to retrieve." One call, whole document, structured
    output. Knowing where NOT to apply the technique is the point.
    """
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(ResumeAnalysis)
    return structured_llm.invoke(
        "You are an expert technical recruiter. Compare the candidate's complete "
        "résumé against the job description and return a structured assessment.\n\n"
        f"=== JOB DESCRIPTION ===\n{jd}\n\n"
        f"=== RÉSUMÉ (complete text) ===\n{resume_text}"
    )
