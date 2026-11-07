"""Smoke test for judge_experiment: case selection and the separation verdict, no API.

    python3 test_judge_experiment.py
"""
import io
import json
import os
import sys
import tempfile

import judge_experiment as j
from eval_agent import build_suites

ctx, cases = build_suites()["damaged"][-2]
Q = ["זמן התגובה לתקלה קריטית הוא ארבע שעות, החל מרגע קבלת הפנייה DION במוקד."]


def run(a, out):
    return {"answer": a, "outcome": out, "replies": [{"answer": a, "quotes": Q}]}


row = {"suite": "damaged", "question": cases[0]["question"], "kind": "relational", "repeat": 0,
       "baseline": run("ארבע שעות", "pass"), "loop": run("המסירה בפועל.", "fail")}
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
    json.dump({"damaged": [row]}, f, ensure_ascii=False)

found = j.ambiguous_cases(json.load(open(f.name, encoding="utf-8")))
results = [[c[1] for c in found] == ["ארבע שעות", "המסירה בפועל."], [c[3] for c in found] == [True, False]]


def verdict_for(judge):
    buf = []
    j.main([f.name], judge=judge, out=buf.append)
    return buf[-1]


results.append("clean separation" in verdict_for(lambda q, a, qs: "correct" if "שעות" in a else "incorrect"))
results.append("NOT clean" in verdict_for(lambda q, a, qs: "correct"))
os.unlink(f.name)
for i, ok in enumerate(results, 1):
    print(f"[{'PASS' if ok else 'FAIL'}] {i}")
print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
