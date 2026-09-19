"""Runs golden_dataset.py's cases against a live backend and reports pass/fail.

Requires the backend running (uvicorn backend.app.main:app) with the 4 projects the
dataset expects already ingested: Lodha Regalia, Golden Willows at Hiranandani Fortune
City, Ashwini Elite, Codename Connexa.

Usage:
    python -m eval.run_eval
"""

import json
import os
import sys

import requests

from .golden_dataset import GOLDEN_CASES

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


def run_case(case: dict) -> dict:
    resp = requests.post(f"{BACKEND_URL}/query", json={"question": case["question"]}, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    haystack = f"{data.get('answer_text', '')} {json.dumps(data.get('table_data') or '')}".lower()

    if case.get("manual_review"):
        return {"case": case, "data": data, "status": "REVIEW", "failures": []}

    failures = []

    expected_intent = case.get("expected_intent")
    if expected_intent and data.get("intent") != expected_intent:
        failures.append(f"intent: expected {expected_intent!r}, got {data.get('intent')!r}")

    for phrase in case.get("must_contain", []):
        if phrase.lower() not in haystack:
            failures.append(f"missing expected phrase: {phrase!r}")

    one_of = case.get("must_contain_one_of")
    if one_of and not any(p.lower() in haystack for p in one_of):
        failures.append(f"missing any phrase from: {one_of}")

    for phrase in case.get("must_not_contain", []):
        if phrase.lower() in haystack:
            failures.append(f"unexpectedly contains: {phrase!r}")

    if case.get("must_have_table") and not data.get("table_data"):
        failures.append("expected non-empty table_data, got none")

    return {"case": case, "data": data, "status": "FAIL" if failures else "PASS", "failures": failures}


def main() -> None:
    try:
        requests.get(f"{BACKEND_URL}/documents", timeout=5).raise_for_status()
    except requests.RequestException as exc:
        print(f"Cannot reach backend at {BACKEND_URL}: {exc}")
        print("Start it with: uvicorn backend.app.main:app --reload")
        sys.exit(2)

    results = [run_case(c) for c in GOLDEN_CASES]

    for r in results:
        print(f"[{r['status']}] {r['case']['question']}")
        if r["status"] == "FAIL":
            for f in r["failures"]:
                print(f"       - {f}")
        elif r["status"] == "REVIEW":
            print(f"       -> {r['data']['answer_text'][:200]}")

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    review = sum(1 for r in results if r["status"] == "REVIEW")

    print()
    print(f"{passed} passed, {failed} failed, {review} need manual review, out of {len(results)} cases")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
