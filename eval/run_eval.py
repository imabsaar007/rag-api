"""Evaluate a running RAG API against eval/eval_set.json.

Usage:
    python eval/run_eval.py --base-url http://localhost:8000

An *answerable* question passes if the API answered and the answer contains at
least one expected phrase (case-insensitive). An *unanswerable* question passes
if the guardrail declined (``answered`` is false). Results are printed and
optionally written to JSON. Exit status is 1 if accuracy is below --min-accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

DEFAULT_SET = Path(__file__).with_name("eval_set.json")


def judge(item: dict, body: dict) -> bool:
    if not item.get("answerable", True):
        return body["answered"] is False
    if not body["answered"]:
        return False
    answer = body["answer"].lower()
    return any(phrase.lower() in answer for phrase in item["expect_any"])


def run(base_url: str, eval_path: Path, timeout: float) -> dict:
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    base = base_url.rstrip("/")

    for doc in data["documents"]:
        r = requests.post(f"{base}/documents", json=doc, timeout=timeout)
        r.raise_for_status()

    results = []
    for item in data["questions"]:
        r = requests.post(
            f"{base}/query", json={"question": item["question"]}, timeout=timeout
        )
        r.raise_for_status()
        body = r.json()
        results.append(
            {
                "question": item["question"],
                "answerable": item.get("answerable", True),
                "passed": judge(item, body),
                "answered": body["answered"],
                "answer": body["answer"],
                "top_score": body["sources"][0]["score"] if body["sources"] else None,
            }
        )

    passed = sum(r["passed"] for r in results)
    return {"passed": passed, "total": len(results), "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_SET)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path, help="write full results to this JSON file")
    parser.add_argument("--min-accuracy", type=float, default=0.0)
    args = parser.parse_args()

    report = run(args.base_url, args.eval_set, args.timeout)

    for r in report["results"]:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{mark}] {r['question']}")
        if not r["passed"]:
            print(f"       answered={r['answered']} top_score={r['top_score']} -> {r['answer']}")

    accuracy = report["passed"] / report["total"]
    print(f"\nAccuracy: {report['passed']}/{report['total']} ({accuracy:.0%})")

    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if accuracy >= args.min_accuracy else 1


if __name__ == "__main__":
    sys.exit(main())
