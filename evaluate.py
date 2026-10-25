"""Kichik, ochiq sinov to'plami. Bu umumiy aniqlik kafolati emas."""
import argparse
import json
from pathlib import Path
import sys
import time

from honda_rag.retrieval import Retriever
from honda_rag import config as c

CASES = [
    {"question": "What engine oil is recommended?", "pages": [664], "fact": "0w-20"},
    {"question": "Qanday motor moyi tavsiya etiladi?", "pages": [664], "fact": "0w-20"},
    {"question": "What fuel octane rating is recommended?", "pages": [32, 642], "fact": "87"},
    {"question": "Benzin qanday oktanli bolishi kerak?", "pages": [32, 642], "fact": "91"},
    {"question": "What is the fuel tank capacity?", "pages": [32, 642, 756, 758], "fact": "46.9"},
    {"question": "Benzin bakining sigimi qancha?", "pages": [32, 642, 756, 758], "fact": "46.9"},
    {"question": "How often should tire pressure be checked?", "pages": [508, 683], "fact": "month"},
    {"question": "Shinalar bosimini qachon tekshirish kerak?", "pages": [508, 683], "fact": "month"},
    {"question": "Which brake fluid is specified?", "pages": [674, 757, 759], "fact": "dot 3"},
    {"question": "What engine coolant should I use?", "pages": [671], "fact": "type 2"},
    {"question": "When should I check the engine oil level?", "pages": [665], "fact": "refuel"},
    {"question": "How do I calibrate the TPMS?", "pages": [504, 505, 506, 507], "fact": "calibrat"},
]

GENERATION = [
    {"question": "Qanday motor moyi tavsiya etiladi?", "language": "uz", "facts": ["0w-20"]},
    {"question": "Benzin bakining sigimi qancha?", "language": "uz", "facts": ["46.9"]},
    {"question": "How often should tire pressure be checked?", "language": "en", "facts": ["month"]},
    {"question": "What is the exact retail price of a 2030 Honda Civic in Uzbekistan?",
     "language": "en", "abstain": True},
]


def run(generate=False):
    start = time.monotonic()
    retriever = Retriever()
    results = []
    for case in CASES:
        found = retriever.search(case["question"])
        hit = any(x["pdf_page"] in case["pages"] and case["fact"] in x["text"].lower() for x in found)
        result = {"question": case["question"], "hit_at_5_with_fact": hit,
                  "expected_pages": case["pages"], "found_pages": [x["pdf_page"] for x in found]}
        results.append(result)
        print(f"{'PASS' if hit else 'FAIL'} retrieval: {case['question']}", flush=True)
    retriever.close()
    report = {"backend": "openrouter", "chat_model": c.CHAT_MODEL, "embedding_model": c.EMBEDDING_MODEL,
              "retrieval": results, "hit_at_5_with_fact": sum(x["hit_at_5_with_fact"] for x in results) / len(results),
              "note": "Small development set; not an independent benchmark or a semantic accuracy score."}
    if generate:
        from honda_rag.pipeline import RAG
        generated = []
        with RAG() as rag:
            for case in GENERATION:
                answer = rag.ask(case["question"], case["language"])
                passed = (not answer["supported"] if case.get("abstain") else
                          answer["supported"] and all(fact in answer["answer"].lower() for fact in case["facts"]))
                generated.append({"question": case["question"], "pass": passed, "result": answer})
                print(f"{'PASS' if passed else 'FAIL'} generation: {case['question']}", flush=True)
                print(answer["answer"], flush=True)
        report["generation"] = generated
    report["elapsed_seconds"] = round(time.monotonic() - start, 2)
    path = Path(__file__).resolve().parent / "openrouter-test-results.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nHit@5 with expected fact: {report['hit_at_5_with_fact']:.0%} ({len(results)} savol)")
    print(f"Batafsil natija: {path}")
    passed = all(x["hit_at_5_with_fact"] for x in results) and all(x["pass"] for x in report.get("generation", []))
    return 0 if passed else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="OpenRouter javob modelini ham tekshirish (API xarajati bor)")
    try:
        sys.exit(run(parser.parse_args().generate))
    except RuntimeError as exc:
        print(f"Xato: {exc}", file=sys.stderr)
        sys.exit(1)
