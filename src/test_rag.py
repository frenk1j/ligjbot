"""
====================================================
TEST RAG — Teston RAG pipeline automatikisht
LigjetBot - Faza 2

Perdorim:
  python src/test_rag.py          # Te gjitha testet
  python src/test_rag.py --quick  # 3 teste te shpejta
====================================================
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Test Cases ─────────────────────────────────────────────────────────────
TEST_CASES = [
    {
        "id": "T01",
        "category": "E drejta gjate ndalimit",
        "query": "Sa ore mund te me mbaje policia pa vendim gjykate?",
        "expected_keywords": ["5 ore", "neni 19", "82/2024"],
        "expected_law": "82/2024"
    },
    {
        "id": "T02",
        "category": "Gjobat rrugore",
        "query": "Sa eshte gjoba per perdorimin e telefonit gjate ngasjes?",
        "expected_keywords": ["celular", "gjobe", "neni 172"],
        "expected_law": "8378"
    },
    {
        "id": "T03",
        "category": "Kontrolli policor",
        "query": "A mund te me kontrolloje policia pa arsye?",
        "expected_keywords": ["kontroll", "neni 22", "procesverbal"],
        "expected_law": "82/2024"
    },
    {
        "id": "T04",
        "category": "Identifikimi",
        "query": "Kur ka te drejte policia te verifikoje identitetin tim?",
        "expected_keywords": ["identitet", "neni 18", "dyshim"],
        "expected_law": "82/2024"
    },
    {
        "id": "T05",
        "category": "Kufiri",
        "query": "Cilat jane rregullat per kalimin e kufirit?",
        "expected_keywords": ["kufi", "neni 13", "kalimit"],
        "expected_law": "39/2025"
    },
    {
        "id": "T06",
        "category": "Gjoba - Apelim",
        "query": "Si mund te ankoj nje gjobe policore?",
        "expected_keywords": ["ankim", "gjobe"],
        "expected_law": None  # Flexible
    },
    {
        "id": "T07",
        "category": "Alkool test",
        "query": "Cfare ndodh nese refuzoj alkool-testin?",
        "expected_keywords": ["alkool", "refuzoj"],
        "expected_law": None  # Flexible
    },
    {
        "id": "T08",
        "category": "Dokumentat e mjetit",
        "query": "Cilat dokumente duhet te kem gjithmone ne makine?",
        "expected_keywords": ["dokument", "mjet"],
        "expected_law": None  # Flexible
    },
]

QUICK_TESTS = TEST_CASES[:3]


def run_tests(quick: bool = False):
    """Ekzekuton te gjitha testet dhe raporton rezultatin."""

    print("\n" + "="*60)
    print("  LigjetBot — Test Suite")
    print("  Faza 2: RAG Core me Gemini")
    print("="*60)

    # Import RAG
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from rag_core import LigjetBotRAG
    except ImportError as e:
        print(f"❌ Import error: {e}")
        sys.exit(1)

    # Inicializo
    bot = LigjetBotRAG()
    if not bot.initialize():
        sys.exit(1)

    tests = QUICK_TESTS if quick else TEST_CASES
    mode  = "QUICK (3 teste)" if quick else f"FULL ({len(tests)} teste)"

    print(f"\n📋 Mode: {mode}\n")

    # Rezultatet
    results = []
    passed  = 0
    failed  = 0

    for test in tests:
        print(f"[{test['id']}] {test['category']}")
        print(f"  Query: {test['query']}")

        start_time = time.time()
        result = bot.ask(test["query"])
        elapsed = time.time() - start_time

        answer_lower = result["answer"].lower()

        # Kontrollo keywords
        keywords_found = []
        keywords_missing = []
        for kw in test["expected_keywords"]:
            if kw.lower() in answer_lower:
                keywords_found.append(kw)
            else:
                keywords_missing.append(kw)

        # Kontrollo ligjin
        law_found = True
        if test["expected_law"]:
            law_found = test["expected_law"] in result["sources"]

        # Pass/Fail
        keyword_score = len(keywords_found) / len(test["expected_keywords"])
        test_passed   = keyword_score >= 0.5  # 50% keywords mjafton

        status = "✅ PASS" if test_passed else "⚠️  PARTIAL"
        if test_passed:
            passed += 1
        else:
            failed += 1

        print(f"  Status: {status} ({elapsed:.1f}s)")
        print(f"  Keywords: {len(keywords_found)}/{len(test['expected_keywords'])} gjetur")
        if keywords_missing:
            print(f"  Mungojne: {keywords_missing}")
        print(f"  Pergjigje (100 fjalett e para): {result['answer'][:100]}...")
        print()

        results.append({
            "id":            test["id"],
            "category":      test["category"],
            "passed":        test_passed,
            "elapsed_sec":   round(elapsed, 2),
            "keyword_score": round(keyword_score, 2),
            "chunks_used":   result["chunks_used"],
            "answer_preview": result["answer"][:150]
        })

    # Summary
    print("="*60)
    print(f"  REZULTATI FINAL")
    print("="*60)
    print(f"  ✅ Kalsuan : {passed}/{len(tests)}")
    print(f"  ⚠️  Partial : {failed}/{len(tests)}")
    print(f"  📊 Score   : {passed/len(tests)*100:.0f}%")
    avg_time = sum(r["elapsed_sec"] for r in results) / len(results)
    print(f"  ⏱️  Koha mes.: {avg_time:.1f} sek/pyetje")
    print("="*60)

    # Ruaj raportin
    report_path = "./vector_store/test_report.json"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total": len(tests),
            "passed": passed,
            "failed": failed,
            "score_pct": round(passed/len(tests)*100, 1),
            "avg_time_sec": round(avg_time, 2),
            "model": os.getenv("LLM_MODEL", "gemini-2.0-flash"),
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n📊 Raport i ruajtur: {report_path}")

    if passed == len(tests):
        print("\n🎉 TE GJITHA TESTET KALUAN! RAG Core funksionon perfekt.")
    elif passed >= len(tests) * 0.7:
        print("\n✅ RAG Core funksionon mire. Disa pergjigje mund te permiressohen.")
    else:
        print("\n⚠️  Disa teste deshtuan. Kontrollo vector store dhe API key.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LigjetBot Test Suite")
    parser.add_argument("--quick", action="store_true", help="Ekzekuto 3 teste te shpejta")
    args = parser.parse_args()
    run_tests(quick=args.quick)
