import json
import csv
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from rag_core import LIGJBOTRAG  # your RAG engine

load_dotenv()

QUESTIONS_FILE = "eval_questions.json"
RESULTS_CSV = "eval_results.csv"
JUDGE_MODEL = "llama-3.1-8b-instant"  # same model is fine


JUDGE_PROMPT = """
Ti je një vlerësues strikt i një asistenti juridik.

Do të marrësh:
- një PYETJE nga qytetari,
- një KONTEKST (copëza nga ligjet),
- një PËRGJIGJE të sistemit.

Detyra jote është të japësh një vlerësim numerik si JSON:
{{
  "answer_relevancy": 0.0-1.0,   // sa mirë përgjigjja i përgjigjet pyetjes
  "faithfulness": 0.0-1.0,       // sa shumë përgjigjja mbështetet në kontekst, pa shpikje
  "comment": "një koment i shkurtër në shqip pse"
}}

Udhëzime:
- answer_relevancy = 1.0 nëse përgjigjja është e qartë, i përgjigjet direkt pyetjes, pa devijime të mëdha.
- faithfulness = 1.0 nëse çdo fakt kryesor në përgjigje gjendet qartë në kontekst.
- Nëse pyetja nuk mbulohet në kontekst, faithfulness duhet të jetë 0.0, edhe nëse përgjigjja duket e saktë.
- Kthe VETËM JSON të vlefshëm, pa tekst shtesë.

PYETJA:
{question}

KONTEKSTI:
{context}

PËRGJIGJJA:
{answer}
"""


def build_context_from_docs(docs: List) -> str:
    """Thjeshton dokumentet në një tekst konteksti për gjykuesin."""
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        law_num = meta.get("law_number", "E panjohur")
        article = meta.get("article_number", "")
        page = meta.get("page_number", "?")
        content = doc.page_content
        if len(content) > 700:
            content = content[:700].rstrip() + "..."
        header = f"[{i}] {law_num}"
        if article:
            header += f" | {article}"
        header += f" | Faqe {page}"
        parts.append(f"{header}\n{content}")
    return "\n\n".join(parts)


def load_questions(path: str) -> list:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def run_eval():
    questions = load_questions(QUESTIONS_FILE)
    print(f"Gjeta {len(questions)} pyetje për evaluim.")

    # 1) Inicializo RAG-in
    bot = LIGJBOTRAG()
    if not bot.initialize(skip_llm=False):
        print("❌ Nuk u inicializua LIGJBOTRAG për evaluim.")
        return

    # 2) Inicializo LLM-in gjykues
    judge = ChatGroq(model=JUDGE_MODEL)

    rows = []
    for idx, q in enumerate(questions, 1):
        print(f"\n[{idx}/{len(questions)}] Pyetja: {q}")
        # pyet sistemin
        result = bot.ask(q)
        answer = result.get("answer", "")
        docs = result.get("docs") or []
        context_text = build_context_from_docs(docs)

        # përgatit promptin për gjykuesin
        prompt_text = JUDGE_PROMPT.format(
            question=q,
            context=context_text,
            answer=answer,
        )

        judge_resp = judge.invoke([HumanMessage(content=prompt_text)])
        raw = judge_resp.content.strip()

        try:
            scores = json.loads(raw)
            rel = float(scores.get("answer_relevancy", 0.0))
            faith = float(scores.get("faithfulness", 0.0))
            comment = str(scores.get("comment", "")).replace("\n", " ")
        except Exception:
            rel = 0.0
            faith = 0.0
            comment = f"PARSE_ERROR: {raw[:200]}"
        print(f"  → answer_relevancy={rel:.2f}, faithfulness={faith:.2f}")

        rows.append({
            "question": q,
            "answer": answer,
            "answer_relevancy": rel,
            "faithfulness": faith,
            "comment": comment,
        })

    # 3) Ruaj në CSV
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "answer", "answer_relevancy", "faithfulness", "comment"],
        )
        writer.writeheader()
        writer.writerows(rows)

    avg_rel = sum(r["answer_relevancy"] for r in rows) / len(rows)
    avg_faith = sum(r["faithfulness"] for r in rows) / len(rows)
    print("\n✅ Evaluimi mbaroi.")
    print(f"Mesatarja answer_relevancy : {avg_rel:.2f}")
    print(f"Mesatarja faithfulness     : {avg_faith:.2f}")
    print(f"Rezultatet u ruajtën te {RESULTS_CSV}")


if __name__ == "__main__":
    run_eval()
