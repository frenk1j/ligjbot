"""
LIGJBOT — Flask Web Interface
ChatGPT-style chat UI for the Albanian Legal RAG system.
"""

import os
import sys
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from dotenv import load_dotenv

load_dotenv()

# Add src to path so rag_core can be imported
sys.path.insert(0, str(Path(__file__).parent))

from rag_core import LIGJBOTRAG

app = Flask(__name__)

# Singleton RAG instance (loaded once at startup)
bot = LIGJBOTRAG()
bot_ready = False
bot_error = None
answer_cache = {}
FAST_MODE = os.getenv("FAST_MODE", "1") == "1"


def init_bot():
    global bot_ready, bot_error
    try:
        result = bot.initialize()
        bot_ready = result
        if not result:
            bot_error = "Nuk u inicializua. Kontrolloni API key dhe vector store."
    except Exception as e:
        bot_ready = False
        bot_error = str(e)


@app.route("/")
def index():
    return render_template("index.html")


PDF_FOLDER = os.getenv("PDF_FOLDER", "./data/pdfs")


@app.route("/pdfs/<path:filename>")
def serve_pdf(filename):
    folder = Path(PDF_FOLDER).resolve()
    return send_from_directory(folder, filename)


@app.route("/api/status")
def status():
    return jsonify({
        "ready": bot_ready,
        "error": bot_error,
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    if not bot_ready:
        return jsonify({
            "error": bot_error or "LIGJBOT nuk është gati akoma."
        }), 503

    data = request.get_json()
    question = (data or {}).get("message", "").strip()

    if not question:
        return jsonify({"error": "Mesazhi është bosh."}), 400

    try:
        start = time.perf_counter()
        cache_key = question.lower()
        if cache_key in answer_cache:
            cached = dict(answer_cache[cache_key])
            cached["cached"] = True
            return jsonify(cached)

        if FAST_MODE:
            # Fast path: retrieval-only response to avoid long LLM latency/timeouts.
            docs = bot.search_relevant_chunks(question)
            if not docs:
                return jsonify({
                    "answer": "Nuk gjeta informacion relevant ne ligjet e indexuara per kete pyetje.",
                    "sources": [],
                    "chunks_used": 0,
                    "cached": False,
                    "response_ms": int((time.perf_counter() - start) * 1000),
                    "mode": "fast",
                })
            snippets = []
            for i, d in enumerate(docs[:2], 1):
                content = d.page_content
                if content.startswith("passage: "):
                    content = content[9:]
                if "---" in content:
                    content = content.split("---", 1)[-1].strip()
                clean = " ".join(content.split())
                snippets.append(f"{i}. {clean[:280]}...")
            result = {
                "answer": "Nga ligjet e indexuara gjeta keto pjese relevante:\n\n" + "\n".join(snippets),
                "docs": docs,
                "chunks_used": len(docs),
            }
        else:
            result = bot.ask(question)
            if "error" in result:
                return jsonify({"error": result["error"]}), 500

        # Build structured sources list for clickable links
        structured_sources = []
        seen = set()
        for doc in result.get("docs", []):
            meta = doc.metadata
            law_num  = meta.get("law_number", "?")
            article  = meta.get("article_number", "")
            page     = meta.get("page_number", 1)
            src_file = meta.get("source_file", "")
            key = f"{src_file}_{article}_{page}"
            if key not in seen:
                seen.add(key)
                label = article if article else law_num
                if not article:
                    label = law_num
                structured_sources.append({
                    "label": f"{label}, {law_num}" if article else law_num,
                    "file": src_file,
                    "page": page,
                    "url": f"/pdfs/{src_file}#page={page}" if src_file else None,
                })

        payload = {
            "answer": result["answer"],
            "sources": structured_sources,
            "chunks_used": result["chunks_used"],
            "cached": False,
            "response_ms": int((time.perf_counter() - start) * 1000),
            "mode": "fast" if FAST_MODE else "llm",
        }
        # lightweight in-memory cache for repeated questions
        answer_cache[cache_key] = payload
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/suggestions")
def suggestions():
    return jsonify([
        "Sa mund të më mbajë policia pa vendim gjykate?",
        "Sa është gjoba për celular gjatë ngasjes?",
        "A lejohet policia të kontrollojë pa arsye?",
        "Çfarë dokumentash duhet të kem gjithmonë në makinë?",
        "Si mund të ankoj një gjobë policore?",
        "Çfarë ndodh nëse refuzoj alkool-testin?",
        "Cilat janë të drejtat e mia kur kryqëzohem në kufi?",
        "What rights do I have if police stop me?",
    ])


if __name__ == "__main__":
    print("🚀 Duke nisur LIGJBOT Web Interface...")
    init_bot()
    app.run(debug=False, host="0.0.0.0", port=5001, threaded=True)
