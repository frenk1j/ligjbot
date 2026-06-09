"""
LIGJBOT — Flask Web Interface
ChatGPT-style chat UI for the Albanian Legal RAG system.
"""

import os
import sys
import time
import socket
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from online_ingest import load_pdf_as_documents

UPLOAD_FOLDER = "./uploaded_pdfs"
ALLOWED_EXTENSIONS = {"pdf"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

load_dotenv()

# Add src to path so rag_core can be imported
sys.path.insert(0, str(Path(__file__).parent))

from rag_core import LIGJBOTRAG, FAST_TOP_K

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["TEMPLATES_AUTO_RELOAD"] = os.getenv("FLASK_DEBUG", "0") == "1"
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = int(os.getenv("STATIC_CACHE_SECONDS", "31536000"))

# Singleton RAG instance (loaded once at startup)
bot = LIGJBOTRAG()
bot_ready = False
bot_error = None
bot_loading = False

answer_cache: dict[str, dict] = {}
FAST_MODE = os.getenv("FAST_MODE", "1") == "1"
CACHE_MAX = int(os.getenv("ANSWER_CACHE_MAX", "128"))

# -------- NEW: per-user server-side memory --------
user_histories: dict[str, list[dict]] = {}
MAX_TURNS = int(os.getenv("MAX_TURNS_HISTORY", "5"))  # Q/A pairs per user
# --------------------------------------------------

SUGGESTIONS = [
    "Sa mund të më mbajë policia pa vendim gjykate?",
    "Sa është gjoba për celular gjatë ngasjes?",
    "A lejohet policia të kontrollojë pa arsye?",
    "Çfarë dokumentash duhet të kem gjithmonë në makinë?",
    "Si mund të ankoj një gjobë policore?",
    "Çfarë ndodh nëse refuzoj alkool-testin?",
    "Cilat janë të drejtat e mia kur kryqëzohem në kufi?",
    "What rights do I have if police stop me?",
]


def _lan_ip() -> str | None:
    """Gjen IP-në lokale (WiFi) për qasje nga telefoni."""
    candidates: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(0.5)
            s.connect(("8.8.8.8", 80))
            candidates.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip.startswith("127.") or ip.startswith("169.254."):
                continue
            candidates.append(ip)
    except OSError:
        pass
    seen: set[str] = set()
    ordered: list[str] = []
    for ip in candidates:
        if ip not in seen:
            seen.add(ip)
            ordered.append(ip)
    for ip in ordered:
        if ip.startswith(("192.168.", "10.")) or ip.startswith("172."):
            return ip
    return ordered[0] if ordered else None


def access_urls(port: int) -> dict[str, str | None]:
    public = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    local = f"http://127.0.0.1:{port}"
    if public:
        return {
            "local_url": public,
            "mobile_url": public,
            "telefon_page": f"{public}/telefon",
        }
    lan = _lan_ip()
    mobile = f"http://{lan}:{port}" if lan else None
    return {
        "local_url": local,
        "mobile_url": mobile,
        "telefon_page": f"{local}/telefon",
    }


def print_access_urls(port: int) -> None:
    urls = access_urls(port)
    print()
    print("=" * 55)
    print(f"  🌐 Kompjuter:       {urls['local_url']}")
    if urls["mobile_url"]:
        print(f"  📱 iPhone/Android:  {urls['mobile_url']}")
        print(f"  📷 QR / udhëzime:   {urls['telefon_page']}")
        print()
        print("  Si ta hapësh në telefon:")
        print("  1. Telefoni dhe Mac në të njëjtin WiFi")
        print("  2. Hape linkun 📱 ose skano QR te /telefon")
    else:
        print("  ⚠️  Nuk u gjet IP WiFi — lidhu me WiFi dhe rinis.")
    print("=" * 55)
    print()


def init_bot() -> None:
    global bot_ready, bot_error, bot_loading
    bot_loading = True
    try:
        result = bot.initialize(skip_llm=FAST_MODE)
        bot_ready = result
        if not result:
            bot_error = "Nuk u inicializua. Kontrolloni API key dhe vector store."
    except Exception as e:
        bot_ready = False
        bot_error = str(e)
    finally:
        bot_loading = False


def start_bot_background() -> None:
    threading.Thread(target=init_bot, daemon=True).start()

# -------- Upload PDF route --------
@app.route("/api/upload_pdf", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    try:
        docs = load_pdf_as_documents(save_path)
        bot.add_documents(docs)
    except Exception as e:
        print(f"[WARN] Ingestion e PDF deshtoi: {e}")
        return jsonify({"error": "PDF u ruajt, por nuk u indeksua."}), 500

    return jsonify({
        "status": "ok",
        "filename": filename,
        "chunks_added": len(docs),
    })
# ----------------------------------

def firebase_config():
    cfg = {
        "apiKey":            os.getenv("FIREBASE_API_KEY", ""),
        "authDomain":        os.getenv("FIREBASE_AUTH_DOMAIN", ""),
        "projectId":         os.getenv("FIREBASE_PROJECT_ID", ""),
        "storageBucket":     os.getenv("FIREBASE_STORAGE_BUCKET", ""),
        "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID", ""),
        "appId":             os.getenv("FIREBASE_APP_ID", ""),
    }
    return cfg if cfg["apiKey"] else None


@app.route("/")
def index():
    port = int(os.getenv("PORT", "5001"))
    urls = access_urls(port)
    return render_template(
        "index.html",
        suggestions=SUGGESTIONS,
        mobile_url=urls["mobile_url"],
        telefon_page=urls["telefon_page"],
        firebase=firebase_config(),
    )


@app.route("/telefon")
def telefon():
    port = int(os.getenv("PORT", "5001"))
    urls = access_urls(port)
    return render_template(
        "telefon.html",
        mobile_url=urls["mobile_url"],
        local_url=urls["local_url"],
    )


@app.route("/api/connect")
def connect_info():
    port = int(os.getenv("PORT", "5001"))
    urls = access_urls(port)
    return jsonify({
        "local_url": urls["local_url"],
        "mobile_url": urls["mobile_url"],
        "telefon_page": urls["telefon_page"],
        "hint": (
            "Hape mobile_url në Safari (iPhone) ose Chrome (Android). "
            "Duhet i njëjti WiFi si kompjuteri."
        ) if urls["mobile_url"] else "Lidhu me WiFi dhe rinis serverin.",
    })


@app.route("/manifest.webmanifest")
def manifest():
    return send_from_directory(
        app.static_folder,
        "manifest.webmanifest",
        mimetype="application/manifest+json",
    )


PDF_FOLDER = os.getenv("PDF_FOLDER", "./data/pdfs")


@app.route("/pdfs/<path:filename>")
def serve_pdf(filename):
    folder = Path(PDF_FOLDER).resolve()
    return send_from_directory(folder, filename)


@app.route("/api/status")
def status():
    return jsonify({
        "ready": bot_ready,
        "loading": bot_loading,
        "error": bot_error,
        "fast_mode": FAST_MODE,
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    if bot_loading and not bot_ready:
        return jsonify({
            "error": "LIGJBOT po ngarkohet. Provo përsëri për disa sekonda."
        }), 503

    if not bot_ready:
        return jsonify({
            "error": bot_error or "LIGJBOT nuk është gati akoma."
        }), 503

    data = request.get_json() or {}
    question = (data.get("message") or "").strip()
    user_id = data.get("user_id") or "anonymous"

    if not question:
        return jsonify({"error": "Mesazhi është bosh."}), 400

    # get server-side history for this user
    history = user_histories.get(user_id, [])

    try:
        start = time.perf_counter()
        cache_key = question.lower()
        if bot_ready and bot.is_likely_legal(question) and cache_key in answer_cache:
            cached = dict(answer_cache[cache_key])
            cached["cached"] = True
            return jsonify(cached)

        mode = "llm"
        if not bot.is_likely_legal(question):
            result = bot.ask_casual(question, history=history)
            mode = "casual"
            if "error" in result:
                return jsonify({"error": result["error"]}), 500
        elif FAST_MODE:
            mode = "fast"
            docs = bot.search_relevant_chunks(question, k=FAST_TOP_K)
            elapsed = time.perf_counter() - start
            if not docs:
                return jsonify({
                    "answer": "Hmm, s'e gjeta këtë te ligjet që kam. Provo ta pyesësh pak ndryshe?",
                    "sources": [],
                    "chunks_used": 0,
                    "cached": False,
                    "response_s": round(elapsed, 2),
                    "mode": "fast",
                })
            parts = []
            for d in docs:
                content = d.page_content
                if content.startswith("passage: "):
                    content = content[9:]
                if "---" in content:
                    content = content.split("---", 1)[-1].strip()
                clean = " ".join(content.split())[:300]
                meta = d.metadata
                article  = meta.get("article_number", "")
                law_num  = meta.get("law_number", "")
                src_file = meta.get("source_file", "")
                page     = meta.get("page_number", 1)
                url = f"/pdfs/{src_file}#page={page}" if src_file else None
                label = f"{article}, {law_num}" if article else law_num
                link  = f"[{label}]({url})" if url else label
                parts.append(f"{clean}...\n\n→ {link}")
            result = {
                "answer": "\n\n---\n\n".join(parts),
                "docs": docs,
                "chunks_used": len(docs),
            }
        else:
            # pass history into RAG
            result = bot.ask(question, history=history)
            if "error" in result:
                return jsonify({"error": result["error"]}), 500

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

        elapsed = time.perf_counter() - start
        payload = {
            "answer": result["answer"],
            "sources": structured_sources,
            "chunks_used": result["chunks_used"],
            "cached": False,
            "response_s": round(elapsed, 2),
            "mode": mode,
        }

        # update per-user history
        if mode in ("llm", "casual"):
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": payload["answer"]})
            if len(history) > 2 * MAX_TURNS:
                history = history[-2 * MAX_TURNS:]
            user_histories[user_id] = history

        if mode != "casual":
            if len(answer_cache) >= CACHE_MAX:
                answer_cache.pop(next(iter(answer_cache)))
            answer_cache[cache_key] = payload
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/suggestions")
def suggestions():
    return jsonify(SUGGESTIONS)

from news_feed import fetch_news

@app.route("/news")
def news_page():
    return render_template("news.html")

@app.route("/api/news")
def api_news():
    articles = fetch_news()
    return jsonify({"articles": articles, "count": len(articles)})


if __name__ == "__main__":
    PORT = int(os.getenv("PORT", "5001"))

    print("🚀 Duke nisur LIGJBOT Web Interface...")
    print_access_urls(PORT)
    start_bot_background()

    app.run(debug=False, host="0.0.0.0", port=PORT, threaded=True)
else:
    # gunicorn / production (Render, Docker, etj.)
    if os.getenv("LIGJBOT_BOOTSTRAP", "0") == "1":
        start_bot_background()
