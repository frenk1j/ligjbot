"""
====================================================
FAZA 2: RAG CORE — Python Q&A via Gemini
LIGJBOT - Albanian Legal RAG System
====================================================

Stack:
  - Vector Search : FAISS + HuggingFace multilingual-e5-large
  - LLM           : Google Gemini 2.0 Flash (1500 req/day FREE)
  - Framework     : LangChain

Si funksionon:
  1. Qytetari shkruan pyetjen
  2. FAISS kerkon chunks me te ngjashme (semantic search)
  3. Chunks i dergohen Gemini si kontekst
  4. Gemini gjeneron pergjigje te qarte me citim neni

Perdorim:
  python src/rag_core.py              # Chat interaktiv
  python src/rag_core.py --query "pyetja"  # Pyetje direkte
  python src/rag_core.py --demo       # Demo me pyetje te paracaktuara
====================================================
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

# ── Imports ────────────────────────────────────────────────────────────────
try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_groq import ChatGroq
    from langchain_core.documents import Document
    from langchain_core.messages import HumanMessage
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError as e:
    print(f"\n❌ Module mungon: {e}")
    print("   Ekzekuto: pip install -r requirements.txt")
    sys.exit(1)

# ── Config ─────────────────────────────────────────────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
LLM_MODEL         = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_store/faiss_index")
TOP_K_RESULTS     = int(os.getenv("TOP_K_RESULTS", 3))
FAST_TOP_K        = int(os.getenv("FAST_TOP_K", 2))
TEMPERATURE       = float(os.getenv("TEMPERATURE", 0.1))
MAX_TOKENS        = int(os.getenv("MAX_TOKENS", 512))
CONTEXT_CHARS_PER_CHUNK = int(os.getenv("CONTEXT_CHARS_PER_CHUNK", 800))

# ── System Prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Ti je LIGJBOT — asistenti juridik i qytetareve shqiptare.

MISIONI YT:
Ndihmon qytetaret shqiptare te kuptojne te drejtat dhe detyrimet e tyre ligjore,
vecanerisht ne situata me policine, gjobat rrugore, dhe kontrollat kufitare.

RREGULLAT E PERGJIGJES:
1. Pergjigju VETEM bazuar ne dokumentet ligjore te dhena si kontekst
2. Citohu GJITHMONE nenin dhe ligjin specifik (p.sh. "Neni 19, Ligji Nr. 82/2024")
3. Shpjego ne gjuhe te thjeshte, pa zhargon juridik
4. Nese pyetja nuk ka pergjigje ne kontekst, thuaj qarte: "Nuk gjeta informacion per kete ne ligjet e indexuara"
5. Per situata serioze (arrest, akuza penale), rekomandon konsultim me avokat
6. Pergjigjet te jene te shkurtra dhe te qarta — max 200 fjale
7. Pergjigju ne gjuhen qe te drejtohet perdoruesi (shqip ose anglisht)

FORMATI I PERGJIGJES:
- Filloje me pergjigjen direkte
- Pastaj citimet ligjore
- Mbylloje me keshille praktike nese eshte e nevojshme

KUJDES:
- Mos shpik informacion qe nuk eshte ne kontekst
- Mos jep keshilla per vepra penale
- Gjithmone kujtoje perdoruesin se LIGJBOT eshte informues, jo zevendesues i avokatit
"""


# ══════════════════════════════════════════════════════════════════════════
# RAG ENGINE
# ══════════════════════════════════════════════════════════════════════════

def _embedding_device() -> str:
    """Përdor MPS në Mac (Apple Silicon) për kërkim më të shpejtë."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class LIGJBOTRAG:
    """
    RAG Engine kryesor i LIGJBOT.
    Menaxhon FAISS search + Gemini generation.
    """

    def __init__(self):
        self.vector_store: Optional[FAISS] = None
        self.llm: Optional[ChatGroq] = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self._initialized = False

    def initialize(self, skip_llm: bool = False) -> bool:
        """Ngarkon vector store; opsionalisht inicializon Groq LLM."""

        print(f"\n{Fore.CYAN}{'='*55}")
        print(f"  LIGJBOT — Asistenti Juridik Shqiptar")
        mode = "FAST (vetëm kërkim)" if skip_llm else f"RAG me Groq {LLM_MODEL}"
        print(f"  {mode}")
        print(f"{'='*55}{Style.RESET_ALL}\n")

        # ── Validim API Key (vetëm kur nevojitet LLM) ─────────────────────
        if not skip_llm and (not GROQ_API_KEY or GROQ_API_KEY == "your-groq-api-key-here"):
            print(f"{Fore.RED}❌ GROQ_API_KEY nuk eshte konfiguruar!")
            print(f"   1. Shko: https://console.groq.com")
            print(f"   2. Klik 'API Keys' → 'Create API Key'")
            print(f"   3. Shto ne .env: GROQ_API_KEY=your-key{Style.RESET_ALL}")
            return False

        # ── Kontrollo vector store ────────────────────────────────────────
        if not Path(VECTOR_STORE_PATH).exists():
            print(f"{Fore.RED}❌ Vector store nuk ekziston: {VECTOR_STORE_PATH}")
            print(f"   Ekzekuto me pare: python src/ingestion.py{Style.RESET_ALL}")
            return False

        # ── Ngarko Embeddings ─────────────────────────────────────────────
        device = _embedding_device()
        print(f"{Fore.YELLOW}⏳ Duke ngarkuar embedding model ({device})...{Style.RESET_ALL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        print(f"{Fore.GREEN}✅ Embedding model i ngarkuar ({EMBEDDING_MODEL}, {device}){Style.RESET_ALL}")

        # ── Ngarko FAISS ──────────────────────────────────────────────────
        print(f"{Fore.YELLOW}⏳ Duke ngarkuar FAISS vector store...{Style.RESET_ALL}")
        self.vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"{Fore.GREEN}✅ FAISS index i ngarkuar{Style.RESET_ALL}")

        # ── Inicializo Groq (anashkalohet në FAST_MODE) ────────────────────
        if skip_llm:
            print(f"{Fore.CYAN}⚡ FAST_MODE: Groq u anashkalua (vetëm kërkim FAISS){Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⏳ Duke u lidhur me Groq {LLM_MODEL}...{Style.RESET_ALL}")
            self.llm = ChatGroq(
                model=LLM_MODEL,
                api_key=GROQ_API_KEY,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            print(f"{Fore.GREEN}✅ Groq {LLM_MODEL} i lidhur{Style.RESET_ALL}")

        self._initialized = True
        print(f"\n{Fore.GREEN}🎯 LIGJBOT eshte gati! Shkruaj pyetjen tend.{Style.RESET_ALL}")
        print(f"{Fore.CYAN}💡 Shkruaj 'exit' ose 'quit' per te dale.{Style.RESET_ALL}\n")
        return True

    def search_relevant_chunks(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Kerkon ne FAISS chunks me relevante per pyetjen.
        Perdor prefix 'query: ' per modelin e5-large.
        """
        if k is None:
            k = TOP_K_RESULTS
        search_query = f"query: {query}"
        return self.vector_store.similarity_search(search_query, k=k)

    def build_context(self, docs: List[Document]) -> str:
        """
        Nderton kontekstin nga chunks e gjetur.
        Formaon cdo chunk me metadata te qarte per Gemini.
        """
        context_parts = []

        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            law_num   = meta.get("law_number", "E panjohur")
            law_title = meta.get("law_title", "")
            article   = meta.get("article_number", "")
            art_title = meta.get("article_title", "")
            chapter   = meta.get("chapter", "")
            page      = meta.get("page_number", "?")

            # Pastro prefixin 'passage: ' nga content
            content = doc.page_content
            if content.startswith("passage: "):
                content = content[9:]
            # Hiq header metadata nga content per te evituar duplikime
            if "---" in content:
                content = content.split("---", 1)[-1].strip()
            # Limit context size per chunk for faster responses
            if len(content) > CONTEXT_CHARS_PER_CHUNK:
                content = content[:CONTEXT_CHARS_PER_CHUNK].rstrip() + "..."

            header = f"[BURIMI {i}] {law_num}"
            if article:
                header += f" | {article}"
                if art_title:
                    header += f" — {art_title}"
            header += f" | Faqe {page}"

            context_parts.append(f"{header}\n{content}")

        return "\n\n" + ("─" * 50) + "\n\n".join(context_parts)

    def build_sources_summary(self, docs: List[Document]) -> str:
        """Nderton nje liste te shkurter te burimeve per display."""
        sources = []
        seen = set()

        for doc in docs:
            meta = doc.metadata
            law_num = meta.get("law_number", "?")
            article = meta.get("article_number", "")
            page    = meta.get("page_number", "?")

            key = f"{law_num}_{article}"
            if key not in seen:
                seen.add(key)
                if article:
                    sources.append(f"  📌 {article}, {law_num} (Faqe {page})")
                else:
                    sources.append(f"  📌 {law_num} (Faqe {page})")

        return "\n".join(sources)

    def ask(self, question: str) -> Dict:
        """
        Pyetja kryesore RAG.

        Args:
            question: Pyetja e qytetarit

        Returns:
            Dict me: answer, sources, chunks_used
        """
        if not self._initialized:
            return {"error": "LIGJBOT nuk eshte inicializuar. Thirr initialize() me pare."}

        # ── Step 1: Semantic Search ───────────────────────────────────────
        relevant_docs = self.search_relevant_chunks(question)

        if not relevant_docs:
            return {
                "answer": "Nuk gjeta informacion relevant ne ligjet e indexuara per kete pyetje.",
                "sources": [],
                "chunks_used": 0
            }

        # ── Step 2: Nderto Kontekstin ─────────────────────────────────────
        context = self.build_context(relevant_docs)

        # ── Step 3: Nderto Promtin ────────────────────────────────────────
        full_prompt = f"""{SYSTEM_PROMPT}

===== DOKUMENTET LIGJORE (KONTEKSTI) =====
{context}
==========================================

PYETJA E QYTETARIT: {question}

PERGJIGJA JOTE (ne shqip, e qarte dhe me citim neni):"""

        # ── Step 4: LLM Generation ────────────────────────────────────────
        if not self.llm:
            return {
                "error": "LLM nuk eshte aktiv. Vendos FAST_MODE=0 ose rinis serverin pa skip_llm.",
            }

        response = self.llm.invoke([HumanMessage(content=full_prompt)])
        answer = response.content.strip()

        # ── Step 5: Sources ───────────────────────────────────────────────
        sources_text = self.build_sources_summary(relevant_docs)

        return {
            "answer": answer,
            "sources": sources_text,
            "chunks_used": len(relevant_docs),
            "docs": relevant_docs
        }

    def format_response(self, result: Dict, question: str) -> str:
        """Formon pergjigjen per display ne terminal."""

        if "error" in result:
            return f"{Fore.RED}❌ {result['error']}{Style.RESET_ALL}"

        output = []
        output.append(f"\n{Fore.CYAN}{'─'*55}{Style.RESET_ALL}")
        output.append(f"{Fore.WHITE}❓ Pyetja: {question}{Style.RESET_ALL}")
        output.append(f"{Fore.CYAN}{'─'*55}{Style.RESET_ALL}")
        output.append(f"\n{Fore.WHITE}{result['answer']}{Style.RESET_ALL}")
        output.append(f"\n{Fore.YELLOW}{'─'*55}")
        output.append(f"📚 Burimet ligjore te perdorura ({result['chunks_used']} chunks):")
        output.append(f"{result['sources']}{Style.RESET_ALL}")
        output.append(f"{Fore.CYAN}{'─'*55}{Style.RESET_ALL}\n")

        return "\n".join(output)


# ══════════════════════════════════════════════════════════════════════════
# DEMO QUERIES
# ══════════════════════════════════════════════════════════════════════════

DEMO_QUERIES = [
    "Cfare te drejta kam kur policia me ndalon ne ruge?",
    "Sa eshte gjoba per perdorimin e telefonit gjate ngasjes se mjetit?",
    "A lejohet policia te me kontrolloje pa arsye?",
    "Sa ore mund te me mbaje policia pa vendim gjykate?",
    "Cfare dokumentash duhet te kem gjithmone ne makine?",
    "Si mund te ankoj nje gjobe policore?",
    "Cfare ndodh nese refuzoj alkool-testin?",
    "Cilat jane te drejtat e mia kur kryqezohem ne kufi?",
]


# ══════════════════════════════════════════════════════════════════════════
# INTERACTIVE CHAT
# ══════════════════════════════════════════════════════════════════════════

def run_interactive_chat(bot: LIGJBOTRAG):
    """Chat interaktiv me LIGJBOT."""

    print(f"{Fore.CYAN}╔══════════════════════════════════════════════════════╗")
    print(f"║           LIGJETBOT — Chat Interaktiv                ║")
    print(f"║     Asistenti Juridik i Qytetarit Shqiptar           ║")
    print(f"╚══════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    print(f"\n{Fore.WHITE}Pyetje shembull:")
    for q in DEMO_QUERIES[:4]:
        print(f"  {Fore.YELLOW}→ {q}{Style.RESET_ALL}")
    print(f"\n{Fore.WHITE}Shkruaj {Fore.RED}'exit'{Fore.WHITE} per te dale.{Style.RESET_ALL}\n")

    while True:
        try:
            # Input prompt
            user_input = input(
                f"{Fore.GREEN}Ju: {Style.RESET_ALL}"
            ).strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "del", "dil"]:
                print(f"\n{Fore.CYAN}👋 Mirupafshim! Shpresoj te kem ndihmuar.{Style.RESET_ALL}\n")
                break

            # Help command
            if user_input.lower() in ["help", "ndihme", "?"]:
                print(f"\n{Fore.CYAN}Pyetje shembull:{Style.RESET_ALL}")
                for q in DEMO_QUERIES:
                    print(f"  {Fore.YELLOW}→ {q}{Style.RESET_ALL}")
                print()
                continue

            # Pyetja
            print(f"\n{Fore.YELLOW}⏳ Duke kerkuar ne ligje...{Style.RESET_ALL}", end="\r")
            result = bot.ask(user_input)
            formatted = bot.format_response(result, user_input)
            print(formatted)

        except KeyboardInterrupt:
            print(f"\n\n{Fore.CYAN}👋 Mirupafshim!{Style.RESET_ALL}\n")
            break
        except Exception as e:
            print(f"\n{Fore.RED}❌ Error: {e}{Style.RESET_ALL}\n")


def run_demo(bot: LIGJBOTRAG):
    """Demo me pyetje te paracaktuara."""

    print(f"\n{Fore.CYAN}{'='*55}")
    print(f"  DEMO MODE — {len(DEMO_QUERIES)} pyetje te paracaktuara")
    print(f"{'='*55}{Style.RESET_ALL}\n")

    for i, query in enumerate(DEMO_QUERIES, 1):
        print(f"{Fore.CYAN}[{i}/{len(DEMO_QUERIES)}]{Style.RESET_ALL} Duke procesuar...")
        result = bot.ask(query)
        formatted = bot.format_response(result, query)
        print(formatted)

        if i < len(DEMO_QUERIES):
            cont = input(f"{Fore.YELLOW}Vazhdo? (Enter=po, q=jo): {Style.RESET_ALL}").strip()
            if cont.lower() == 'q':
                break


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

def main():
    global TOP_K_RESULTS
    parser = argparse.ArgumentParser(
        description="LIGJBOT RAG Core — Faza 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Shembuj perdorimi:
  python src/rag_core.py                          # Chat interaktiv
  python src/rag_core.py --demo                   # Demo me pyetje
  python src/rag_core.py --query "pyetja ime"     # Pyetje direkte
        """
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Pyetje direkte (per scripting/testing)"
    )
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Ekzekuto demo me pyetje te paracaktuara"
    )
    parser.add_argument(
        "--topk", "-k",
        type=int,
        default=TOP_K_RESULTS,
        help=f"Sa chunks te ktheje FAISS (default: {TOP_K_RESULTS})"
    )

    args = parser.parse_args()

    # Override top_k nese specifikohet
    if args.topk != TOP_K_RESULTS:
        TOP_K_RESULTS = args.topk

    # Inicializo bot
    bot = LIGJBOTRAG()
    if not bot.initialize():
        sys.exit(1)

    # Zgjedh modalitetin
    if args.query:
        # Pyetje direkte
        result = bot.ask(args.query)
        print(bot.format_response(result, args.query))
    elif args.demo:
        # Demo mode
        run_demo(bot)
    else:
        # Chat interaktiv (default)
        run_interactive_chat(bot)


if __name__ == "__main__":
    main()
