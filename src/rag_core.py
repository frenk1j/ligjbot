"""
====================================================
FAZA 2: RAG CORE — Python Q&A via Groq
LIGJBOT - Albanian Legal RAG System
====================================================

Stack:
  - Vector Search : FAISS + HuggingFace multilingual-e5-large
  - LLM          : Groq llama-3.1-8b-instant
  - Framework    : LangChain

Si funksionon:
  1. Qytetari shkruan pyetjen
  2. FAISS kerkon chunks me te ngjashme (semantic search)
  3. (Tani) BM25 ben kërkim me fjalë kyçe
  4. Rezultatet kombinohen (hybrid) dhe i dërgohen LLM si kontekst
  5. LLM gjeneron pergjigje të qartë me citim neni

Perdorim:
  python src/rag_core.py                    # Chat interaktiv
  python src/rag_core.py --query "pyetja"   # Pyetje direkte
  python src/rag_core.py --demo             # Demo me pyetje te paracaktuara
====================================================
"""

import os
import re
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
    from langchain_community.retrievers import BM25Retriever  # NEW
    from langchain_core.documents import Document
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError as e:
    print(f"\n❌ Module mungon: {e}")
    print("   Ekzekuto: pip install -r requirements.txt")
    sys.exit(1)

# ── Config ─────────────────────────────────────────────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
LLM_MODEL         = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_store/faiss_index")
TOP_K_RESULTS     = int(os.getenv("TOP_K_RESULTS", 3))
FAST_TOP_K        = int(os.getenv("FAST_TOP_K", 2))
TEMPERATURE       = float(os.getenv("TEMPERATURE", 0.55))
MAX_TOKENS        = int(os.getenv("MAX_TOKENS", 650))
CONTEXT_CHARS_PER_CHUNK = int(os.getenv("CONTEXT_CHARS_PER_CHUNK", 800))
USE_HYBRID        = os.getenv("USE_HYBRID", "1") == "1"
LEGAL_SCORE_MAX   = float(os.getenv("LEGAL_SCORE_MAX", "0.82"))

# ── System Prompts ───────────────────────────────────────────────────────
CORE_PERSONA = """Ti je LIGJBOT — miku në chat që njeh mirë ligjet shqiptare.

Si flet (për ÇDO pyetje):
- Ngrohtë dhe natyral, si shok që e di temën — jo si robot apo zyrtar.
- Saktë: thuaj vetëm çfarë di me siguri; mos trill.
- Përgjigju pyetjes që të bëhet TANI — mos përsërit pyetjen, mos përsërit përgjigje të vjetra.
- "Unë" = ti (LIGJBOT). "Ti" = personi që pyet. Kurrë mos i këmbe rolet.
- Shqip i saktë. Fjali të shkurtra.

FORMATI I PËRGJIGJES (shumë i rëndësishëm):
- Kthe VETËM tekstin e përgjigjes — asgjë tjetër.
- MOS shkruaj: "Ti:", "LIGJBOT:", "Përdoruesi:", "Pyetja:", ose format dialogu.
- MOS përsërit pyetjen e përdoruesit.

Mos: "Sipas ligjit", "Ju lutem", "Jam i gatshëm", "do të jem i gatshëm", avokat."""

LEGAL_RULES = """Konteksti ligjor më poshtë është burimi yt për fakte.
- Fillo me përgjigjen e drejtpërdrejtë (po/jo/shifra), pastaj shpjego thjesht.
- Përdor vetëm informacion nga konteksti për nene dhe sanksione.
- Kur citon, link inline: [Neni X, Ligji Y](URL). Mos kopjo paragrafë të gjatë.
- 2–4 fjali. Gjuha: si pyetja."""

CASUAL_RULES = """Nuk ke dokumente ligjore për këtë mesazh.
- Bisedo normalisht, 1–2 fjali miqësore.
- Nëse pyetja s'është ligjore, përgjigju natyrshëm; nëse ka lidhje me ligjet, fto butësisht të pyesësh konkretisht.
- Mos cito nene pa nevojë. Gjuha: si pyetja."""

LEGAL_KEYWORDS = (
    "ligj", "nen", "neni", "polici", "policia", "gjob", "gjoba", "kod", "rrugor",
    "procedur", "drejtat", "detyrim", "arrest", "denim", "gjykat", "gjykate",
    "burg", "kundërvaj", "kundervaj", "patent", "patenta", "targa", "shofer",
    "makina", "kufi", "kontroll", "ndalim", "ndaloj", "ndalojn", "paraburgim",
    "transport", "leje", "dokument", "gjobat", "ndëshkim", "ndeshkim",
    "mbaj", "mbaje", "kohe", "koha", "ore", "orë", "ndodh", "nese", "nëse",
    "lejohet", "ndalohet", "radar", "alkool", "denim", "padit", "ankim",
    "kontrat", "pune", "punë", "taks", "aksident", "pasaport", "viz",
    "grind", "dhun", "arm", "drog", "mitur", "familj", "qira", "shtepi", "shtëpi",
)


def is_legal_question(question: str) -> bool:
    if get_meta_answer(question):
        return False
    q = question.lower().strip()
    return any(kw in q for kw in LEGAL_KEYWORDS)


def _norm_question(question: str) -> str:
    q = question.lower().strip()
    q = q.replace("ç", "c").replace("ë", "e")
    q = re.sub(r"[^\w\s]", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def get_meta_answer(question: str) -> Optional[str]:
    """Përgjigje të fiksuara për pyetje identiteti/salutimi — pa LLM."""
    q = _norm_question(question)

    if re.search(r"^(pershendetje|hello|hi|hej|miremengjes|miredita)\b", q):
        return "Hej! Si mund të të ndihmoj me ligjet sot?"

    if re.search(r"\b(si je|si jeni|si kalon|si po shkon|how are you)\b", q):
        return "Mirë faleminderit! Po jam këtu. Ke ndonjë pyetje për ligjet?"

    if re.search(r"\b(faleminderit|thanks|thank you)\b", q):
        return "S'ka përse! Nëse ke diçka tjetër, thuaj."

    if re.search(r"\b(kush je|cfare je|cili je|who are you)\b", q):
        return (
            "Unë jam LIGJBOT — të ndihmoj me ligjet shqiptare. "
            "Më pyet për gjoba, policë, leje, të drejtat... ta shpjegoj thjesht."
        )

    if re.search(r"(cfare|si).*(me )?ndihmon", q) or re.search(r"me ndihmon.*pyes", q):
        return (
            "Më shkruaj thjesht çfarë ke në mendje — p.sh. gjoba, policë, leje drejtimi — "
            "dhe unë ta shpjegoj me fjalë normale, me burime kur duhet."
        )

    if re.search(r"(cfare ben|cfare bo).*(per mua|pune)", q) or re.search(r"per mua.*(cfare ben|cfare bo)", q):
        return (
            "Unë të ndihmoj të kuptosh ligjet — i përkthej nga gjuhë ligjore "
            "në gjuhë normale, me burime kur duhet. Thuaj çfarë të intereson."
        )

    if re.search(r"\b(cfare ben|cfare bo|what do you do)\b", q):
        return (
            "Unë i shpjegoj ligjet shqiptare me fjalë të thjeshta — "
            "gjoba, policë, leje, të drejtat. Më pyet konkretisht çfarë të duhet."
        )

    if re.search(r"\b(cfare mund|si mund te me ndihmosh|si funksionon)\b", q):
        return (
            "Më shkruan pyetjen tënde dhe unë ta shpjegoj me fjalë normale, "
            "me burime nga ligjet kur ka nevojë."
        )

    return None


def _history_to_messages(history: Optional[list], max_pairs: int = 4) -> list:
    if not history:
        return []
    msgs = []
    for turn in history[-max_pairs * 2:]:
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if turn.get("role") == "user":
            msgs.append(HumanMessage(content=content))
        elif turn.get("role") == "assistant":
            msgs.append(AIMessage(content=content))
    return msgs


def _is_broken_answer(text: str) -> bool:
    if not text or len(text.strip()) < 4:
        return True
    if re.search(r"^(Ti|Ju|LIGJBOT|Përdoruesi|Pyetja)\s*:", text, re.I | re.M):
        return True
    n = _norm_question(text)
    broken_patterns = (
        r"ti je i rendesishem",
        r"te perdorur si",
        r"me ben te mund",
        r"ndihmes i mire qe",
        r"robot te vetem",
        r"i gatshem per te",
        r"do te jem i gatshem",
        r"pyetjet tuaja ligjore",
        r"asistenti juaj",
    )
    if any(re.search(p, n) for p in broken_patterns):
        return True
    if re.match(r"^ti je\b", n) and "?" not in text:
        return True
    return False


def _strip_dialogue_format(text: str) -> str:
    """Heq format dialogu që modeli ndonjëherë nxjerr gabimisht."""
    out = text.strip()
    m = re.search(r"LIGJBOT\s*:\s*(.+)", out, re.I | re.S)
    if m:
        return m.group(1).strip()
    lines = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^(Ti|Ju|Përdoruesi|User|Pyetja)\s*:", line, re.I):
            continue
        line = re.sub(r"^LIGJBOT\s*:\s*", "", line, flags=re.I)
        lines.append(line)
    return "\n".join(lines).strip() if lines else out


def _clean_answer(text: str) -> str:
    """Heq fraza robotike, format dialogu dhe paralajmërime."""
    out = _strip_dialogue_format(text.strip())
    patterns = [
        r"\s*Për situata serioze[^.]*avokat[^.]*\.?",
        r"\s*Konsultohu[^.]*avokat[^.]*\.?",
        r"\s*Kontaktoni avokatin[^.]*\.?",
        r"\s*Do të ishte mirë të flisje me avokat[^.]*\.?",
        r"\s*rekomandoj[^.]*avokat[^.]*\.?",
        r"^Jam i gatshëm[^.]*\.?\s*",
        r"^Unë jam i gatshëm[^.]*\.?\s*",
        r",?\s*unë do të jem i gatshëm[^.]*\.?",
        r"^Si asistent[^.]*\.?\s*",
        r"^Bazuar në[^.]*\.?\s*",
        r"^Sipas ligjit[^.]*\.?\s*",
        r"^Sipas nenit[^.]*\.?\s*",
    ]
    for pat in patterns:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    return out.strip()

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

def _dedupe_paragraphs(text: str) -> str:
    """Heq paragrafe të përsëritura thuajse identike."""
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    seen = set()
    clean_parts = []
    for p in parts:
        key = p.replace(" ", "").lower()
        if key in seen:
            continue
        seen.add(key)
        clean_parts.append(p)
    return "\n\n".join(clean_parts)

class LIGJBOTRAG:
    """
    RAG Engine kryesor i LIGJBOT.
    Menaxhon FAISS search + Groq generation.
    """

    def __init__(self):
        self.vector_store: Optional[FAISS] = None
        self.llm: Optional[ChatGroq] = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self._initialized = False

        # Semantic retriever (FAISS-based)
        self.retriever = None

        # NEW: BM25 lexical retriever
        self.bm25_retriever: Optional[BM25Retriever] = None

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

        # Krijo semantic retriever standard nga FAISS
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": FAST_TOP_K})

        # ── NEW: Nderto BM25 retriever nga të njëjtat dokumente ────────────
        try:
            all_docs = list(self.vector_store.docstore._dict.values())
            if all_docs:
                print(f"{Fore.YELLOW}⏳ Duke ndertuar BM25 lexical index ({len(all_docs)} chunks)...{Style.RESET_ALL}")
                self.bm25_retriever = BM25Retriever.from_documents(
                    all_docs,
                    k=FAST_TOP_K,
                )
                print(f"{Fore.GREEN}✅ BM25 retriever i gatshem{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}⚠️ Nuk u gjetën dokumente në docstore për BM25{Style.RESET_ALL}")
                self.bm25_retriever = None
        except Exception as e:
            print(f"{Fore.RED}⚠️ Nuk u ndertua BM25 retriever: {e}{Style.RESET_ALL}")
            self.bm25_retriever = None

        # ── Inicializo Groq (anashkalohet në FAST_MODE) ───────────────────
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

    def is_likely_legal(self, question: str) -> bool:
        """Pyetje ligjore: fjalë kyçe, ngjashmëri semantike, ose jo meta/bisedë."""
        if get_meta_answer(question):
            return False
        if is_legal_question(question):
            return True
        if not self.vector_store:
            return False
        try:
            results = self.vector_store.similarity_search_with_score(
                f"query: {question}", k=1
            )
            if results:
                _, score = results[0]
                return score < LEGAL_SCORE_MAX
        except Exception:
            pass
        return False

    def _generate(
        self,
        system: str,
        user: str,
        history: Optional[list] = None,
    ) -> str:
        if not self.llm:
            return ""
        messages: list = [SystemMessage(content=system)]
        messages.extend(_history_to_messages(history))
        messages.append(HumanMessage(content=user))

        answer = ""
        for attempt in range(2):
            response = self.llm.invoke(messages)
            answer = _clean_answer(_dedupe_paragraphs(response.content.strip()))
            if answer and not _is_broken_answer(answer):
                return answer
            if attempt == 0:
                messages.append(AIMessage(content=answer or "..."))
                messages.append(HumanMessage(content=(
                    "Gabim. Rishkruaje vetëm përgjigjen — pa 'Ti:', pa 'LIGJBOT:', "
                    "pa përsëritje pyetjeje. Shqip i saktë, shkurt, miqësor."
                )))
        return answer or "Më vjen keq, nuk e kapërcova mirë. Mund ta pyesësh ndryshe?"

    def search_relevant_chunks(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Kërkon në FAISS chunks më relevante për pyetjen (vetëm semantik).
        Përdor prefix 'query: ' për modelin e5-large.
        """
        if k is None:
            k = TOP_K_RESULTS
        search_query = f"query: {query}"
        return self.vector_store.similarity_search(search_query, k=k)

    def hybrid_search_relevant_chunks(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Hybrid search: kombinon FAISS (semantik) + BM25 (fjalë kyçe),
        pastaj bën një re-rankim të thjeshtë duke përdorur renditjen si score.
        """
        if not self.vector_store:
            return []

        top_k = k or TOP_K_RESULTS

        # 1) Semantic search via FAISS (përdor funksionin ekzistues që punon)
        semantic_docs: List[Document] = []
        try:
            semantic_docs = self.search_relevant_chunks(query, k=top_k)
        except Exception as e:
            print(f"[WARN] Semantic search deshtoi: {e}")

        bm25_docs: List[Document] = []
        if self.bm25_retriever is not None:
            try:
                # LangChain retrievers use .invoke(query) in newer versions
                bm25_docs = self.bm25_retriever.invoke(query)[:top_k]
            except Exception as e:
                print(f"[WARN] BM25 search deshtoi: {e}")

        # 3) Fuzionim i thjeshtë i rezultateve
        combined: dict[str, dict] = {}

        def add_docs(docs: List[Document], weight: float, source: str):
            for rank, d in enumerate(docs):
                meta = d.metadata or {}
                doc_id = (
                    meta.get("id")
                    or meta.get("source")
                    or f"{id(d)}"
                )
                score = (top_k - rank) / top_k  # 1.0 për vendin e parë, pastaj më pak
                if doc_id not in combined:
                    combined[doc_id] = {"doc": d, "score": 0.0, "sources": set()}
                combined[doc_id]["score"] += weight * score
                combined[doc_id]["sources"].add(source)

        # Pesha: semantik 0.6, BM25 0.4 (mund t'i ndryshosh me vone)
        add_docs(semantic_docs, weight=0.6, source="semantic")
        add_docs(bm25_docs,    weight=0.4, source="bm25")

        if not combined:
            return []

        ranked = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in ranked[:top_k]]

    def add_documents(self, docs: List[Document]):
        """
        Shton dokumente të reja në FAISS dhe rifreskon BM25 retrieverin.
        Përdoret kur ngarkohen ligje të reja (p.sh. PDF e Kodi Rrugor).
        """
        if not self.vector_store or not docs:
            return

        # 1) shto në FAISS
        self.vector_store.add_documents(docs)

        # 2) rifresko docstore list
        all_docs = list(self.vector_store.docstore._dict.values())

        # 3) rifresko BM25
        try:
            self.bm25_retriever = BM25Retriever.from_documents(
                all_docs,
                k=FAST_TOP_K,
            )
            print("✅ BM25 retriever u rifreskua pas ngarkimit të PDF.")
        except Exception as e:
            print(f"[WARN] Rifreskimi i BM25 deshtoi: {e}")

    def build_context(self, docs: List[Document]) -> str:
        """
        Ndërton kontekstin nga chunks e gjetur.
        Format çdo chunk me metadata të qartë për LLM, duke përfshirë URL-në.
        """
        context_parts = []

        for i, doc in enumerate(docs, 1):
            meta      = doc.metadata
            law_num   = meta.get("law_number", "E panjohur")
            article   = meta.get("article_number", "")
            art_title = meta.get("article_title", "")
            page      = meta.get("page_number", "?")
            src_file  = meta.get("source_file", "")

            url = f"/pdfs/{src_file}#page={page}" if src_file else None

            # Pastro content
            content = doc.page_content
            if content.startswith("passage: "):
                content = content[9:]
            if "---" in content:
                content = content.split("---", 1)[-1].strip()
            if len(content) > CONTEXT_CHARS_PER_CHUNK:
                content = content[:CONTEXT_CHARS_PER_CHUNK].rstrip() + "..."

            header = f"[BURIMI {i}] {law_num}"
            if article:
                header += f" | {article}"
                if art_title:
                    header += f" — {art_title}"
            header += f" | Faqe {page}"
            if url:
                header += f" | URL: {url}"

            context_parts.append(f"{header}\n{content}")

        return "\n\n" + ("─" * 50) + "\n\n".join(context_parts)

    def build_sources_summary(self, docs: List[Document]) -> str:
        """Ndërton një listë të shkurtër të burimeve për display."""
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

    def ask(self, question: str, history: Optional[list[dict]] = None) -> Dict:
        """
        Pyetja kryesore RAG.

        Args:
            question: Pyetja e qytetarit
            history:  Lista me turns te meparshme (role, content)

        Returns:
            Dict me: answer, sources, chunks_used
        """
        if not self._initialized:
            return {"error": "LIGJBOT nuk eshte inicializuar. Thirr initialize() me pare."}

        # ── Step 1: Retrieval (FAISS vs Hybrid) ─────────────────────────
        if USE_HYBRID:
            relevant_docs = self.hybrid_search_relevant_chunks(question)
        else:
            relevant_docs = self.search_relevant_chunks(question)

        if not relevant_docs:
            return {
                "answer": "Hmm, s'e gjeta këtë te ligjet që kam në dorë. Mund ta riformulosh pak, ose më thuaj më konkretisht çfarë të ndodhi?",
                "sources": [],
                "chunks_used": 0,
            }

        # ── Step 2: Ndërto Kontekstin ───────────────────────────────────
        context = self.build_context(relevant_docs)

        # ── Step 3: LLM Generation ──────────────────────────────────────
        if not self.llm:
            return {
                "error": "LLM nuk eshte aktiv. Vendos FAST_MODE=0 ose rinis serverin pa skip_llm.",
            }

        system = f"{CORE_PERSONA}\n\n{LEGAL_RULES}"
        user = (
            f"KONTEKSTI LIGJOR:\n{context}\n\n"
            f"{question}\n\n"
            "Përgjigja (vetëm teksti, pa etiketa):"
        )
        answer = self._generate(system, user, history=history)

        # ── Step 5: Sources ─────────────────────────────────────────────
        sources_text = self.build_sources_summary(relevant_docs)

        return {
            "answer": answer,
            "sources": sources_text,
            "chunks_used": len(relevant_docs),
            "docs": relevant_docs,
        }

    def ask_casual(self, question: str, history: Optional[list[dict]] = None) -> Dict:
        """Përgjigje miqësore pa RAG — për bisedë të përditshme."""
        if not self._initialized:
            return {"error": "LIGJBOT nuk eshte inicializuar."}

        meta = get_meta_answer(question)
        if meta:
            return {
                "answer": meta,
                "sources": "",
                "chunks_used": 0,
                "docs": [],
            }

        if not self.llm:
            return {
                "answer": "Hej! Jam LIGJBOT — më pyet çfarëdo për ligjet, ta shpjegoj thjesht.",
                "sources": [],
                "chunks_used": 0,
                "docs": [],
            }

        system = f"{CORE_PERSONA}\n\n{CASUAL_RULES}"
        user = f"{question}\n\nPërgjigja (vetëm teksti, pa etiketa):"
        answer = self._generate(system, user, history=history)
        return {
            "answer": answer,
            "sources": "",
            "chunks_used": 0,
            "docs": [],
        }

    def format_response(self, result: Dict, question: str) -> str:
        """Formon përgjigjen për display në terminal."""

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
    print(f"║           LIGJETBOT — Chat Interaktiv               ║")
    print(f"║     Asistenti Juridik i Qytetarit Shqiptar          ║")
    print(f"╚══════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    print(f"\n{Fore.WHITE}Pyetje shembull:")
    for q in DEMO_QUERIES[:4]:
        print(f"  {Fore.YELLOW}→ {q}{Style.RESET_ALL}")
    print(f"\n{Fore.WHITE}Shkruaj {Fore.RED}'exit'{Fore.WHITE} per te dale.{Style.RESET_ALL}\n")

    while True:
        try:
            user_input = input(
                f"{Fore.GREEN}Ju: {Style.RESET_ALL}"
            ).strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "del", "dil"]:
                print(f"\n{Fore.CYAN}👋 Mirupafshim! Shpresoj te kem ndihmuar.{Style.RESET_ALL}\n")
                break

            if user_input.lower() in ["help", "ndihme", "?"]:
                print(f"\n{Fore.CYAN}Pyetje shembull:{Style.RESET_ALL}")
                for q in DEMO_QUERIES:
                    print(f"  {Fore.YELLOW}→ {q}{Style.RESET_ALL}")
                print()
                continue

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
  python src/rag_core.py                             # Chat interaktiv
  python src/rag_core.py --demo                      # Demo me pyetje
  python src/rag_core.py --query "pyetja ime"        # Pyetje direkte
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
        help=f"Sa chunks te ktheje (default: {TOP_K_RESULTS})"
    )

    args = parser.parse_args()

    if args.topk != TOP_K_RESULTS:
        TOP_K_RESULTS = args.topk

    bot = LIGJBOTRAG()
    if not bot.initialize():
        sys.exit(1)

    if args.query:
        result = bot.ask(args.query)
        print(bot.format_response(result, args.query))
    elif args.demo:
        run_demo(bot)
    else:
        run_interactive_chat(bot)

if __name__ == "__main__":
    main()
