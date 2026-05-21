"""
====================================================
FAZA 1: INGESTION PIPELINE  v2
Ligjet Chatbot - Albanian Legal RAG System

NDRYSHIMI nga v1:
  ❌ OpenAI embeddings  ($0.02/1M tokens)
  ✅ HuggingFace embeddings (100% FALAS, local)
     Model: intfloat/multilingual-e5-large
     → Mbeshtet shqipen, anglishten, + 100 gjuhe
     → Shkarkohet vetem here te pare (~1.2 GB)
     → Pas kesaj, ekzekuton 100% offline

  ❌ OpenAI GPT  (me pare)
  ✅ Groq + Llama 3.3 70B  (FALAS)
     → 30,000 tokens/minute falas
     → Modeli me i mire open-source aktualisht

====================================================

Perdorim:
    python src/ingestion.py              # Te gjitha PDF ne data/pdfs/
    python src/ingestion.py --pdf x.pdf  # PDF specifike
    python src/ingestion.py --test --pdf x.pdf  # Pa embedding (debug)
"""

import os
import re
import json
import time
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

from dotenv import load_dotenv
from tqdm import tqdm

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from pypdf import PdfReader

# ============================================================
load_dotenv()
# ============================================================

GROQ_API_KEY      = os.getenv("GROQ_API_KEY")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_store/faiss_index")
PDF_FOLDER        = os.getenv("PDF_FOLDER", "./data/pdfs")
CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP", 200))


# ============================================================
# DATA CLASS
# ============================================================

@dataclass
class LegalChunk:
    """Një chunk i ligjit me metadata të plota."""
    content: str
    source_file: str
    law_number: str
    law_title: str
    chapter: str
    article_number: str
    article_title: str
    page_number: int
    chunk_index: int
    chunk_id: str


# ============================================================
# STEP 1: PDF EXTRACTION
# ============================================================

def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """Lexon PDF dhe kthen lista e faqeve me tekst."""
    pages = []
    reader = PdfReader(pdf_path)

    print(f"  📄 Faqe totale: {len(reader.pages)}")

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            text = clean_text(text)
            if len(text) > 50:
                pages.append({"page_number": page_num, "text": text})

    return pages


def clean_text(text: str) -> str:
    """Pastron tekstin e ekstraktuar nga PDF."""
    text = re.sub(r'-\n', '', text)           # Fix hyphenation
    text = re.sub(r'[ \t]+', ' ', text)       # Normalize spaces
    text = re.sub(r'\n{3,}', '\n\n', text)    # Max 2 newlines
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # Page numbers
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    return text.strip()


# ============================================================
# STEP 2: METADATA EXTRACTION
# ============================================================

def extract_law_metadata(pdf_path: str, full_text: str) -> Dict:
    """Ekstrakton metadata nga emri i file dhe permbajtja."""
    filename = Path(pdf_path).stem

    law_number = filename
    for pattern in [r'Ligj\s+Nr\.\s*([\d/]+)', r'LIGJ\s+Nr\.\s*([\d/]+)', r'Nr\.\s*([\d/]+)']:
        match = re.search(pattern, full_text[:2000], re.IGNORECASE)
        if match:
            law_number = f"Nr. {match.group(1)}"
            break

    law_title = "E panjohur"
    for pattern in [r'PËR\s+([A-ZËÇÀÈÙÀ\s]+?)(?:\n|Në mbështetje)',
                    r'PER\s+([A-ZËÇÀÈÙÀ\s]+?)(?:\n|Në mbështetje)']:
        match = re.search(pattern, full_text[:3000], re.IGNORECASE | re.DOTALL)
        if match:
            candidate = re.sub(r'\s+', ' ', match.group(1).strip())
            if 5 < len(candidate) < 200:
                law_title = candidate
                break

    if law_title == "E panjohur":
        law_title = filename.replace('_', ' ').replace('-', ' ')

    return {
        "law_number": law_number,
        "law_title": law_title,
        "source_file": Path(pdf_path).name
    }


def extract_chapter(text_before: str) -> str:
    """Gjen KREUN e fundit para chunk-it aktual."""
    matches = list(re.finditer(
        r'(KREU\s+[IVXLCDM]+[\s\n]+[A-ZËÇÀÈÙÀ\s]+?)(?=\n)',
        text_before, re.IGNORECASE
    ))
    if matches:
        return re.sub(r'\s+', ' ', matches[-1].group(1).strip())
    return "I panjohur"


def extract_article_info(chunk_text: str) -> Tuple[str, str]:
    """Gjen numrin dhe titullin e nenit ne chunk."""
    match = re.search(r'Neni\s+(\d+[a-z/]*)\s*\n([^\n]{3,80})', chunk_text, re.IGNORECASE)
    if match:
        return f"Neni {match.group(1)}", re.sub(r'\s+', ' ', match.group(2).strip())

    match = re.search(r'Neni\s+(\d+[a-z/]*)', chunk_text, re.IGNORECASE)
    if match:
        return f"Neni {match.group(1)}", ""

    return "", ""


# ============================================================
# STEP 3: SMART CHUNKING
# ============================================================

def create_smart_chunks(pages: List[Dict], law_metadata: Dict) -> List[LegalChunk]:
    """Krijon chunk-e inteligjente duke respektuar strukturen e ligjit."""

    full_text_parts = []
    page_markers = {}
    current_pos = 0

    for page in pages:
        page_markers[current_pos] = page["page_number"]
        full_text_parts.append(page["text"])
        current_pos += len(page["text"]) + 2

    full_text = "\n\n".join(full_text_parts)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\nNeni ", "\nKREU ", "\n\n", "\n", ". ", " "],
        keep_separator=True,
        length_function=len,
    )

    raw_chunks = splitter.split_text(full_text)
    print(f"  🔪 Chunks te krijuara: {len(raw_chunks)}")

    legal_chunks = []
    current_chapter = "I panjohur"
    text_so_far = ""

    for idx, chunk_text in enumerate(raw_chunks):
        text_so_far += chunk_text
        chapter_candidate = extract_chapter(text_so_far)
        if chapter_candidate != "I panjohur":
            current_chapter = chapter_candidate

        article_number, article_title = extract_article_info(chunk_text)

        approx_position = (idx / len(raw_chunks)) * len(full_text)
        page_num = 1
        for pos, pnum in sorted(page_markers.items()):
            if pos <= approx_position:
                page_num = pnum

        chunk_id = hashlib.md5(
            f"{law_metadata['source_file']}_{idx}_{chunk_text[:50]}".encode()
        ).hexdigest()[:12]

        enriched = build_enriched_content(
            chunk_text, law_metadata, current_chapter,
            article_number, article_title
        )

        legal_chunks.append(LegalChunk(
            content=enriched,
            source_file=law_metadata["source_file"],
            law_number=law_metadata["law_number"],
            law_title=law_metadata["law_title"],
            chapter=current_chapter,
            article_number=article_number,
            article_title=article_title,
            page_number=page_num,
            chunk_index=idx,
            chunk_id=chunk_id
        ))

    return legal_chunks


def build_enriched_content(chunk_text: str, law_metadata: Dict,
                            chapter: str, article_num: str,
                            article_title: str) -> str:
    """
    Nderton content te pasuruar me kontekst.
    SHENIM: Per multilingual-e5-large, shtohet prefix 'passage: '
    per embedding me te sakte (kerkesa e modelit).
    """
    context = f"""passage: Ligji: {law_metadata['law_title']} ({law_metadata['law_number']})
Kreu: {chapter}
{f'Neni: {article_num} - {article_title}' if article_num else ''}
---
{chunk_text}"""
    return context.strip()


# ============================================================
# STEP 4: DOCUMENTS
# ============================================================

def chunks_to_documents(legal_chunks: List[LegalChunk]) -> List[Document]:
    """Konverton LegalChunk ne LangChain Document me metadata."""
    return [
        Document(
            page_content=chunk.content,
            metadata={
                "source_file":    chunk.source_file,
                "law_number":     chunk.law_number,
                "law_title":      chunk.law_title,
                "chapter":        chunk.chapter,
                "article_number": chunk.article_number,
                "article_title":  chunk.article_title,
                "page_number":    chunk.page_number,
                "chunk_index":    chunk.chunk_index,
                "chunk_id":       chunk.chunk_id,
            }
        )
        for chunk in legal_chunks
    ]


# ============================================================
# STEP 5: HUGGINGFACE EMBEDDINGS + FAISS
# ============================================================

def load_embedding_model() -> HuggingFaceEmbeddings:
    """
    Ngarkon modelin e embedding nga HuggingFace.

    intfloat/multilingual-e5-large:
    - ✅ 100% FALAS - asnje API key
    - ✅ Mbeshtet shqipen + 100 gjuhe te tjera
    - ✅ Shkarkohet vetem here te pare (~1.2 GB)
    - ✅ Pas kesaj, 100% offline
    - ✅ 1024 dimensione - shume i sakte
    - ✅ Ranked #1 per multilingual tasks (MTEB benchmark)
    """
    print(f"\n🤖 Duke ngarkuar embedding model...")
    print(f"   Model: {EMBEDDING_MODEL}")
    print(f"   (Here e pare: shkarkon ~1.2GB. Pas kesaj: offline)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   # Ndrysho ne "cuda" nese ke GPU
        encode_kwargs={
            "normalize_embeddings": True,  # E nevojshme per e5 models
            "batch_size": 32,
        },
        show_progress=True,
    )

    print(f"   ✅ Model i ngarkuar!")
    return embeddings


def build_vector_store(documents: List[Document]) -> FAISS:
    """Gjeneron embeddings dhe krijon FAISS index."""

    embeddings = load_embedding_model()

    print(f"\n📊 Duke gjeneruar embeddings per {len(documents)} chunks...")
    print(f"   Kjo mund te zgjase 2-5 minuta (varet nga CPU)...")

    # HuggingFace nuk ka rate limits - no batching needed
    vector_store = FAISS.from_documents(documents, embeddings)

    return vector_store


def save_vector_store(vector_store: FAISS, save_path: str):
    """Ruan FAISS index ne disk."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(save_path)
    print(f"\n✅ FAISS index ruajtur: {save_path}")


def save_ingestion_report(all_chunks: List[LegalChunk],
                           output_path: str = "./vector_store/ingestion_report.json"):
    """Ruan raport te ingestion per debugging."""
    report = {
        "total_chunks": len(all_chunks),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_provider": "HuggingFace (FREE, local)",
        "llm_model": os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        "llm_provider": "Groq (FREE)",
        "files_processed": list(set(c.source_file for c in all_chunks)),
        "laws_indexed": list(set(
            f"{c.law_number}: {c.law_title}" for c in all_chunks
        )),
        "chunks_per_file": {},
        "articles_found": list(set(
            c.article_number for c in all_chunks if c.article_number
        )),
    }

    for chunk in all_chunks:
        fname = chunk.source_file
        report["chunks_per_file"][fname] = report["chunks_per_file"].get(fname, 0) + 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"📊 Raport: {output_path}")
    return report


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_single_pdf(pdf_path: str) -> List[LegalChunk]:
    """Proceson nje PDF te vetme."""
    print(f"\n📖 Processing: {Path(pdf_path).name}")
    print("=" * 50)

    print("  Step 1/3: Ekstrakton tekst...")
    pages = extract_text_from_pdf(pdf_path)

    if not pages:
        print(f"  ⚠️  Asnje tekst nga {pdf_path}")
        return []

    full_text = "\n\n".join(p["text"] for p in pages)
    print(f"  ✓ Karaktere: {len(full_text):,}")

    print("  Step 2/3: Ekstrakton metadata...")
    law_metadata = extract_law_metadata(pdf_path, full_text)
    print(f"  ✓ Ligji: {law_metadata['law_number']}")
    print(f"  ✓ Titulli: {law_metadata['law_title'][:60]}...")

    print("  Step 3/3: Krijon chunks...")
    chunks = create_smart_chunks(pages, law_metadata)
    print(f"  ✓ Chunks: {len(chunks)}")

    return chunks


def run_ingestion_pipeline(pdf_folder: str = None, single_pdf: str = None):
    """Pipeline kryesor i ingestion."""

    print("\n" + "="*60)
    print("🚀 FAZA 1: INGESTION PIPELINE v2")
    print("   Stack: HuggingFace Embeddings + Groq Llama 3.3")
    print("   Kosto: 100% FALAS")
    print("="*60)

    # Gjej PDF files
    pdf_files = []
    if single_pdf:
        if Path(single_pdf).exists():
            pdf_files = [single_pdf]
        else:
            print(f"❌ File nuk ekziston: {single_pdf}")
            return
    else:
        folder = pdf_folder or PDF_FOLDER
        if not Path(folder).exists():
            print(f"❌ Folder nuk ekziston: {folder}")
            print(f"   Krijo: mkdir -p {folder}")
            return

        pdf_files = list(Path(folder).glob("*.pdf"))
        pdf_files += list(Path(folder).glob("*.PDF"))

        if not pdf_files:
            print(f"❌ Nuk u gjet asnje PDF ne: {folder}")
            return

    print(f"\n📁 PDF files: {len(pdf_files)}")
    for f in pdf_files:
        size_mb = Path(f).stat().st_size / 1024 / 1024
        print(f"   • {Path(f).name} ({size_mb:.1f} MB)")

    # Process çdo PDF
    all_chunks = []
    for pdf_path in pdf_files:
        try:
            chunks = process_single_pdf(str(pdf_path))
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"\n❌ Error: {pdf_path}: {e}")
            continue

    if not all_chunks:
        print("\n❌ Asnje chunk. Kontrollo PDF-te.")
        return

    print(f"\n📊 SUMMARY:")
    print(f"   • Total chunks: {len(all_chunks):,}")
    print(f"   • PDF files: {len(pdf_files)}")
    print(f"   • Ligje: {len(set(c.law_number for c in all_chunks))}")
    print(f"   • Nene: {len(set(c.article_number for c in all_chunks if c.article_number))}")
    print(f"\n💰 Kosto: $0.00 (HuggingFace eshte FALAS)")
    print(f"⏱️  Koha: ~2-5 minuta (CPU)")

    confirm = input(f"\n❓ Vazhdo me embedding? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Anuluar.")
        return

    documents = chunks_to_documents(all_chunks)

    vector_store = build_vector_store(documents)

    save_vector_store(vector_store, VECTOR_STORE_PATH)

    report = save_ingestion_report(all_chunks)

    print("\n" + "="*60)
    print("✅ INGESTION PERFUNDOI!")
    print("="*60)
    print(f"📁 Vector store: {VECTOR_STORE_PATH}")
    print(f"📊 Chunks: {len(all_chunks):,}")
    print(f"🏛️  Ligje:")
    for law in report["laws_indexed"]:
        print(f"   • {law}")
    print("\n🎯 Hapi tjeter: python src/verify_store.py")
    print("="*60)


# ============================================================
# TEST MODE
# ============================================================

def test_extraction_only(pdf_path: str):
    """Test vetem extraction dhe chunking, pa embedding."""
    print("\n🧪 TEST MODE (pa embeddings, pa kosto)")
    print("="*50)

    pages = extract_text_from_pdf(pdf_path)
    full_text = "\n\n".join(p["text"] for p in pages)
    law_metadata = extract_law_metadata(pdf_path, full_text)
    chunks = create_smart_chunks(pages, law_metadata)

    print(f"\n📋 REZULTATE:")
    print(f"   Ligji: {law_metadata['law_number']}")
    print(f"   Titulli: {law_metadata['law_title']}")
    print(f"   Faqe: {len(pages)}")
    print(f"   Chunks: {len(chunks)}")

    print(f"\n📝 CHUNK I PARE:")
    print("-"*40)
    print(chunks[0].content[:500] if chunks else "Asnje")

    print(f"\n📝 CHUNK I FUNDIT:")
    print("-"*40)
    print(chunks[-1].content[:500] if chunks else "Asnje")

    articles = [c.article_number for c in chunks if c.article_number]
    print(f"\n📌 Nene: {len(set(articles))}")
    print(f"   Shembuj: {list(set(articles))[:10]}")

    return chunks


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestion Pipeline - Ligjet Chatbot v2")
    parser.add_argument("--pdf",    type=str, help="Path i PDF specifike")
    parser.add_argument("--folder", type=str, help="Folder me PDFs")
    parser.add_argument("--test",   action="store_true", help="Test pa embeddings")

    args = parser.parse_args()

    if args.test:
        if not args.pdf:
            print("❌ Per --test duhet: --pdf path/to/file.pdf")
        else:
            test_extraction_only(args.pdf)
    else:
        run_ingestion_pipeline(pdf_folder=args.folder, single_pdf=args.pdf)
