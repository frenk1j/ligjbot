"""
====================================================
VERIFY SCRIPT v2 - Teston FAISS vector store
HuggingFace embeddings (falas, local)
====================================================
Perdorim: python scripts/verify_store.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_store/faiss_index")


def verify_vector_store():
    print("\n🔍 VERIFIKIMI I VECTOR STORE v2")
    print("=" * 50)

    if not Path(VECTOR_STORE_PATH).exists():
        print(f"❌ Vector store nuk ekziston: {VECTOR_STORE_PATH}")
        print("   Ekzekuto: python scripts/ingestion.py")
        return

    files = list(Path(VECTOR_STORE_PATH).parent.glob("*"))
    print(f"📁 Files ne vector_store/:")
    for f in files:
        if f.is_file():
            print(f"   • {f.name}: {f.stat().st_size/1024:.1f} KB")

    print(f"\n⏳ Duke ngarkuar FAISS index...")

    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        print(f"✅ Vector store u ngarkua!")

        # Test queries - shqip
        test_queries = [
            "query: ndalim i perkohshem policor te drejtat",
            "query: gjoba per perdorim celular gjate ngasjes",
            "query: kontrolli i identitetit kur ndalon policia",
            "query: te drejtat e qytetarit gjate arrestimit",
            "query: kufiri shtetëror kalimi i paligjshëm",
        ]

        print(f"\n🔎 TEST QUERIES:")
        print("=" * 50)

        for query in test_queries:
            display_q = query.replace("query: ", "")
            print(f"\n❓ '{display_q}'")

            results = vector_store.similarity_search(query, k=2)

            for i, doc in enumerate(results, 1):
                meta = doc.metadata
                print(f"  [{i}] {meta.get('law_number','?')} | "
                      f"{meta.get('article_number','?')} | "
                      f"Faqe {meta.get('page_number','?')}")
                # Pastro prefixin 'passage: ' per display
                content = doc.page_content.replace("passage: ", "", 1)
                print(f"       {content[:180]}...")

        print("\n✅ VERIFY OK - Vector store funksionon perfekt!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    verify_vector_store()
