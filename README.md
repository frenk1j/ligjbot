# LIGJBOT

Asistent juridik për qytetarët shqiptarë, i ndërtuar me RAG (Retrieval-Augmented Generation).

## Çfarë bën

- Përgjigje juridike në shqip dhe anglisht
- Kërkim semantik mbi ligjet e indeksuara (FAISS)
- Burime të klikueshme (ligj / nen / faqe)
- UI web me histori bisede, dosje, dark/light mode
- Login me Google ose email (Firebase)
- PWA për iOS & Android

## Stack

- Python 3.11+ · Flask · LangChain · FAISS
- HuggingFace embeddings (`intfloat/multilingual-e5-large`)
- Groq LLM (`llama-3.1-8b-instant`)
- Firebase Auth + Firestore

## Struktura

```
ligjbot/
├── data/
│   ├── pdfs/                  # ligjet shqiptare (PDF)
│   └── eval_questions.json    # pyetje për evaluim RAG
├── scripts/
│   ├── ingestion.py           # PDF → FAISS (ekzekuto një herë)
│   ├── verify_store.py        # kontrollo vector store
│   ├── evaluate_rag.py        # evaluim cilësie me LLM
│   └── start_production.sh    # nis serverin me gunicorn
├── src/
│   ├── app.py                 # Flask web app
│   ├── rag_core.py            # RAG engine
│   ├── online_ingest.py       # ngarkim PDF në runtime
│   ├── news_feed.py           # lajme rrugore (RSS)
│   ├── static/
│   └── templates/
├── tests/
│   └── test_rag.py
├── Dockerfile
├── render.yaml
├── Procfile
├── requirements.txt
└── .env.example
```

## Setup lokal

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Plotëso `.env` me `GROQ_API_KEY` dhe konfigurimin Firebase.

## Ingestion (një herë)

```bash
python scripts/ingestion.py
python scripts/verify_store.py
```

## Ekzekutimi

```bash
python src/app.py
```

Hape: **http://localhost:5001**

## Teste

```bash
python tests/test_rag.py
python tests/test_rag.py --quick
python scripts/evaluate_rag.py
```

## Deploy (Render)

1. Push në GitHub
2. Krijo Web Service në [render.com](https://render.com)
3. Vendos env vars nga `.env`
4. Shto domain-in në Firebase → Authorized domains

---

> LIGJBOT është mjet informues dhe nuk zëvendëson këshillën e avokatit.
