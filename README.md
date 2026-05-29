# LIGJBOT

Asistent juridik për qytetarët shqiptarë, i ndërtuar me RAG (Retrieval-Augmented Generation).  
Sistemi kërkon në ligjet e indeksuara dhe kthen përgjigje të qarta me burime ligjore.

## Çfarë bën

- Përgjigjet pyetjeve juridike në shqip dhe anglisht
- Kërkim semantik mbi FAISS
- Kthen burime të klikueshme (ligj / nen / faqe)
- UI web me histori bisede, dark/light mode, PWA (iOS & Android)

## Stack

- Python 3.11+
- LangChain + FAISS
- HuggingFace embeddings (`intfloat/multilingual-e5-large`)
- Groq LLM (`llama-3.1-8b-instant`)
- Flask

## Struktura

```
ligjet-chatbot/
├── data/
│   └── pdfs/               ← ligjet shqiptare (PDF)
├── scripts/
│   ├── ingestion.py        ← PDF → FAISS vector store (ekzekuto një herë)
│   └── verify_store.py     ← kontrollo vector store
├── src/
│   ├── app.py              ← Flask web app
│   ├── rag_core.py         ← RAG engine (kërkim + LLM)
│   ├── static/
│   │   ├── ligjbot-logo.png
│   │   ├── ligjbot-logo-light.png
│   │   └── manifest.webmanifest
│   └── templates/
│       ├── index.html      ← UI kryesore
│       └── telefon.html    ← faqe QR për telefon
├── tests/
│   └── test_rag.py         ← teste automatike RAG
├── .env.example            ← kopjo si .env dhe plotëso
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Vendos `GROQ_API_KEY` te `.env` (falas: [console.groq.com](https://console.groq.com)).

## Ingestion (një herë, para startit)

```bash
# test i shpejtë me një PDF
python scripts/ingestion.py --test --pdf data/pdfs/kodi_rrugor.pdf

# ingestion i plotë (të gjitha PDF-të)
python scripts/ingestion.py

# verifikim
python scripts/verify_store.py
```

## Ekzekutimi

```bash
# Web app
python src/app.py
```

Hape në browser: **http://localhost:5001**

### Telefon (iOS / Android)

1. Mac dhe telefoni në **të njëjtin WiFi**
2. Shiko linkun `📱 iPhone/Android:` në terminal pas `python src/app.py`
3. Ose hap **http://localhost:5001/telefon** dhe skano QR-in

## Teste

```bash
python tests/test_rag.py          # të gjitha testet
python tests/test_rag.py --quick  # 3 teste të shpejta
```

## Konfigurime (`.env`)

| Variabla | Përshkrimi | Default |
|---|---|---|
| `GROQ_API_KEY` | API key nga console.groq.com | — |
| `LLM_MODEL` | Modeli Groq | `llama-3.1-8b-instant` |
| `FAST_MODE` | Vetëm kërkim FAISS, pa LLM (1=po) | `1` |
| `TOP_K_RESULTS` | Chunk-e për kërkim | `3` |
| `MAX_TOKENS` | Gjatësia max e përgjigjes | `512` |

## Troubleshooting

**`Vector store nuk ekziston`** → ekzekuto `python scripts/ingestion.py`

**`GROQ_API_KEY nuk eshte konfiguruar`** → shto key në `.env`

**Port 5001 i zënë** → `lsof -i :5001` pastaj `kill <PID>`

---

> LIGJBOT është mjet informues dhe nuk zëvendëson këshillën e avokatit.
