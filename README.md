# LIGJBOT

Asistent juridik për qytetarët shqiptarë, i ndërtuar me RAG (Retrieval-Augmented Generation).  
Sistemi kërkon në ligjet e indeksuara dhe kthen përgjigje të qarta me burime ligjore.

## Çfarë bën

- Përgjigjet pyetjeve juridike në shqip dhe anglisht.
- Përdor kërkim semantik mbi FAISS.
- Kthen burime të klikueshme (ligj/nen/faqe).
- Ofron UI web me histori bisede, dark/light mode dhe mobile support.

## Stack

- Python 3.11+
- LangChain + FAISS
- HuggingFace embeddings (`intfloat/multilingual-e5-large`)
- Groq LLM
- Flask (web app)

## Struktura

```text
ligjet-chatbot/
├── src/
│   ├── ingestion.py
│   ├── verify_store.py
│   ├── rag_core.py
│   ├── app.py
│   ├── templates/index.html
│   └── static/ligjbot-logo.png
├── data/pdfs/
├── vector_store/            # krijohet pas ingestion
├── requirements.txt
├── .env.example
└── README.md
```

## Setup i shpejtë

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Vendos `GROQ_API_KEY` te `.env`.

## Ingestion (një herë)

```bash
# test i shpejtë
python src/ingestion.py --test --pdf data/pdfs/kodi_rrugorlatestupdate.pdf

# full ingestion
python src/ingestion.py

# verifikim
python src/verify_store.py
```

## Ekzekutimi

### CLI

```bash
python src/rag_core.py
```

### Web app

```bash
python src/app.py
```

Hape në browser:

- `http://localhost:5001`

## Konfigurime kryesore (`.env`)

- `GROQ_API_KEY=...`
- `LLM_MODEL=llama-3.1-8b-instant`
- `TOP_K_RESULTS=3`
- `MAX_TOKENS=512`
- `CONTEXT_CHARS_PER_CHUNK=800`
- `FAST_MODE=1`

`FAST_MODE=1` përdor mënyrë më të shpejtë (retrieval-first).

## Troubleshooting

- `Vector store nuk ekziston`  
  Ekzekuto ingestion: `python src/ingestion.py`

- `GROQ_API_KEY nuk eshte konfiguruar`  
  Shto API key në `.env`.

- Port `5001` i zënë  
  Mbyll procesin ekzistues ose përdor port tjetër te `src/app.py`.

## Siguri dhe Git

Mos bëj commit këto:

- `.env`
- `.venv/`
- `vector_store/`
- `__pycache__/`

## Disclaimer

LIGJBOT është mjet informues dhe nuk zëvendëson këshillën e një avokati të licencuar.

