# 🏛️ LigjetBot — Asistenti Juridik i Qytetarit Shqiptar

> **"Sepse çdo qytetar meriton të dijë të drejtat e tij."**

LigjetBot është një sistem **RAG (Retrieval-Augmented Generation)** që u përgjigjet pyetjeve juridike të qytetarëve shqiptarë — në gjuhë të thjeshtë, me citim të nenit exact, 24/7, me kosto zero.

---

## 📋 Përmbajtja

- [Vizioni i Projektit](#-vizioni-i-projektit)
- [Stack Teknologjik](#️-stack-teknologjik)
- [Struktura e Projektit](#-struktura-e-projektit)
- [Faza 1 — Ingestion Pipeline](#-faza-1--ingestion-pipeline-bërë-)
- [Faza 2 — RAG Core me Groq](#-faza-2--rag-core-me-groq-bërë-)
- [Setup i Shpejtë](#-setup-i-shpejtë)
- [Ligjet e Indexuara](#-ligjet-e-indexuara)
- [Shembuj Bisedash](#-shembuj-bisedash-reale)
- [Troubleshooting](#-troubleshooting)
- [Udhëzues për Ekipin](#-udhëzues-për-ekipin)
- [Roadmap](#-roadmap)

---

## 🎯 Vizioni i Projektit

### Problemi Real

Çdo ditë qytetarët shqiptarë përballen me situata ligjore pa e ditur të drejtat e tyre:

| Situata reale | Qytetari nuk di | Neni |
|---------------|-----------------|------|
| Policia e ndalon | Sa mund ta mbajë — max 5 orë | Neni 19, L.82/2024 |
| Merr gjobë për celular | Shumën e saktë dhe si të ankohet | Neni 172, Kodi Rrugor |
| Kontrollohet pa arsye | Që duhet procesverbal gjithmonë | Neni 22, L.82/2024 |
| Kalon kufirin | Dokumentat dhe procedurat | Neni 13, L.39/2025 |

**Rezultati:** Qytetarët paguajnë gjoba që nuk i meritojnë, nuk ankohen kur duhet, dhe ndjehen të pafuqishëm përpara sistemit.

### Zgjidhja

```
Qytetari pyet                LigjetBot përgjigjet
─────────────────────        ─────────────────────────────────────────
"Sa mund të më mbajë    →    "Sipas Nenit 19, Ligji Nr. 82/2024:
 policia?"                    • Maksimumi: 5 orë pa vendim gjykate
                              • Keni të drejtë të kontaktoni avokat
                              • Duhet të informoheni për arsyen
                              📌 Neni 19, Ligji 82/2024 — Faqe 8"
```

---

## 🛠️ Stack Teknologjik

| Komponenti | Teknologjia | Pse | Kosto |
|------------|-------------|-----|-------|
| **PDF Parsing** | `pypdf` | Lexon PDF shqip me encoding korrekt | $0 |
| **Embeddings** | `HuggingFace multilingual-e5-large` | #1 multilingual, mbështet shqipen, 100% offline | $0 |
| **Vector DB** | `FAISS` (local) | Ultra i shpejtë, ruhet në disk, pa cloud | $0 |
| **LLM** | `Groq — Llama 3.3 70B Versatile` | Ultra i shpejtë (LPU hardware), tier falas bujar | $0 |
| **Framework** | `LangChain 1.x` | Orchestration RAG, integrim i lehtë | $0 |
| **TOTAL** | | | **$0.00** |

### Pse Groq + Llama 3.3?

| Feature | Vlera |
|---------|-------|
| Kërkesa falas/ditë | **14,400** |
| Tokens/minutë | **500,000** |
| Shpejtësia | **~500 tokens/sek** (LPU hardware) |
| Gjuhët | Shqip + shumëgjuhësh |
| Modeli | **llama-3.3-70b-versatile** |
| Regjistrimi | Falas — [console.groq.com](https://console.groq.com) |

> **Shënim:** Projekti fillimisht përdorte Gemini 2.0 Flash, por u migrua te Groq për shkak të kufizimeve të tier falas të Gemini (quota e shterur shpejt).

### Pse multilingual-e5-large?

- Ranked **#1** për multilingual semantic search (MTEB benchmark)
- Shkarkohet **një herë** (~1.2 GB), pas kësaj 100% **offline**
- **1024 dimensione** — shumë i saktë për kërkim ligjor
- Mbështet shqipen natyrshëm pa fine-tuning

---

## 📁 Struktura e Projektit

```
ligjet_chatbot_v2/
│
├── 📂 src/                         ← Kodi kryesor
│   ├── ingestion.py                ← FAZA 1: PDF → Chunks → FAISS
│   ├── rag_core.py                 ← FAZA 2: Pyetje → FAISS → Groq → Përgjigje
│   ├── test_rag.py                 ← Teste automatike të Fazës 2
│   └── verify_store.py             ← Verifikon FAISS index pas Fazës 1
│
├── 📂 data/
│   └── pdfs/                       ← Vendos TË GJITHA PDF-të këtu
│       ├── Ligji82_2024_Policia.pdf
│       ├── KodiRrugor.pdf          ← FOKUSI KRYESOR i projektit
│       ├── Ligji38_2025.pdf
│       ├── Ligji39_2025_Kufi.pdf
│       ├── Ligji44_2025_Asetet.pdf
│       └── ... (shto sa të duash)
│
├── 📂 vector_store/                 ← Krijohet AUTOMATIKISHT nga Faza 1
│   ├── faiss_index/
│   │   ├── index.faiss             ← Vektorët (2,093+ chunks)
│   │   └── index.pkl               ← Metadata (neni, ligji, faqja)
│   ├── ingestion_report.json       ← Statistika të Fazës 1
│   └── test_report.json            ← Rezultate testesh të Fazës 2
│
├── 📂 docs/                         ← Dokumentacion shtesë
│
├── .env                            ← ⚠️ API keys — MOS e commit në Git!
├── .env.example                    ← Template i .env (commit ky)
├── .gitignore                      ← Eksludon .env, vector_store, .venv
├── requirements.txt                ← Të gjitha dependencies
└── README.md                       ← Ky dokument
```

---

## 🔵 Faza 1 — Ingestion Pipeline (Bërë ✅)

> **File:** `src/ingestion.py`
> **Qëllimi:** Merr PDF-të e ligjeve → i copëzon inteligjentëm → i ruan si vektorë në FAISS

### Si funksionon hap pas hapi

```
┌─────────────────────────────────────────────────────────────┐
│                   INGESTION PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  📄 PDF Files (data/pdfs/)                                   │
│       │                                                      │
│       ▼                                                      │
│  HAPI 1: Text Extraction (pypdf)                             │
│       │  • Lexon çdo faqe                                    │
│       │  • Pastron encoding shqip (ë, ç, ë)                 │
│       │  • Fix hyphenation dhe whitespace                    │
│       │                                                      │
│       ▼                                                      │
│  HAPI 2: Metadata Extraction                                 │
│       │  • Gjen: numrin e ligjit (Nr. 82/2024)               │
│       │  • Gjen: titullin e ligjit                           │
│       │  • Gjen: kreun aktual (KREU I, KREU II...)           │
│       │                                                      │
│       ▼                                                      │
│  HAPI 3: Smart Chunking (LangChain RecursiveTextSplitter)    │
│       │  Separators: ["\nNeni ", "\nKREU ", "\n\n", "\n"]    │
│       │  • Ndan PARA çdo "Neni X" — kurrë BRENDA nenit       │
│       │  • Chunk size: 1000 karaktere, overlap: 200          │
│       │  • Çdo chunk ruan kontekstin e plotë ligjor          │
│       │                                                      │
│       ▼                                                      │
│  HAPI 4: Content Enrichment                                  │
│       │  Çdo chunk pasohet me header:                        │
│       │  "passage: Ligji: [titulli] ([numri])                │
│       │   Kreu: [kreu] | Neni: [neni] — [titulli_nenit]     │
│       │   --- [teksti i nenit]"                              │
│       │                                                      │
│       ▼                                                      │
│  HAPI 5: Embeddings (multilingual-e5-large)                  │
│       │  • Shkarkon modelin herën e parë (~1.2GB)            │
│       │  • Konverton çdo chunk → vektor 1024-dimensional     │
│       │  • Batch processing: 32 chunks njëherësh             │
│       │                                                      │
│       ▼                                                      │
│  HAPI 6: FAISS Index                                         │
│       │  • Ruan të gjithë vektorët në disk                   │
│       │  • Ruan metadata (law_number, article, page...)      │
│       │                                                      │
│       ▼                                                      │
│  📦 vector_store/faiss_index/  ← Gati për Fazën 2!           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Metadata e çdo chunk

```json
{
  "chunk_id":       "a3f9b2c1d4e5",
  "source_file":    "Ligji2.pdf",
  "law_number":     "Nr. 82/2024",
  "law_title":      "POLICINË E SHTETIT",
  "chapter":        "KREU II — KOMPETENCAT E POLICISË",
  "article_number": "Neni 19",
  "article_title":  "Ndalimi i përkohshëm policor",
  "page_number":    8,
  "chunk_index":    47
}
```

### Komandat e Fazës 1

```bash
# Test i shpejtë — pa embedding, pa kosto (debug)
python src/ingestion.py --test --pdf data/pdfs/KodiRrugor.pdf

# Proceson VETËM një PDF
python src/ingestion.py --pdf data/pdfs/KodiRrugor.pdf

# Proceson TË GJITHA PDF-të në data/pdfs/
python src/ingestion.py

# Verifiko që FAISS funksionon
python src/verify_store.py
```

### Rezultatet aktuale të Fazës 1

| Metrika | Vlera |
|---------|-------|
| PDF files të procesuar | **12** |
| Chunks totale të indexuara | **2,093** |
| Nene unike të gjetura | **228** |
| Kosto totale | **$0.00** |
| Koha e embedding (CPU) | ~19 minuta |
| Model i shkarkuar | 1x (pas kësaj offline) |

---

## 🟠 Faza 2 — RAG Core me Groq (Bërë ✅)

> **File:** `src/rag_core.py`
> **Qëllimi:** Merr pyetjen → Kërkon në FAISS → Dërgon te Groq Llama 3.3 → Kthen përgjigje me citim neni

### Si funksionon hap pas hapi

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG PIPELINE                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  👤 Qytetari: "Sa mund të më mbajë policia?"                 │
│       │                                                      │
│       ▼                                                      │
│  HAPI 1: Query Preprocessing                                 │
│       │  • Shton prefix "query: " (e5-large requirement)    │
│       │  • Konverton pyetjen → vektor 1024-dim               │
│       │                                                      │
│       ▼                                                      │
│  HAPI 2: Semantic Search në FAISS                            │
│       │  • Krahason vektorin e pyetjes me 2,093 chunks       │
│       │  • Kthen TOP 5 chunks më relevante                   │
│       │  • Similarity score për çdo chunk                    │
│       │                                                      │
│       ▼                                                      │
│  HAPI 3: Context Building                                    │
│       │  Bashkon chunks:                                     │
│       │  [BURIMI 1] Nr. 82/2024 | Neni 19 — Ndalimi...      │
│       │  [teksti i nenit]                                    │
│       │  ────────────────                                    │
│       │  [BURIMI 2] Nr. 82/2024 | Neni 18...                │
│       │                                                      │
│       ▼                                                      │
│  HAPI 4: Groq — Llama 3.3 70B                                │
│       │  System Prompt (roli i LigjetBot)                    │
│       │  + Konteksti ligjor (chunks)                         │
│       │  + Pyetja e qytetarit                                │
│       │  → Groq gjeneron përgjigje (~500 tok/sek)            │
│       │                                                      │
│       ▼                                                      │
│  HAPI 5: Response Formatting                                 │
│       │  • Përgjigja e qartë                                 │
│       │  • Citimi i neneve (📌 Neni 19, Ligji 82/2024)       │
│       │  • Lista e burimeve                                  │
│       │                                                      │
│       ▼                                                      │
│  ✅ "Sipas Nenit 19, Ligji 82/2024 — max 5 orë..."           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Komandat e Fazës 2

```bash
# Chat interaktiv (mënyra kryesore e përdorimit)
python src/rag_core.py

# Pyetje direkte (për scripting ose testing)
python src/rag_core.py --query "Sa është gjoba për celular gjatë ngasjes?"

# Demo me pyetje të paracaktuara
python src/rag_core.py --demo

# Top-K results custom (default: 5)
python src/rag_core.py --topk 3

# Teste automatike
python src/test_rag.py           # Të gjitha testet
python src/test_rag.py --quick   # 3 teste të shpejta
```

### Demo Pyetjet

| Pyetja | Burimi Ligjor |
|--------|---------------|
| Sa mund të më mbajë policia? | Neni 19, L.82/2024 |
| Sa është gjoba për celular? | Neni 172, Kodi Rrugor |
| A lejohet policia të kontrollojë pa arsye? | Neni 22, L.82/2024 |
| Çfarë dokumentash duhet të kem në makinë? | Kodi Rrugor |
| Si mund të ankoj një gjobë? | Kodi Rrugor |
| Çfarë ndodh nëse refuzoj alkool-testin? | Kodi Rrugor |
| Cilat janë të drejtat e mia në kufi? | Neni 6, L.39/2025 |
| What rights do I have if police stop me? | Art. 19, L.82/2024 (EN) |

---

## ⚡ Setup i Shpejtë (nga zero deri te chatbot që punon)

> **Kërkesat:** Python 3.11–3.14, pip, internet për shkarkimin e modelit herën e parë

### Hapi 1 — Klono Projektin

```bash
git clone https://github.com/username/ligjet-chatbot.git
cd ligjet_chatbot_v2
```

### Hapi 2 — Krijo Virtual Environment

```bash
# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

> Duhet të shohësh `(.venv)` në fillim të terminalit — kjo do të thotë se je brenda environment.

### Hapi 3 — Instalo Dependencies

```bash
pip install -r requirements.txt
```

> ⏳ Herën e parë zgjat ~3–5 minuta pasi shkarkon paketa si `torch`, `transformers` etj.

**Versionet e sakta të LangChain (të testuara):**

| Paketa | Versioni |
|--------|----------|
| `langchain` | 1.3.1 |
| `langchain-community` | 0.4.1 |
| `langchain-huggingface` | 1.2.2 |
| `langchain-groq` | 1.1.2 |
| `faiss-cpu` | 1.13.2 |
| `numpy` | ≥2.1.0 |

> **Python 3.13/3.14:** Kërkon `numpy>=2.1.0` dhe `langchain-huggingface>=1.2.2`. Versionet e vjetra të këtyre paketave **nuk funksionojnë** me Python 3.13+.

### Hapi 4 — Merr Groq API Key (falas, 2 minuta)

1. Shko: **https://console.groq.com**
2. Regjistrohu falas (mund të përdorësh Google/GitHub login)
3. Klik **"API Keys"** → **"Create API Key"**
4. Kopjo key-n (fillon me `gsk_...`)

```bash
# Edito .env dhe shto key-n:
nano .env
```

Ndryshoni vetëm këtë rresht:

```env
GROQ_API_KEY=gsk_your_actual_key_here
```

> `.env.example` ka të gjithë variablat — kopjoje si referencë nëse të duhet.

### Hapi 5 — Vendos PDF-të e Ligjeve

```bash
# Kopjo PDF-të e ligjeve këtu:
cp /path/to/your/pdfs/*.pdf data/pdfs/

# Struktura e duhur:
data/pdfs/
├── KodiRrugor.pdf          ← OBLIGATOR (fokusi kryesor)
├── Ligji82_2024.pdf        ← Policia e Shtetit
├── Ligji39_2025.pdf        ← Kontrolli Kufitar
└── ...
```

### Hapi 6 — Ekzekuto Fazën 1: Ingestion

> **Kjo hap duhet vetëm **njëherë**. Krijon FAISS index nga PDF-të.**

```bash
# Test i shpejtë fillimisht (pa kosto, pa embedding)
python src/ingestion.py --test --pdf data/pdfs/KodiRrugor.pdf

# Full run — proceson të gjitha PDF-të
# Herën e parë shkarkon modelin e embedding (~1.2 GB) dhe zgjat ~20 min
python src/ingestion.py

# Verifiko që u krijua index FAISS
python src/verify_store.py
```

Output i suksesshëm duket kështu:
```
✅ FAISS index i ngarkuar
📊 Chunks totale: 2,093
📌 Nene unike: 228
✅ VERIFY OK - Vector store funksionon perfekt!
```

### Hapi 7 — Ekzekuto Fazën 2: Chat

```bash
# Pyetje direkte (test i shpejtë)
python src/rag_core.py --query "Sa është gjoba për celular gjatë ngasjes?"

# Chat interaktiv
python src/rag_core.py

# Demo me pyetje të paracaktuara
python src/rag_core.py --demo
```

Output i suksesshëm:
```
=======================================================
  LigjetBot — Asistenti Juridik Shqiptar
  Faza 2: RAG Core me Groq llama-3.3-70b-versatile
=======================================================

✅ Embedding model i ngarkuar (intfloat/multilingual-e5-large)
✅ FAISS index i ngarkuar
✅ Groq llama-3.3-70b-versatile i lidhur
🎯 LigjetBot eshte gati!
```

---

## 📚 Ligjet e Indexuara

| # | Emri i Ligjit | Numri | Fokusi | Gjuha |
|---|---------------|-------|--------|-------|
| 1 | **Kodi Rrugor** | Nr. 8378 | 🎯 **FOKUSI KRYESOR** — gjoba, trafik, shoferë | 🇦🇱 |
| 2 | Policia e Shtetit | Nr. 82/2024 | Ndalim, kontroll, arrestim, të drejta | 🇦🇱 |
| 3 | Kontrolli Kufitar | Nr. 39/2025 | Kalimi kufi, dokumente, procedura | 🇦🇱 |
| 4 | Informacioni Udhëtimit | Nr. 38/2025 | Pasagjerë, PNR, të dhëna | 🇦🇱 |
| 5 | Zyra Rikuperimit | Nr. 44/2025 | Asetet kriminale | 🇦🇱 |
| 6 | Masat Sigurisë Publike | Nr. 19/2016 | Siguria publike shtesë | 🇦🇱 |
| 7 | Policia e Shtetit (vjetër) | Nr. 112 | Ligji i vjetër (referencë) | 🇦🇱 |
| 8-12 | Versionet angleze | - | Pyetje në anglisht | 🇬🇧 |

> **Shtimi i ligjeve të reja:** Vendos PDF-in në `data/pdfs/` dhe ekzekuto `python src/ingestion.py`. Sistemi i shton automatikisht pa fshirë ato ekzistueset.

---

## 💬 Shembuj Bisedash Reale

```
════════════════════════════════════════════════════
  LIGJETBOT — Chat Interaktiv
════════════════════════════════════════════════════

Ju: Sa mund të më mbajë policia pa vendim gjykate?

─────────────────────────────────────────────────
Sipas Nenit 19 të Ligjit Nr. 82/2024, ndalimi i
përkohshëm policor mund të zgjasë maksimum 5 orë.

Gjatë kësaj kohe keni të drejtë:
• Të informoheni menjëherë për arsyen e ndalimit
• Të kontaktoni avokat ose person të besuar
• Të heshtin (nuk jeni të detyruar të deklaroni)

Nëse kalojnë 5 orët pa vendim gjykate, ndalimi
është i paligjshëm dhe mund të ankohet.

📌 Neni 19, Ligji Nr. 82/2024 — Faqe 8
📌 Neni 4, Ligji Nr. 82/2024 — Faqe 3
─────────────────────────────────────────────────

Ju: Sa është gjoba për celular gjatë ngasjes?

─────────────────────────────────────────────────
Sipas Nenit 172 të Kodit Rrugor Nr. 8378, përdorimi
i telefonit gjatë ngasjes dënohet me:

• Gjobë: 5,000 — 10,000 lekë
• Heqje pike patente: 10 pikë
• Keni 30 ditë për ankim nga data e gjobës

📌 Neni 172, Kodi Rrugor Nr. 8378 — Faqe 98
─────────────────────────────────────────────────

Ju: What rights do I have if police stop me?

─────────────────────────────────────────────────
According to Article 19, Law No. 82/2024, temporary
police detention cannot exceed 5 hours.

You have the right to:
• Be immediately informed of the reason
• Contact a lawyer or trusted person immediately
• Remain silent (no obligation to make statements)

📌 Article 19, Law No. 82/2024 — Page 8
─────────────────────────────────────────────────
```

---

## 🔧 Troubleshooting

| Problem | Zgjidhja |
|---------|----------|
| `source: no such file or directory: .venv/bin/activate` | `.venv` nuk ekziston — ekzekuto fillimisht `python3 -m venv .venv` |
| `ResolutionImpossible` gjatë `pip install` | Konflikti i versioneve — sigurohu që `requirements.txt` ka versionet nga tabela e mësipërme |
| `TypeError: cannot pickle 'classmethod'` | Versioni i vjetër i `langchain-huggingface` — duhet `>=1.2.2` për Python 3.13/3.14 |
| `No module named 'langchain.schema'` | LangChain 1.x e hoqi këtë modul — kode e vjetër, bëj update të importeve |
| `429 ResourceExhausted` (Groq) | Ke arritur limitin e minutës — prit 60 sek dhe provo sërish |
| `GROQ_API_KEY nuk eshte konfiguruar` | Shto `GROQ_API_KEY=gsk_...` në `.env` nga **console.groq.com** |
| Vector store nuk ekziston | Ekzekuto `python src/ingestion.py` |
| Chunks=0 pas ingestion | PDF është i skanuar (image-based) — nuk mund të lexohet tekstin |
| Ngadalë gjatë embedding | Normal herën e parë — modeli 1.2GB shkarkohet njëherë |
| `externally-managed-environment` | Krijo `.venv` — mos instalo pa virtual environment |

---

## 👥 Udhëzues për Ekipin

### Setup Fillestar (për anëtarë të rinj)

```bash
# 1. Klono
git clone <repo-url>
cd ligjet_chatbot_v2

# 2. Virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Mac/Linux
# .venv\Scripts\activate       # Windows

# 3. Instalo
pip install -r requirements.txt

# 4. Konfiguro Groq API key
#    Shko: https://console.groq.com → API Keys → Create API Key
nano .env
# Shto: GROQ_API_KEY=gsk_your_key_here

# 5. Merr vector store nga team lead (kërko faiss_index/ folder)
# OSE ekzekuto ingestion vetë (kërkon PDF-të dhe ~20 min herën e parë)
python src/ingestion.py

# 6. Testo
python src/rag_core.py --query "Sa është gjoba për celular?"
```

### Rregullat e Git

```bash
# ✅ COMMIT këto
git add src/
git add requirements.txt
git add README.md
git add .gitignore
git add .env.example           # Template pa keys reale

# ❌ MOS COMMIT kurrë
# .env                         ← ka API keys private
# vector_store/                ← shumë i madh (GB)
# data/pdfs/                   ← PDF ligjet (opcional)
# .venv/                       ← environment lokal
# __pycache__/
```

### Workflow i Rekomanduar

```bash
# Para çdo sesioni
source .venv/bin/activate

# Para push
python src/test_rag.py --quick   # Kontrollo që gjithçka punon

# Branches
main           ← kodi i qëndrueshëm
faza-2         ← RAG core (aktual)
faza-3-ui      ← Web interface (i ardhshëm)
```

### Struktura e Commit Messages

```
feat: shto Kodin Rrugor në ingestion pipeline
fix: rregull metadata extraction për ligje angleze
docs: përditëso README me fazën 2
test: shto test cases për gjobat rrugore
refactor: optimizo context building në rag_core
```

---

## 📊 Roadmap

```
FAZA 1 ✅          FAZA 2 ✅          FAZA 3 ⏳          FAZA 4 ⏳          FAZA 5 ⏳
──────────         ──────────         ──────────         ──────────         ──────────
Ingestion          RAG Core           Web Interface      Memory             Evaluation
Pipeline           Groq Llama 3.3     Flask Chat UI      Session Mgmt       RAGAS Metrics

• PDF → FAISS      • FAISS search     • Web chat UI      • Histori           • Precision
• 12 ligje         • Groq LLM         • Citim neni       • Kontekst          • Recall
• 2,093 chunks     • System prompt    • Mobile UI        • Multi-turn        • Faithfulness
• 228 nene         • Chat interaktiv  • AL/EN toggle     • User sessions     • Auto-feedback
• $0 kosto         • Tests auto.      • Deploy Heroku

                                      FAZA 6 ⏳
                                      ──────────
                                      Advanced

                                      • Hybrid search
                                      • Auto-sync ligje
                                      • 100+ ligje
                                      • 10,000 users
```

---

## 📈 Statistikat e Projektit

```
Gjendja aktuale (pas Fazës 1 + 2):

📁 PDF files:          12 ligje (shqip + anglisht)
📊 Chunks totale:   2,093 copëza të indexuara
📌 Nene unike:        228 nene të gjenduara
🔍 Semantic search:   ~200ms për pyetje
🤖 LLM latency:       ~0.5–1 sekondë (Groq LPU)
💰 Kosto totale:      $0.00
🌐 Gjuhët:            Shqip + Anglisht
```

---

## ⚖️ Disclaimer

> **LigjetBot** është mjet informues bazuar në Inteligjencë Artificiale.
> Nuk zëvendëson këshillën juridike profesionale të një avokati të licencuar.
> Për situata serioze ligjore, gjithmonë konsultohuni me avokat.

---

*Ndërtuar me ❤️ për qytetarët shqiptarë*

*Stack: Python • LangChain 1.x • FAISS • HuggingFace multilingual-e5-large • Groq Llama 3.3 70B*
