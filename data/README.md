# Të dhëna ligjore

Vendos skedarët PDF të ligjeve shqiptare në `data/pdfs/`.

Pas shtimit ose ndryshimit të PDF-ve, rigjenero indeksin:

```bash
python scripts/ingestion.py
python scripts/verify_store.py
```

`eval_questions.json` përmban pyetje test për evaluimin e cilësisë RAG (`scripts/evaluate_rag.py`).
