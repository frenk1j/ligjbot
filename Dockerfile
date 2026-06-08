FROM python:3.11-slim

WORKDIR /app

# System deps for torch/faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    LIGJBOT_BOOTSTRAP=1 \
    VECTOR_STORE_PATH=/app/vector_store/faiss_index \
    PDF_FOLDER=/app/data/pdfs \
    PORT=5001

# Build vector store nëse mungon (herën e parë ~5-10 min)
RUN if [ ! -d "/app/vector_store/faiss_index" ]; then \
      echo y | python scripts/ingestion.py; \
    fi

EXPOSE 5001

CMD ["bash", "scripts/start_production.sh"]
