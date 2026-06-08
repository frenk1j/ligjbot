#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export LIGJBOT_BOOTSTRAP=1
export VECTOR_STORE_PATH="${VECTOR_STORE_PATH:-$ROOT/vector_store/faiss_index}"
export PDF_FOLDER="${PDF_FOLDER:-$ROOT/data/pdfs}"

PORT="${PORT:-5001}"

echo "🚀 LIGJBOT production — port $PORT"
echo "   Vector store: $VECTOR_STORE_PATH"

exec gunicorn \
  --chdir "$ROOT/src" \
  --bind "0.0.0.0:${PORT}" \
  --workers 1 \
  --threads 4 \
  --timeout 180 \
  app:app
