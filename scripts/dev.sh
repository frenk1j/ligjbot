#!/usr/bin/env bash
# Nis LIGJBOT lokalisht dhe hap në shfletues
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d ".venv" ]]; then
  echo "❌ Mungon .venv — ekzekuto: python3 -m venv .venv && pip install -r requirements.txt"
  exit 1
fi

source .venv/bin/activate

PORT="${PORT:-5001}"
URL="http://localhost:${PORT}"

if lsof -ti :"$PORT" >/dev/null 2>&1; then
  echo "✅ Serveri po punon tashmë në port $PORT"
else
  echo "🚀 Duke nisur LIGJBOT..."
  python src/app.py &
  sleep 2
fi

if command -v open >/dev/null 2>&1; then
  open -a "Google Chrome" "$URL" 2>/dev/null || open "$URL"
fi

echo "🌐 Hapet: $URL"
