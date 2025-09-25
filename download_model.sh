#!/usr/bin/env bash
set -euo pipefail

set -a
source <(tr -d '\r' < .env-test)
set +a

MODELS_DIR="$(pwd)/ollama"
mkdir -p "$MODELS_DIR"

CID=$(docker run -d \
  --mount type=bind,source="$MODELS_DIR",target=/root/.ollama \
  -p 11434:11434 \
  ollama/ollama:latest serve)


until curl -s http://localhost:11434/api/version >/dev/null; do
  sleep 1
done

docker exec "$CID" ollama pull "$OLLAMA_GENERATOR"
docker exec "$CID" ollama pull "$OLLAMA_RETRIEVER"

docker stop "$CID"