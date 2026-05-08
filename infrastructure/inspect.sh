#!/bin/bash
# Instala dependencias de inspección y corre inspect_chunks.py dentro del contenedor.
# Uso: ./inspect.sh [--splitter TIPO] [--split_length N] [--split_overlap N]
#
# Splitters disponibles:
#   Haystack   : recursive (default), word, sentence, passage
#   LlamaIndex : llama_sentence, llama_semantic, llama_semantic_double
#   LangChain  : langchain_recursive
#
# Ejemplos:
#   ./inspect.sh
#   ./inspect.sh --splitter sentence --split_length 5
#   ./inspect.sh --splitter llama_semantic
#   ./inspect.sh --splitter langchain_recursive --split_length 150
#
# Los archivos de salida quedan en infrastructure/

set -e

CONTAINER="infrastructure-pipelines-1"
SCRIPT_SRC="$(dirname "$0")/../examples/local_docs/inspect_chunks.py"
SCRIPT_DST="/app/pipelines/inspect_chunks.py"

echo "==> Copiando script al contenedor..."
docker cp "$SCRIPT_SRC" "$CONTAINER:$SCRIPT_DST"

echo "==> Instalando dependencias de inspección..."
docker exec "$CONTAINER" pip install -q \
    llama-index-core \
    llama-index-embeddings-ollama \
    langchain-text-splitters

echo "==> Corriendo inspector con args: $@"
docker exec -t "$CONTAINER" python "$SCRIPT_DST" "$@"

# Copiar todos los archivos de splitter generados a infrastructure/
echo "==> Copiando resultados..."
for f in $(docker exec "$CONTAINER" find /app/pipelines -maxdepth 1 -name "splitter_*.txt" 2>/dev/null); do
    fname=$(basename "$f")
    docker cp "$CONTAINER:$f" "$(dirname "$0")/$fname" 2>/dev/null || true
    echo "    -> $fname"
done
