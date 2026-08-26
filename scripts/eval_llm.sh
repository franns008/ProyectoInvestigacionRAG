#!/usr/bin/env bash
# Ídem eval.sh pero para el Tier 3 (juez LLM). Uso: scripts/eval_llm.sh [args]
set -euo pipefail
# Ver el comentario en eval.sh: Git Bash reescribe las rutas POSIX de los argumentos.
export MSYS_NO_PATHCONV=1
cd "$(dirname "$0")/.."
GIT_COMMIT="$(git rev-parse --short HEAD)"
GIT_BRANCH="$(git branch --show-current)"
GIT_DIRTY="$([ -n "$(git status --porcelain)" ] && echo 1 || echo 0)"
cd infrastructure
docker compose exec \
  -e GIT_COMMIT="$GIT_COMMIT" -e GIT_BRANCH="$GIT_BRANCH" -e GIT_DIRTY="$GIT_DIRTY" \
  pipelines python /app/pipelines/eval/run_eval_llm.py "$@"
