#!/usr/bin/env bash
# Corre el eval harness dentro del container `pipelines`, inyectando metadata de git
# (el container no ve .git). Uso: scripts/eval.sh [args de run_eval.py]
set -euo pipefail
# Git Bash (MSYS) en Windows reescribe los argumentos que parecen rutas POSIX:
# /app/pipelines/... se convierte en C:/Program Files/Git/app/pipelines/... y el
# container no lo encuentra. Sin efecto en Linux/macOS.
export MSYS_NO_PATHCONV=1
cd "$(dirname "$0")/.."
GIT_COMMIT="$(git rev-parse --short HEAD)"
GIT_BRANCH="$(git branch --show-current)"
GIT_DIRTY="$([ -n "$(git status --porcelain)" ] && echo 1 || echo 0)"
cd infrastructure
docker compose exec \
  -e GIT_COMMIT="$GIT_COMMIT" -e GIT_BRANCH="$GIT_BRANCH" -e GIT_DIRTY="$GIT_DIRTY" \
  pipelines python /app/pipelines/eval/run_eval.py "$@"
