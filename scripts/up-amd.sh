#!/usr/bin/env bash
# Levanta el stack con el overlay AMD (Vulkan). Uso: scripts/up-amd.sh [args de compose]
set -euo pipefail

cd "$(dirname "$0")/../infrastructure"

docker compose -f docker-compose.yml -f docker-compose.amd.yml up -d --build "$@"
