#!/usr/bin/env bash
# Baja los tres dumps que necesita el escaneo de dependencias.
#
# PROVISIONAL: la Fase 1 de docs/escaneo_dependencias.md prevé fetchers en Python
# (src/ingestion/fetch_osv.py y compañía) con checkpoint e incrementales, siguiendo el
# patrón de fetch_nvd.py. Este script existe para que el escaneo se pueda replicar HOY
# sin esperar esa fase; cuando los fetchers estén, se borra.
#
#   ./scripts/fetch_deps_data.sh          # baja lo que falte
#   ./scripts/fetch_deps_data.sh --force  # vuelve a bajar todo
#
# Los tres archivos van a data/raw/, que está en .gitignore: son dumps regenerables,
# no artefactos del repo.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data/raw"

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

# nombre|destino|url|tamaño mínimo esperado en bytes (para detectar descargas truncadas)
FUENTES=(
  "OSV (advisories de PyPI)|osv/all.zip|https://storage.googleapis.com/osv-vulnerabilities/PyPI/all.zip|20000000"
  "EPSS (probabilidad de exploit)|epss/epss_scores-current.csv.gz|https://epss.empiricalsecurity.com/epss_scores-current.csv.gz|1000000"
  "CISA KEV (explotadas hoy)|kev/known_exploited_vulnerabilities.json|https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json|500000"
)

for fuente in "${FUENTES[@]}"; do
  IFS="|" read -r nombre destino url minimo <<< "${fuente}"
  ruta="${DATA_DIR}/${destino}"

  if [[ -f "${ruta}" && ${FORCE} -eq 0 ]]; then
    echo "✓ ${nombre} — ya está ($(du -h "${ruta}" | cut -f1)). Usá --force para rebajarlo."
    continue
  fi

  echo "↓ ${nombre}…"
  mkdir -p "$(dirname "${ruta}")"
  # A un temporal primero: si la descarga se corta, no queda un archivo a medias que
  # el escáner intente leer.
  if ! curl -fsSL --retry 3 --retry-delay 2 -o "${ruta}.tmp" "${url}"; then
    rm -f "${ruta}.tmp"
    echo "✗ ${nombre} — falló la descarga de ${url}" >&2
    exit 1
  fi

  tamano=$(stat -c%s "${ruta}.tmp" 2>/dev/null || stat -f%z "${ruta}.tmp")
  if (( tamano < minimo )); then
    rm -f "${ruta}.tmp"
    echo "✗ ${nombre} — descarga sospechosamente chica (${tamano} bytes, se esperaban >${minimo})" >&2
    exit 1
  fi

  mv "${ruta}.tmp" "${ruta}"
  echo "✓ ${nombre} — $(du -h "${ruta}" | cut -f1)"
done

echo
echo "Listo. Los datos están en ${DATA_DIR}."
echo "Probalo:  PYTHONPATH=src/pipeline .venv/bin/python -m deps.cli <requirements.txt> --data data/raw"
