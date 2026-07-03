"""Fetch NVD/CVE (Capa 1) y guarda el JSON crudo en data/raw/nvd/.

No normaliza ni transforma nada — eso lo hace un paso posterior siguiendo
docs/data_transform_spec.md. Este script solo trae los datos y los persiste
tal cual llegan de la API, con un checkpoint para no repetir la carga completa
en cada corrida.

Uso:
    python src/ingestion/fetch_nvd.py            # incremental (usa el checkpoint si existe)
    python src/ingestion/fetch_nvd.py --full      # ignora el checkpoint, trae el catálogo completo
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
RESULTS_PER_PAGE = 2000
REQUEST_DELAY_SECONDS = 0.7  # ~43 req/30s, margen bajo el límite de 50/30s con API key
MAX_DATE_RANGE_DAYS = 120    # restricción de la API para lastMod*Date
MAX_RETRIES = 3

REPO_ROOT = Path(__file__).resolve().parents[2]
# Escribe dentro de data/raw/, que el docker-compose bind-montea como el volumen
# `rawdata` → /app/pipelines/rawdata (INPUT_DIR del pipeline), para que el
# NVDJsonConverter levante estas páginas en el próximo arranque.
RAW_DIR = REPO_ROOT / "data" / "raw" / "nvd"
CHECKPOINT_PATH = RAW_DIR / "_checkpoint.json"
DATE_FMT = "%Y-%m-%dT%H:%M:%S.000Z"


def _load_api_key() -> str:
    load_dotenv(REPO_ROOT / "infrastructure" / ".env")
    api_key = os.environ.get("NVD_API_KEY")
    if not api_key:
        raise RuntimeError("NVD_API_KEY no encontrada (se esperaba en infrastructure/.env)")
    return api_key


def _read_checkpoint() -> str | None:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text())["last_run_utc"]
    return None


def _write_checkpoint(run_start: str) -> None:
    CHECKPOINT_PATH.write_text(json.dumps({"last_run_utc": run_start}, indent=2))


def _date_windows(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Parte [start, end) en ventanas <= MAX_DATE_RANGE_DAYS (restricción de la API de NVD)."""
    windows = []
    current = start
    while current < end:
        window_end = min(current + timedelta(days=MAX_DATE_RANGE_DAYS), end)
        windows.append((current, window_end))
        current = window_end
    return windows


def _get_with_retries(headers: dict, params: dict) -> dict:
    for attempt in range(1, MAX_RETRIES + 1):
        response = requests.get(BASE_URL, headers=headers, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        if attempt == MAX_RETRIES:
            response.raise_for_status()
        wait = 2 ** attempt
        print(f"[nvd] status {response.status_code}, reintentando en {wait}s (intento {attempt}/{MAX_RETRIES})")
        time.sleep(wait)
    raise RuntimeError("unreachable")


def _fetch_window(api_key: str, run_dir: Path, page_counter: list[int],
                   window_start: datetime | None, window_end: datetime | None) -> int:
    headers = {"apiKey": api_key}
    params: dict = {"resultsPerPage": RESULTS_PER_PAGE, "startIndex": 0}
    if window_start is not None:
        params["lastModStartDate"] = window_start.strftime(DATE_FMT)
        params["lastModEndDate"] = window_end.strftime(DATE_FMT)

    total_results = None
    fetched = 0

    while total_results is None or params["startIndex"] < total_results:
        data = _get_with_retries(headers, params)
        total_results = data["totalResults"]
        vulnerabilities = data.get("vulnerabilities", [])
        fetched += len(vulnerabilities)

        page = page_counter[0]
        out_path = run_dir / f"cves_page_{page:04d}.json"
        out_path.write_text(json.dumps(data))
        print(f"[nvd] página {page}: {len(vulnerabilities)} CVEs -> {out_path.name} "
              f"({params['startIndex'] + len(vulnerabilities)}/{total_results})")
        page_counter[0] += 1

        params["startIndex"] += RESULTS_PER_PAGE
        if params["startIndex"] < total_results:
            time.sleep(REQUEST_DELAY_SECONDS)

    return fetched


def fetch_cves(api_key: str, last_mod_start: str | None, run_start_dt: datetime) -> int:
    run_dir = RAW_DIR / run_start_dt.strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    page_counter = [0]

    if last_mod_start is None:
        return _fetch_window(api_key, run_dir, page_counter, None, None)

    start_dt = datetime.strptime(last_mod_start, DATE_FMT).replace(tzinfo=timezone.utc)
    total = 0
    for window_start, window_end in _date_windows(start_dt, run_start_dt):
        total += _fetch_window(api_key, run_dir, page_counter, window_start, window_end)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NVD/CVE (Capa 1) a data/raw/nvd/")
    parser.add_argument("--full", action="store_true", help="Ignora el checkpoint y trae el catálogo completo")
    args = parser.parse_args()

    api_key = _load_api_key()
    run_start_dt = datetime.now(timezone.utc)
    run_start = run_start_dt.strftime(DATE_FMT)
    last_mod_start = None if args.full else _read_checkpoint()

    mode = f"incremental desde {last_mod_start}" if last_mod_start else "completa"
    print(f"[nvd] Iniciando carga {mode}")

    total = fetch_cves(api_key, last_mod_start, run_start_dt)
    _write_checkpoint(run_start)

    print(f"[nvd] Listo: {total} CVEs guardados en {RAW_DIR}")


if __name__ == "__main__":
    main()
