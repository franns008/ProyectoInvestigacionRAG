"""Fetch CISA KEV (vulnerabilidades con explotación confirmada) a data/raw/kev/.

Opcional para el escaneo de dependencias: sin él, deps/cli.py corre igual pero sin la
marca [KEV], que es la señal de prioridad más fuerte del embudo.

Uso:
    python src/ingestion/fetch_kev.py
    python src/ingestion/fetch_kev.py --force   # rebaja aunque ya esté
"""

import argparse

from _http import RAW_DIR, Result, download

SOURCE = "kev"
URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
DEST = RAW_DIR / "kev" / "known_exploited_vulnerabilities.json"
MIN_BYTES = 500_000


def fetch(force: bool = False) -> Result:
    return download(URL, DEST, MIN_BYTES, SOURCE, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch CISA KEV a data/raw/kev/")
    parser.add_argument("--force", action="store_true",
                        help="rebaja el catálogo aunque el servidor diga que no cambió")
    args = parser.parse_args()
    fetch(force=args.force)


if __name__ == "__main__":
    main()
