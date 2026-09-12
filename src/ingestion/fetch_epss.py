"""Fetch EPSS (probabilidad de explotación) y guarda el CSV crudo en data/raw/epss/.

Opcional para el escaneo de dependencias: sin él, deps/cli.py corre igual pero sin el
filtro por EPSS. El feed se recalcula a diario.

La URL `-current` responde 302 al CSV del día; requests sigue el redirect solo.

Uso:
    python src/ingestion/fetch_epss.py
    python src/ingestion/fetch_epss.py --force   # rebaja aunque ya esté
"""

import argparse

from _http import RAW_DIR, Result, download

SOURCE = "epss"
URL = "https://epss.empiricalsecurity.com/epss_scores-current.csv.gz"
DEST = RAW_DIR / "epss" / "epss_scores-current.csv.gz"
MIN_BYTES = 1_000_000


def fetch(force: bool = False) -> Result:
    return download(URL, DEST, MIN_BYTES, SOURCE, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch EPSS a data/raw/epss/")
    parser.add_argument("--force", action="store_true",
                        help="rebaja el CSV aunque el servidor diga que no cambió")
    args = parser.parse_args()
    fetch(force=args.force)


if __name__ == "__main__":
    main()
