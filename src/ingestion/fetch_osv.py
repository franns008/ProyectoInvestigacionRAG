"""Fetch OSV (advisories de PyPI) y guarda el dump crudo en data/raw/osv/.

Es el archivo obligatorio del escaneo de dependencias: deps/cli.py lo busca en
`data/raw/osv/all.zip`.

Uso:
    python src/ingestion/fetch_osv.py
    python src/ingestion/fetch_osv.py --force   # rebaja aunque ya esté
"""

import argparse

from _http import RAW_DIR, Result, download

SOURCE = "osv"
URL = "https://storage.googleapis.com/osv-vulnerabilities/PyPI/all.zip"
DEST = RAW_DIR / "osv" / "all.zip"
MIN_BYTES = 20_000_000


def fetch(force: bool = False) -> Result:
    return download(URL, DEST, MIN_BYTES, SOURCE, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch OSV (advisories de PyPI) a data/raw/osv/")
    parser.add_argument("--force", action="store_true",
                        help="rebaja el dump aunque el servidor diga que no cambió")
    args = parser.parse_args()
    fetch(force=args.force)


if __name__ == "__main__":
    main()
