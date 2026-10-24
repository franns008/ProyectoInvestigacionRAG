"""Baja el catálogo CWE completo de MITRE y deja el XML plano en data/raw/cwec_latest.xml.

Descarga condicional (304 entre releases), sin API key. Por qué el XML va suelto y por qué
no se usa la REST API de MITRE: docs/ingestion_fetchers.md.

    python src/ingestion/fetch_cwe.py [--force]
"""

import io
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone

from _http import (ACTUALIZADO, NUEVO, RAW_DIR, REPO_ROOT, SIN_CAMBIOS, Result,
                   checked_payload, conditional_get, human_size, run_cli, write_atomic)

SOURCE = "cwe"
URL = "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip"
ARCHIVE_DIR = RAW_DIR / "cwe"
PUBLISHED_XML = RAW_DIR / "cwec_latest.xml"
MIN_BYTES = 1_000_000
# Con los primeros bytes alcanza para leer los atributos del <Weakness_Catalog> raíz.
HEADER_BYTES = 4096


def _catalog_header(xml_head: bytes) -> str:
    """'4.20 (2026-04-30)' leyendo sólo el elemento raíz.

    Parsear los 18 MB enteros para dos atributos cuesta ~280 ms y ~130 MB de RAM; con el
    parser incremental son microsegundos, porque alcanza con el primer elemento.
    """
    parser = ET.XMLPullParser(events=("start",))
    parser.feed(xml_head)
    for _, element in parser.read_events():
        return f"{element.get('Version') or '?'} ({element.get('Date') or '?'})"
    return "versión desconocida"


def fetch(force: bool = False) -> Result:
    response = conditional_get(URL, SOURCE, PUBLISHED_XML, force=force)

    if response.status_code == 304:
        with PUBLISHED_XML.open("rb") as handle:
            version = _catalog_header(handle.read(HEADER_BYTES))
        detail = f"CWE {version}, {human_size(PUBLISHED_XML.stat().st_size)}"
        print(f"[{SOURCE}] sin cambios ({detail})")
        return Result(SIN_CAMBIOS, detail)

    existia = PUBLISHED_XML.exists()
    zip_bytes = checked_payload(response, MIN_BYTES)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        # El nombre de adentro cambia con cada release (cwec_v4.20.xml), así que se
        # toma del namelist en vez de hardcodearlo.
        names = [name for name in archive.namelist() if name.lower().endswith(".xml")]
        if not names:
            raise RuntimeError(f"el zip de CWE no trae ningún .xml: {archive.namelist()}")
        xml_name = names[0]
        xml_bytes = archive.read(xml_name)

    run_dir = ARCHIVE_DIR / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_atomic(run_dir / "cwec_latest.xml.zip", zip_bytes)
    write_atomic(PUBLISHED_XML, xml_bytes, response)

    status = ACTUALIZADO if existia else NUEVO
    detail = (f"CWE {_catalog_header(xml_bytes[:HEADER_BYTES])}, "
              f"{human_size(len(xml_bytes))} -> {PUBLISHED_XML.relative_to(REPO_ROOT)}")
    print(f"[{SOURCE}] {status}: {detail}")
    return Result(status, detail)


def main() -> None:
    run_cli("Fetch del catálogo completo de CWE a data/raw/", fetch, "el catálogo")


if __name__ == "__main__":
    main()
