"""Indexa CVEs de NVD (JSON crudo en data/raw/nvd/) en el mismo document store
pgvector que usa el RAG (src/pipeline/pipeline_ciberseguridad.py).

No chunkea: cada CVE es un Document atómico (regla de docs/data_transform_spec.md
para toda la Capa 1). Solo se embebe la descripción; el resto de los campos quedan
como metadata filtrable, sin vectorizar.

El id de cada Document es un hash determinístico de cve_id, así reindexar (por
ejemplo tras un fetch incremental) actualiza el CVE existente en vez de duplicarlo.

Requiere que los contenedores `vdb` (pgvector) y `ollama` de infrastructure/
estén levantados (docker compose up), y NVD_API_KEY/PGVECTOR_* en infrastructure/.env
(los mismos que ya usa docker-compose).

Uso:
    python src/ingestion/index_nvd.py                     # indexa todo lo que haya en data/raw/nvd/
    python src/ingestion/index_nvd.py --raw-dir data/raw/nvd/20260701T230947Z   # solo una corrida puntual
    python src/ingestion/index_nvd.py --limit 500          # solo los primeros N CVEs (para probar)
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from haystack import Document
from haystack.utils import Secret
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "nvd"

DB_TABLE = "ciberseguridad_docs"              # misma tabla que pipeline_ciberseguridad.py
KEYWORD_INDEX_NAME = "ciberseguridad_keyword_index"
EMBEDDING_DIMENSION = 1024                    # dimensión de bge-m3
EMBEDDING_MODEL = "bge-m3"
BATCH_SIZE = 200                              # CVEs por lote (parse -> embed -> write)

# CWE "genéricos" que NVD usa cuando no hay un CWE real mapeado: no son IDs de CWE reales.
_NON_CWE_PLACEHOLDERS = {"NVD-CWE-Other", "NVD-CWE-noinfo"}


def _load_env() -> None:
    load_dotenv(REPO_ROOT / "infrastructure" / ".env")


def _get_document_store() -> PgvectorDocumentStore:
    user = os.environ.get("PGVECTOR_USR")
    password = os.environ.get("PGVECTOR_PASS")
    port = os.environ.get("VDB_PORT", "5433")
    if not user or not password:
        raise RuntimeError(
            "PGVECTOR_USR/PGVECTOR_PASS no encontradas (se esperaban en infrastructure/.env)"
        )
    connection_string = f"postgresql://{user}:{password}@localhost:{port}/pgvdb"
    return PgvectorDocumentStore(
        connection_string=Secret.from_token(connection_string),
        embedding_dimension=EMBEDDING_DIMENSION,
        table_name=DB_TABLE,
        keyword_index_name=KEYWORD_INDEX_NAME,
    )


def _ollama_url() -> str:
    port = os.environ.get("OLLAMA_PORT", "11434")
    return f"http://localhost:{port}"


# ------------------------------------------------------------------
# Parseo de un CVE (JSON crudo de la API 2.0) -> Document de Haystack
# ------------------------------------------------------------------

def _pick_description(cve: dict) -> str:
    descriptions = cve.get("descriptions", [])
    for d in descriptions:
        if d.get("lang") == "en":
            return d["value"]
    return descriptions[0]["value"] if descriptions else ""


def _pick_cvss(cve: dict) -> tuple[float | None, str | None, str | None]:
    """Devuelve (score, vector, version) priorizando la versión de CVSS más común/reciente
    presente en el registro. No todos los CVE traen todas las versiones."""
    metrics = cve.get("metrics", {})
    for key, version in (
        ("cvssMetricV31", "3.1"),
        ("cvssMetricV40", "4.0"),
        ("cvssMetricV30", "3.0"),
        ("cvssMetricV2", "2.0"),
    ):
        entries = metrics.get(key)
        if entries:
            data = entries[0]["cvssData"]
            return data.get("baseScore"), data.get("vectorString"), version
    return None, None, None


def _cwe_ids(cve: dict) -> list[str]:
    ids = set()
    for w in cve.get("weaknesses", []):
        for desc in w.get("description", []):
            value = desc.get("value", "")
            if value.startswith("CWE-") and value not in _NON_CWE_PLACEHOLDERS:
                ids.add(value)
    return sorted(ids)


def _vendors_products(cve: dict) -> tuple[list[str], list[str]]:
    vendors, products = set(), set()
    for a in cve.get("affected", []):
        for ad in a.get("affectedData", []):
            vendor, product = ad.get("vendor"), ad.get("product")
            if vendor and vendor != "n/a":
                vendors.add(vendor)
            if product and product != "n/a":
                products.add(product)

    if not vendors and not products:
        # Fallback: parsear vendor/producto directo de los criteria CPE 2.3
        # (cpe:2.3:a:<vendor>:<product>:...) cuando "affected" no trae datos útiles.
        for config in cve.get("configurations", []):
            for node in config.get("nodes", []):
                for match in node.get("cpeMatch", []):
                    parts = match.get("criteria", "").split(":")
                    if len(parts) > 4:
                        if parts[3] not in ("*", ""):
                            vendors.add(parts[3])
                        if parts[4] not in ("*", ""):
                            products.add(parts[4])

    return sorted(vendors), sorted(products)


def _references(cve: dict, limit: int = 10) -> list[str]:
    urls, seen = [], set()
    for r in cve.get("references", []):
        url = r.get("url")
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
        if len(urls) >= limit:
            break
    return urls


def cve_to_document(cve: dict) -> Document:
    cve_id = cve["id"]
    cvss_score, cvss_vector, cvss_version = _pick_cvss(cve)
    vendors, products = _vendors_products(cve)

    return Document(
        id=hashlib.sha256(cve_id.encode()).hexdigest(),
        content=_pick_description(cve),
        meta={
            "source_type": "nvd_cve",
            "cve_id": cve_id,
            "vuln_status": cve.get("vulnStatus"),
            "published_date": cve.get("published"),
            "last_modified": cve.get("lastModified"),
            "cvss_score": cvss_score,
            "cvss_vector": cvss_vector,
            "cvss_version": cvss_version,
            "vendors": vendors,
            "products": products,
            "cwe_ids": _cwe_ids(cve),
            "references": _references(cve),
        },
    )


# ------------------------------------------------------------------
# Lectura de los JSON crudos e indexado por lotes
# ------------------------------------------------------------------

def _iter_cve_records(raw_dir: Path):
    page_files = sorted(raw_dir.glob("**/cves_page_*.json"))
    if not page_files:
        raise RuntimeError(f"No se encontraron cves_page_*.json bajo {raw_dir}")
    seen_ids = set()
    for page_file in page_files:
        data = json.loads(page_file.read_text())
        for item in data.get("vulnerabilities", []):
            cve = item["cve"]
            if cve["id"] in seen_ids:
                continue  # mismo CVE repetido entre corridas (ej. incremental re-fetch)
            seen_ids.add(cve["id"])
            yield cve


def index_nvd(raw_dir: Path, limit: int | None = None) -> int:
    store = _get_document_store()
    embedder = OllamaDocumentEmbedder(
        model=EMBEDDING_MODEL,
        url=_ollama_url(),
        batch_size=32,
    )

    batch: list[Document] = []
    total_indexed = 0

    def _flush(batch: list[Document]) -> None:
        nonlocal total_indexed
        if not batch:
            return
        embedded = embedder.run(batch)["documents"]
        store.write_documents(embedded, policy=DuplicatePolicy.OVERWRITE)
        total_indexed += len(embedded)
        print(f"[index-nvd] {total_indexed} CVEs indexados (último lote: {len(embedded)})")

    for cve in _iter_cve_records(raw_dir):
        batch.append(cve_to_document(cve))
        if len(batch) >= BATCH_SIZE:
            _flush(batch)
            batch = []
        if limit is not None and total_indexed + len(batch) >= limit:
            break

    _flush(batch)
    return total_indexed


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa CVEs de NVD en pgvector")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR,
                         help="Carpeta con cves_page_*.json (default: data/raw/nvd/, todas las corridas)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Indexar solo los primeros N CVEs (para pruebas)")
    args = parser.parse_args()

    _load_env()
    total = index_nvd(args.raw_dir, limit=args.limit)
    print(f"[index-nvd] Listo: {total} CVEs indexados en la tabla '{DB_TABLE}'")


if __name__ == "__main__":
    main()
