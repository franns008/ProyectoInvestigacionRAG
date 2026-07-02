"""Indexa los CVEs crudos de data/raw/nvd/ en la tabla pgvector del RAG.

Lee los JSON que dejó src/ingestion/fetch_nvd.py, arma un Document de Haystack
por CVE (sin chunkear — una entrada CVE es atómica, ver docs/indexing_design_capa1.md),
lo embebe con Ollama (bge-m3) y lo escribe en la MISMA tabla que usa el pipeline
de producción (ciberseguridad_docs), para que CVEs y PDFs se recuperen juntos en
la búsqueda híbrida.

Diseño: docs/indexing_design_capa1.md y docs/ingestion_nvd_setup.md (sección 6).
- content  = solo la descripción (en inglés) del CVE → es lo único que se embebe.
- meta     = cve_id, cvss, fechas, vendor/producto, cwe_ids, references, source_type
             → filtrable en pgvector, NO se vectoriza.
- id       = determinístico (hash de cve_id) → reindexar actualiza en vez de duplicar.
- skip     = si el CVE ya está en el store con el mismo last_modified, no se re-embebe.

Conexión configurable por variables de entorno (default = puertos que el
docker-compose del proyecto expone al host, para correr el script en local):
    PGVECTOR_HOST (localhost)   PGVECTOR_PORT (5433)   PGVECTOR_DB (pgvdb)
    PGVECTOR_USR  (avdbuser)    PGVECTOR_PASS (avdbpass)
    OLLAMA_URL    (http://localhost:11434)   EMBEDDING_MODEL (bge-m3)

Uso:
    python src/ingestion/index_nvd.py            # indexa lo nuevo/modificado
    python src/ingestion/index_nvd.py --dry-run  # parsea y reporta, sin embeber ni escribir
    python src/ingestion/index_nvd.py --limit 100  # solo los primeros N CVEs (prueba rápida)
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

# La consola de Windows usa cp1252 por defecto y crashea con caracteres fuera
# de ese set (flechas, y texto no-latino en las descripciones de CVE).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from haystack import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack.utils import Secret

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "nvd"

# Debe coincidir con pipeline_ciberseguridad.py para compartir tabla/índice.
DB_TABLE = "ciberseguridad_docs"
KEYWORD_INDEX = "ciberseguridad_keyword_index"
EMBEDDING_DIMENSION = 1024

CWE_RE = re.compile(r"^CWE-\d+$")


# ---------------------------------------------------------------------------
# Parsing de un CVE crudo -> dict normalizado
# ---------------------------------------------------------------------------

def _english_description(cve: dict) -> str:
    """Descripción en inglés (lo único que se embebe). Fallback: la primera que haya."""
    descriptions = cve.get("descriptions", [])
    for d in descriptions:
        if d.get("lang") == "en":
            return d.get("value", "")
    return descriptions[0].get("value", "") if descriptions else ""


def _cvss(metrics: dict) -> dict:
    """Extrae score/vector/severity. Prioriza v3.1 > v3.0 para 'v3', guarda v2 aparte.

    Cada métrica en el JSON de NVD es una LISTA; tomamos el primer elemento
    (habitualmente el 'Primary' de nvd@nist.gov).
    """
    out = {
        "cvss_v3_score": None, "cvss_v3_vector": None,
        "cvss_v2_score": None, "cvss_v2_vector": None,
        "severity": None,
    }
    for key in ("cvssMetricV31", "cvssMetricV30"):
        entries = metrics.get(key)
        if entries:
            data = entries[0].get("cvssData", {})
            out["cvss_v3_score"] = data.get("baseScore")
            out["cvss_v3_vector"] = data.get("vectorString")
            out["severity"] = data.get("baseSeverity") or entries[0].get("baseSeverity")
            break
    entries_v2 = metrics.get("cvssMetricV2")
    if entries_v2:
        data = entries_v2[0].get("cvssData", {})
        out["cvss_v2_score"] = data.get("baseScore")
        out["cvss_v2_vector"] = data.get("vectorString")
        if out["severity"] is None:
            out["severity"] = entries_v2[0].get("baseSeverity")
    return out


def _vendors_products(cve: dict) -> tuple[list[str], list[str]]:
    """Vendors y productos únicos, parseados de los CPE de configurations.

    CPE 2.3 = cpe:2.3:<part>:<vendor>:<product>:<version>:...  (índices 3 y 4).
    """
    vendors, products = set(), set()
    for config in cve.get("configurations", []):
        for node in config.get("nodes", []):
            for match in node.get("cpeMatch", []):
                parts = match.get("criteria", "").split(":")
                if len(parts) > 4:
                    vendor, product = parts[3], parts[4]
                    if vendor and vendor != "*":
                        vendors.add(vendor)
                    if product and product != "*":
                        products.add(product)
    return sorted(vendors), sorted(products)


def _cwe_ids(cve: dict) -> list[str]:
    ids = set()
    for weakness in cve.get("weaknesses", []):
        for desc in weakness.get("description", []):
            value = desc.get("value", "")
            if CWE_RE.match(value):
                ids.add(value)
    return sorted(ids)


def cve_to_document(cve: dict) -> Document | None:
    """Convierte un objeto 'cve' crudo de NVD en un Document de Haystack.

    Devuelve None si el CVE no tiene descripción utilizable (nada que embeber).
    """
    cve_id = cve.get("id")
    description = _english_description(cve)
    if not cve_id or not description.strip():
        return None

    vendors, products = _vendors_products(cve)
    meta = {
        "source_type": "nvd_cve",
        "cve_id": cve_id,
        "published_date": cve.get("published"),
        "last_modified": cve.get("lastModified"),
        "vendors": vendors,
        "products": products,
        "cwe_ids": _cwe_ids(cve),
        "references": [r.get("url") for r in cve.get("references", []) if r.get("url")],
        **_cvss(cve.get("metrics", {})),
    }
    # id determinístico: reindexar el mismo CVE actualiza la fila, no la duplica.
    doc_id = hashlib.sha256(cve_id.encode()).hexdigest()
    return Document(id=doc_id, content=description, meta=meta)


# ---------------------------------------------------------------------------
# Lectura de los JSON crudos
# ---------------------------------------------------------------------------

def load_cves_from_raw(limit: int | None = None) -> list[dict]:
    """Junta todos los objetos 'cve' de todos los cves_page_*.json bajo data/raw/nvd/.

    Si un CVE aparece en más de una corrida, gana el de lastModified más reciente.
    """
    by_id: dict[str, dict] = {}
    page_files = sorted(RAW_DIR.glob("*/cves_page_*.json"))
    if not page_files:
        raise FileNotFoundError(
            f"No hay JSON de CVEs en {RAW_DIR}. Corré primero src/ingestion/fetch_nvd.py"
        )
    for page_file in page_files:
        data = json.loads(page_file.read_text(encoding="utf-8"))
        for entry in data.get("vulnerabilities", []):
            cve = entry.get("cve", {})
            cve_id = cve.get("id")
            if not cve_id:
                continue
            prev = by_id.get(cve_id)
            if prev is None or cve.get("lastModified", "") >= prev.get("lastModified", ""):
                by_id[cve_id] = cve
    cves = list(by_id.values())
    return cves[:limit] if limit else cves


# ---------------------------------------------------------------------------
# Store / conexión
# ---------------------------------------------------------------------------

def _connection_string() -> str:
    host = os.environ.get("PGVECTOR_HOST", "localhost")
    port = os.environ.get("PGVECTOR_PORT", "5433")
    db = os.environ.get("PGVECTOR_DB", "pgvdb")
    usr = os.environ.get("PGVECTOR_USR", "avdbuser")
    pwd = os.environ.get("PGVECTOR_PASS", "avdbpass")
    return f"postgresql://{usr}:{pwd}@{host}:{port}/{db}"


def _get_store() -> PgvectorDocumentStore:
    return PgvectorDocumentStore(
        connection_string=Secret.from_token(_connection_string()),
        embedding_dimension=EMBEDDING_DIMENSION,
        table_name=DB_TABLE,
        keyword_index_name=KEYWORD_INDEX,
    )


def _existing_last_modified(store: PgvectorDocumentStore) -> dict[str, str]:
    """id de documento -> last_modified de los CVEs ya indexados (para saltear no-modificados)."""
    existing = {}
    for doc in store.filter_documents():
        if doc.meta.get("source_type") == "nvd_cve":
            existing[doc.id] = doc.meta.get("last_modified")
    return existing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa CVEs de data/raw/nvd/ en pgvector")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parsea y reporta, sin conectar a Ollama/pgvector")
    parser.add_argument("--limit", type=int, default=None,
                        help="Procesar solo los primeros N CVEs (prueba rápida)")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / "infrastructure" / ".env")

    print(f"[index] Leyendo CVEs crudos de {RAW_DIR} ...")
    cves = load_cves_from_raw(limit=args.limit)
    documents = [doc for cve in cves if (doc := cve_to_document(cve)) is not None]
    print(f"[index] {len(cves)} CVEs leidos -> {len(documents)} Documents armados "
          f"({len(cves) - len(documents)} descartados por falta de descripción)")

    if args.dry_run:
        for doc in documents[:3]:
            print("-" * 60)
            print(f"id={doc.id}  cve_id={doc.meta['cve_id']}")
            print(f"cvss_v3={doc.meta['cvss_v3_score']} severity={doc.meta['severity']} "
                  f"vendors={doc.meta['vendors'][:3]} cwe={doc.meta['cwe_ids']}")
            print(f"content: {doc.content[:200]}...")
        print("-" * 60)
        print("[index] --dry-run: no se embebió ni escribió nada.")
        return

    store = _get_store()

    existing = _existing_last_modified(store)
    new_or_changed = [
        doc for doc in documents
        if existing.get(doc.id) != doc.meta.get("last_modified")
    ]
    skipped = len(documents) - len(new_or_changed)
    print(f"[index] {skipped} sin cambios (saltados), {len(new_or_changed)} a (re)indexar")

    if not new_or_changed:
        print("[index] Nada nuevo para indexar.")
        return

    print(f"[index] Embebiendo {len(new_or_changed)} CVEs con Ollama...")
    embedder = OllamaDocumentEmbedder(
        model=os.environ.get("EMBEDDING_MODEL", "bge-m3"),
        url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        batch_size=32,
    )
    embedded = embedder.run(new_or_changed)["documents"]
    store.write_documents(embedded, policy=DuplicatePolicy.OVERWRITE)
    print(f"[index] Listo: {len(embedded)} CVEs indexados en '{DB_TABLE}'.")


if __name__ == "__main__":
    main()
