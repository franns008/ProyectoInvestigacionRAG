"""
Extensión de detección de vulnerabilidades - etapa determinística.

Flujo:
  1. Parsea requirements.txt -> lista de (paquete, version)
  2. Consulta OSV.dev en batch (POST /v1/querybatch) -> IDs de vulnerabilidad por paquete
  3. Para cada ID, trae el registro completo (GET /v1/vulns/{id}) -> extrae los CVE-ID (aliases)
  4. Cruza esos CVE-ID contra la base local de CVE/CWE ya indexada (Postgres)

Nada de esto pasa por el RAG ni por el LLM: es consulta estructurada, 100% reproducible.
Dependencias: requests, psycopg2-binary
"""

import re
import time
import requests
import psycopg2
from psycopg2.extras import RealDictCursor

OSV_BATCH_URL = "https://api.osv.dev/v1/querybatch"
OSV_VULN_URL = "https://api.osv.dev/v1/vulns/{}"
DB_DSN = "dbname=cyberrag user=cyberrag host=localhost"  # ajustar a su config real


def parse_requirements(path):
    """Extrae (nombre, version) de un requirements.txt con pines exactos (==)."""
    packages = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            m = re.match(r"^([A-Za-z0-9_.\-]+)\s*==\s*([A-Za-z0-9_.\-]+)", line)
            if m:
                packages.append((m.group(1), m.group(2)))
            # líneas sin pin exacto (>=, ~=, sin versión) quedan afuera del
            # matching determinístico: no hay una única versión que consultar.
            # Si necesitan cubrir ese caso, resuelvan la version instalada
            # real (ej. via `pip freeze` en el entorno) antes de este paso.
    return packages


def query_osv_batch(packages):
    """Batch query: devuelve, por paquete, la lista de IDs de vulnerabilidad (sin detalle)."""
    queries = [
        {"package": {"name": name, "ecosystem": "PyPI"}, "version": version}
        for name, version in packages
    ]
    resp = requests.post(OSV_BATCH_URL, json={"queries": queries}, timeout=30)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return {
        packages[i]: [v["id"] for v in results[i].get("vulns", [])]
        for i in range(len(packages))
    }


def fetch_cve_aliases(vuln_id, _cache={}):
    """Trae el registro completo de una vulnerabilidad OSV y devuelve sus CVE-ID."""
    if vuln_id in _cache:
        return _cache[vuln_id]
    resp = requests.get(OSV_VULN_URL.format(vuln_id), timeout=15)
    resp.raise_for_status()
    aliases = resp.json().get("aliases", [])
    cve_ids = [a for a in aliases if a.startswith("CVE-")]
    _cache[vuln_id] = cve_ids
    time.sleep(0.05)  # backoff suave; OSV no documenta rate limit pero conviene ser prudentes
    return cve_ids


def join_with_local_db(cve_ids, conn):
    """Cruza los CVE-ID contra la base local (ya cargada desde NVD). Ajustar nombre de tabla/columnas."""
    if not cve_ids:
        return {}
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT cve_id, description, cwe_id, cvss_score
            FROM cve_entries
            WHERE cve_id = ANY(%s)
            """,
            (cve_ids,),
        )
        return {row["cve_id"]: row for row in cur.fetchall()}


def scan_requirements(path):
    packages = parse_requirements(path)
    osv_results = query_osv_batch(packages)

    conn = psycopg2.connect(DB_DSN)
    report = []
    for (name, version), vuln_ids in osv_results.items():
        cve_ids = sorted({c for vid in vuln_ids for c in fetch_cve_aliases(vid)})
        local_matches = join_with_local_db(cve_ids, conn)
        report.append({
            "package": name,
            "version": version,
            "cve_ids": cve_ids,
            "en_corpus_local": {c: (c in local_matches) for c in cve_ids},
            "detalle_local": local_matches,
        })
    conn.close()
    return report


if __name__ == "__main__":
    resultado = scan_requirements("requirements.txt")
    for r in resultado:
        estado = "vulnerable" if r["cve_ids"] else "sin vulnerabilidades"
        print(f"{r['package']}=={r['version']}: {estado} ({len(r['cve_ids'])} CVE)")
