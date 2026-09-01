"""
Extensión de detección de vulnerabilidades - etapa determinística.

Flujo:
  1. Parsea requirements.txt -> lista de (paquete, version)
  2. Consulta OSV.dev en batch (POST /v1/querybatch) -> IDs de vulnerabilidad por paquete
  3. Para cada ID, trae el registro completo (GET /v1/vulns/{id}) -> extrae los CVE-ID (aliases)
  4. Arma, por cada paquete vulnerable, la consulta en lenguaje natural que se le
     pasa al pipeline de RAG ya existente (Haystack) para que retrieve y explique.

Esta etapa NO arma el contexto a mano ni hace join contra la base local: eso lo
resuelve el retriever del RAG cuando recibe la query. Acá solo se determina, de
forma determinística, QUÉ paquete es vulnerable y a QUÉ CVE corresponde.

Dependencias: requests
"""

import re
import time
import requests

OSV_BATCH_URL = "https://api.osv.dev/v1/querybatch"
OSV_VULN_URL = "https://api.osv.dev/v1/vulns/{}"


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


def find_vulnerable_packages(path):
    """Devuelve solo lo mínimo: paquete, versión y CVE-ID(s) encontrados. Nada más."""
    packages = parse_requirements(path)
    osv_results = query_osv_batch(packages)

    vulnerable = []
    for (name, version), vuln_ids in osv_results.items():
        cve_ids = sorted({c for vid in vuln_ids for c in fetch_cve_aliases(vid)})
        if cve_ids:
            vulnerable.append({"package": name, "version": version, "cve_ids": cve_ids})
    return vulnerable


def build_rag_queries(vulnerable_packages):
    """Arma, por cada CVE encontrado, la consulta que se le pasa al RAG existente."""
    queries = []
    for pkg in vulnerable_packages:
        for cve_id in pkg["cve_ids"]:
            text = (
                f"¿Qué es la vulnerabilidad {cve_id} que afecta al paquete "
                f"{pkg['package']} versión {pkg['version']}? Explicá el riesgo "
                f"y cómo mitigarlo."
            )
            queries.append({
                "package": pkg["package"],
                "version": pkg["version"],
                "cve_id": cve_id,
                "query": text,
            })
    return queries


def send_to_rag_pipeline(query_text):
    """Placeholder: reemplazar por la llamada real al Pipeline de Haystack ya armado."""
    raise NotImplementedError("Conectar acá con el pipeline.run(query=...) existente")


if __name__ == "__main__":
    vulnerables = find_vulnerable_packages("requirements.txt")
    rag_queries = build_rag_queries(vulnerables)

    if not rag_queries:
        print("Sin vulnerabilidades detectadas.")

    for q in rag_queries:
        print(f"[{q['package']}=={q['version']}] {q['cve_id']} -> enviando al RAG...")
        # explicacion = send_to_rag_pipeline(q["query"])
        # print(explicacion)
