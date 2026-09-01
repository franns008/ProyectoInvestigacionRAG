"""
Prueba standalone de la etapa determinística.

Recorre requirements.txt, consulta OSV.dev PAQUETE POR PAQUETE (sin batch,
sin RAG, sin LLM) y muestra las vulnerabilidades encontradas para cada uno.

Uso:
    python check_requirements.py [ruta_a_requirements.txt]

Si no se pasa ruta, busca "requirements.txt" en el directorio actual.

Dependencias: requests
"""

import re
import sys
import time
import requests

OSV_QUERY_URL = "https://api.osv.dev/v1/query"


def parse_requirements(path):
    """Extrae (nombre, version) de las líneas con pin exacto (==)."""
    packages = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            m = re.match(r"^([A-Za-z0-9_.\-]+)\s*==\s*([A-Za-z0-9_.\-]+)", line)
            if m:
                packages.append((m.group(1), m.group(2)))
            else:
                print(f"  (omitido, sin pin exacto: {line})")
    return packages


def query_osv(name, version):
    """Consulta OSV.dev para UN paquete a la vez (POST /v1/query)."""
    payload = {"package": {"name": name, "ecosystem": "PyPI"}, "version": version}
    resp = requests.post(OSV_QUERY_URL, json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json().get("vulns", [])


def check_requirements(path):
    packages = parse_requirements(path)
    print(f"\nPaquetes a revisar: {len(packages)}\n")

    report = []
    for name, version in packages:
        print(f"Consultando {name}=={version} ...")
        vulns = query_osv(name, version)
        osv_ids = [(v["id"], v.get("severity", [])) for v in vulns]
        cve_ids_severity = sorted({
            a for v in vulns for a in v.get("aliases", []) if a.startswith("CVE-")

        })

        if vulns:
            severity_strs = []
            for id, severity in osv_ids:
                score = severity[0].get('score', 'N/A') if severity else 'N/A'
                severity_strs.append(f'{id} Score:{score}')
            print(f"  VULNERABLE - {len(vulns)} hallazgo(s): {', '.join(severity_strs)}")
            if cve_ids_severity:
                print(f"  CVE asociados: {', '.join(cve_ids_severity)}")
        else:
            print("  sin vulnerabilidades conocidas")

        report.append({
            "package": name,
            "version": version,
            "osv_ids": osv_ids,
            "cve_ids": cve_ids_severity,
        })
        time.sleep(0.1)  # backoff suave entre requests, uno por uno

    return report


if __name__ == "__main__":
    ruta = sys.argv[1] if len(sys.argv) > 1 else "requirements.txt"
    resultado = check_requirements(ruta)

    vulnerables = [r for r in resultado if r["osv_ids"]]
    print(f"\nResumen: {len(vulnerables)} de {len(resultado)} paquetes con vulnerabilidades conocidas.")
