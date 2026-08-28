"""
Priorización por explotabilidad real: qué arreglar primero.

resolver.py responde *qué* te afecta; este módulo responde *en qué orden*. La
distinción es el hallazgo central de docs/escaneo_dependencias.md: sobre un
requirements.txt real de 10 dependencias, filtrar por severidad alta deja 75 de 144
hallazgos (52%) — o sea, no filtra nada. Ordenar por explotabilidad deja 1 en el
catálogo de CISA KEV.

Son preguntas distintas:
- **CVSS** = qué tan grave sería si te la explotan.
- **EPSS** = qué probabilidad hay de que se explote en los próximos 30 días.
- **CISA KEV** = se está explotando, confirmado, hoy.

El orden es KEV > EPSS > CVSS. Ningún score lo escribe un LLM: todos salen de acá.

OSV publica el *vector* CVSS, no el número, así que el base score se calcula. Se
implementa CVSS 3.1 (especificación FIRST); los advisories que sólo traen vector v4
(7,3% del dump) quedan con `cvss_score = None` — el score numérico llega después por
enriquecimiento con NVD, no se inventa.
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import math
from pathlib import Path
from typing import Iterable, Sequence

from .resolver import Vulnerability

logger = logging.getLogger("DepsScan")

# Pesos de las métricas base de CVSS 3.1 (tabla 8.4 de la especificación de FIRST).
_WEIGHTS = {
    "AV": {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.20},
    "AC": {"L": 0.77, "H": 0.44},
    "UI": {"N": 0.85, "R": 0.62},
    "C":  {"H": 0.56, "L": 0.22, "N": 0.00},
    "I":  {"H": 0.56, "L": 0.22, "N": 0.00},
    "A":  {"H": 0.56, "L": 0.22, "N": 0.00},
}
# Privileges Required cambia de peso si el alcance (Scope) cambia.
_PR = {
    False: {"N": 0.85, "L": 0.62, "H": 0.27},
    True:  {"N": 0.85, "L": 0.68, "H": 0.50},
}
_REQUIRED_METRICS = ("AV", "AC", "PR", "UI", "S", "C", "I", "A")


def _roundup(value: float) -> float:
    """Roundup() de la especificación CVSS 3.1.

    No es un `round()` común: redondea SIEMPRE hacia arriba al siguiente decimal, y se
    define sobre enteros para no arrastrar el error de coma flotante que haría que
    8.799999 quedara en 8.8 o en 8.9 según el humor del intérprete.
    """
    scaled = int(round(value * 100_000))
    if scaled % 10_000 == 0:
        return scaled / 100_000.0
    return (math.floor(scaled / 10_000) + 1) / 10.0


def cvss31_base_score(vector: str | None) -> float | None:
    """Base score CVSS 3.1 a partir del vector. None si el vector no es v3 o está roto.

    >>> cvss31_base_score("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H")
    10.0
    """
    if not vector or not vector.startswith("CVSS:3"):
        return None

    try:
        metrics = dict(part.split(":", 1) for part in vector.split("/")[1:] if ":" in part)
    except ValueError:
        return None
    if any(m not in metrics for m in _REQUIRED_METRICS):
        return None

    try:
        scope_changed = metrics["S"] == "C"
        privileges = _PR[scope_changed][metrics["PR"]]
        conf, integ, avail = (_WEIGHTS[m][metrics[m]] for m in ("C", "I", "A"))
        attack_vector = _WEIGHTS["AV"][metrics["AV"]]
        complexity = _WEIGHTS["AC"][metrics["AC"]]
        interaction = _WEIGHTS["UI"][metrics["UI"]]
    except KeyError:
        return None

    iss = 1 - ((1 - conf) * (1 - integ) * (1 - avail))
    if scope_changed:
        impact = 7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15
    else:
        impact = 6.42 * iss
    if impact <= 0:
        return 0.0

    exploitability = 8.22 * attack_vector * complexity * privileges * interaction
    base = min((1.08 if scope_changed else 1.0) * (impact + exploitability), 10.0)
    return _roundup(base)


# ======================================================================
# Fuentes de explotabilidad
# ======================================================================
def load_epss(path: str | Path) -> dict[str, float]:
    """CSV de EPSS (FIRST) -> `CVE -> probabilidad de explotación`.

    El archivo trae un par de líneas de metadata con '#' antes del encabezado.
    """
    scores: dict[str, float] = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 2 or not row[0].startswith("CVE-"):
                continue
            try:
                scores[row[0]] = float(row[1])
            except ValueError:
                continue
    return scores


def load_kev(path: str | Path) -> set[str]:
    """Catálogo CISA KEV -> conjunto de CVE con explotación confirmada."""
    with open(path, encoding="utf-8") as handle:
        catalog = json.load(handle)
    return {
        entry["cveID"]
        for entry in catalog.get("vulnerabilities", [])
        if entry.get("cveID")
    }


# ======================================================================
# Priorización
# ======================================================================
def prioritize(
    vulnerabilities: Iterable[Vulnerability],
    epss: dict[str, float] | None = None,
    kev: set[str] | None = None,
) -> list[Vulnerability]:
    """Completa `cvss_score`, `epss` y `kev`, y devuelve la lista ordenada.

    Orden: primero lo que CISA confirma que se explota, después por probabilidad de
    exploit descendente, y CVSS como desempate. Un hallazgo sin score CVSS calculable
    no se promueve por las dudas: queda último dentro de su tramo.

    Muta los objetos recibidos (les estampa los scores) además de devolverlos
    ordenados; son los mismos objetos, no copias.
    """
    epss = epss or {}
    kev = kev or set()

    enriched = list(vulnerabilities)
    for vulnerability in enriched:
        if vulnerability.cvss_score is None:
            vulnerability.cvss_score = cvss31_base_score(vulnerability.cvss_vector)
        if vulnerability.cve:
            vulnerability.epss = epss.get(vulnerability.cve, 0.0)
            vulnerability.kev = vulnerability.cve in kev

    enriched.sort(
        key=lambda v: (
            not v.kev,                    # KEV primero
            -v.epss,                      # después, probabilidad de exploit
            -(v.cvss_score or 0.0),       # CVSS desempata
            v.package,                    # y el identificador hace el orden estable
            v.identifier,
        )
    )
    return enriched


def funnel(vulnerabilities: Sequence[Vulnerability]) -> dict[str, int]:
    """El embudo de la demo: cuántos quedan con cada criterio de filtrado.

    Es lo que hace visible que filtrar por severidad no filtra nada. Los umbrales
    (7.0 de CVSS, 10% de EPSS) son los de docs/escaneo_dependencias.md.
    """
    return {
        "total": len(vulnerabilities),
        "paquetes": len({v.package for v in vulnerabilities}),
        "cvss_alto": sum(1 for v in vulnerabilities if (v.cvss_score or 0.0) >= 7.0),
        "epss_alto": sum(1 for v in vulnerabilities if v.epss >= 0.10),
        "kev": sum(1 for v in vulnerabilities if v.kev),
        "sin_cvss": sum(1 for v in vulnerabilities if v.cvss_score is None),
        "sin_cve": sum(1 for v in vulnerabilities if v.cve is None),
    }
