"""Escaneo determinístico de dependencias contra advisories de OSV.

Ver docs/escaneo_dependencias.md. El paquete no importa Haystack ni toca la base:
es lógica pura y se testea sin levantar el stack.
"""

from .resolver import (
    Requirement,
    Vulnerability,
    build_index,
    load_osv_zip,
    normalize_name,
    parse_requirements,
    scan,
)
from .prioritize import cvss31_base_score, load_epss, load_kev, prioritize

__all__ = [
    "Requirement",
    "Vulnerability",
    "build_index",
    "load_osv_zip",
    "normalize_name",
    "parse_requirements",
    "scan",
    "cvss31_base_score",
    "load_epss",
    "load_kev",
    "prioritize",
]
