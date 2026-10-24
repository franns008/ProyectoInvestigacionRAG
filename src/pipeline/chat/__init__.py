"""Conversación sobre los hallazgos adjuntos: contexto del escaneo e historial del chat.

Lógica pura: no importa Haystack ni toca la base, así que se testea con `pytest` sin
levantar nada, igual que `deps/` y `explain/`. La usa `pipeline_ciberseguridad.py`.
"""

from .context import (
    extract_findings_context,
    finding_vuln_ids,
    findings_vuln_ids,
    format_finding,
    format_findings,
    select_history,
)
from .vuln_ids import extract_vuln_ids, resolve_vuln_ids

__all__ = [
    "extract_findings_context",
    "finding_vuln_ids",
    "findings_vuln_ids",
    "format_finding",
    "format_findings",
    "select_history",
    "extract_vuln_ids",
    "resolve_vuln_ids",
]
