"""Conversación sobre un hallazgo: contexto del escaneo e historial del chat.

Lógica pura: no importa Haystack ni toca la base, así que se testea con `pytest` sin
levantar nada, igual que `deps/` y `explain/`. La usa `pipeline_ciberseguridad.py`.
"""

from .context import extract_finding_context, finding_vuln_ids, format_finding, select_history

__all__ = [
    "extract_finding_context",
    "finding_vuln_ids",
    "format_finding",
    "select_history",
]
