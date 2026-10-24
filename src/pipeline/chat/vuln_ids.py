"""IDs de vulnerabilidad (CWE/CVE) en la pregunta y en la conversación, en forma canónica."""

from __future__ import annotations

import re

from common.text import message_text

_CWE_ID_RE = re.compile(r"cwe[\s\-_]?(\d{1,5})", re.IGNORECASE)
_CVE_ID_RE = re.compile(r"cve[\s\-_]?(\d{4})[\s\-_]?(\d{4,7})", re.IGNORECASE)


def extract_vuln_ids(text: str) -> list[str]:
    """CWE y CVE en forma canónica, en orden de aparición: el orden es el peso en
    VulnIdLookup (docs/analisis_keyword_retriever.md)."""
    found: list[tuple[int, str]] = []
    for m in _CWE_ID_RE.finditer(text):
        found.append((m.start(), f"CWE-{int(m.group(1))}"))
    for m in _CVE_ID_RE.finditer(text):
        found.append((m.start(), f"CVE-{m.group(1)}-{m.group(2)}"))
    found.sort(key=lambda t: t[0])
    seen, out = set(), []
    for _, vid in found:
        if vid not in seen:
            seen.add(vid)
            out.append(vid)
    return out


_HISTORY_TURNS = 4   # ~2 intercambios hacia atrás
_MAX_VULN_IDS  = 6


def resolve_vuln_ids(
    user_message: str,
    messages: list[dict] | None = None,
) -> tuple[list[str], str]:
    """IDs del turno actual; si no trae ninguno, los de los últimos turnos, del más
    reciente al más viejo. Devuelve (ids, origen). Ver docs/analisis_keyword_retriever.md."""
    current = extract_vuln_ids(user_message)
    if current:
        return current[:_MAX_VULN_IDS], "turno"
    if not messages:
        return [], "ninguno"

    inherited: list[str] = []
    # El `system` de los hallazgos no cuenta: sus IDs entran aparte (findings_vuln_ids).
    turns = [m for m in messages if m.get("role") in ("user", "assistant")]
    for msg in reversed(turns[-_HISTORY_TURNS:]):
        text = message_text(msg)
        if not text or text.strip() == user_message.strip():
            continue  # messages trae el turno actual al final
        for vid in extract_vuln_ids(text):
            if vid not in inherited:
                inherited.append(vid)
    if not inherited:
        return [], "ninguno"
    return inherited[:_MAX_VULN_IDS], "historial"
