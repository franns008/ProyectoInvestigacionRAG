"""IDs de vulnerabilidad (CWE/CVE) en la pregunta y en la conversación, en forma canónica."""

from __future__ import annotations

import re

from common.text import message_text

_CWE_ID_RE = re.compile(r"cwe[\s\-_]?(\d{1,5})", re.IGNORECASE)
_CVE_ID_RE = re.compile(r"cve[\s\-_]?(\d{4})[\s\-_]?(\d{4,7})", re.IGNORECASE)


def extract_vuln_ids(text: str) -> list[str]:
    """IDs en ORDEN DE APARICIÓN en el texto, no agrupados por tipo.

    Antes se recorrían todos los CWE y después todos los CVE, así que el orden
    reflejaba el tipo y no la pregunta: "¿qué CVE explota CWE-502?" ponía siempre
    el CWE primero. El orden importa porque decide el desempate y el corte por
    top_k en VulnIdLookup (y, vía RRF, cuánto pesa cada uno).
    """
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
    """IDs del turno actual; si no hay ninguno, los hereda de los últimos turnos.

    El turno actual manda: sólo se mira hacia atrás cuando no trae ningún ID, así
    un cambio explícito de tema nunca queda contaminado. Dentro del historial se
    acumulan del más reciente al más viejo, y ese orden ES el peso: VulnIdLookup
    lo respeta y el RRF le da 1/(k+rango) a cada documento.

    Devuelve (ids, origen); `origen` se loguea para poder auditar el arrastre.
    """
    current = extract_vuln_ids(user_message)
    if current:
        return current[:_MAX_VULN_IDS], "turno"
    if not messages:
        return [], "ninguno"

    inherited: list[str] = []
    for msg in reversed(messages[-_HISTORY_TURNS:]):
        text = message_text(msg)
        if not text or text.strip() == user_message.strip():
            continue  # messages trae el turno actual al final
        for vid in extract_vuln_ids(text):
            if vid not in inherited:
                inherited.append(vid)
    if not inherited:
        return [], "ninguno"
    return inherited[:_MAX_VULN_IDS], "historial"
