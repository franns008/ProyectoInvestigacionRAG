"""
Contexto del hallazgo e historial para el chat "Hablar en profundidad".

La extensión de VSCode abre el chat mandando el hallazgo del escaneo como primer mensaje,
con `role: "system"`: no es algo que haya dicho el usuario, es el tema de la charla.

    [{"role": "system",    "content": '{"vulnerability": "CVE-...", "package": ...}'},
     {"role": "user",      "content": "¿qué tan grave es?"},
     {"role": "assistant", "content": "..."},
     {"role": "user",      "content": "¿cómo lo mitigo?"}]   ← el turno actual

Cada request trae la conversación entera (el historial lo guarda el cliente), así que el
pipeline no guarda estado: sólo decide qué parte de `messages` entra al prompt.

Sin mensaje `system` (el chat normal de OpenWebUI) no hay hallazgo y todo sigue como antes.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

# Recorte de summary + details del advisory. Mismo criterio que explain/prompt.py.
MAX_ADVISORY_CHARS = 1200

_VULN_ID_RE = re.compile(r"^(CVE-\d{4}-\d{4,7}|CWE-\d{1,5})$")


def _message_text(msg: Mapping[str, Any]) -> str:
    """El content puede ser str o una lista de partes (multimodal)."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def extract_finding_context(messages: Sequence[Mapping[str, Any]] | None) -> dict | None:
    """El hallazgo del primer mensaje `system` que sea un JSON de la extensión, o None."""
    for msg in messages or []:
        if msg.get("role") != "system":
            continue
        try:
            payload = json.loads(_message_text(msg))
        except (TypeError, json.JSONDecodeError):
            continue  # un system prompt de texto (p. ej. de OpenWebUI): no es un hallazgo
        if isinstance(payload, dict) and ("vulnerability" in payload or "package" in payload):
            return payload
    return None


def finding_vuln_ids(finding: Mapping[str, Any] | None) -> list[str]:
    """CVE y CWE del hallazgo, en la forma canónica que guarda `meta.source`.

    Los GHSA/PYSEC se descartan: el corpus no los tiene como `source`, y sólo le
    quitarían lugar a los que sí existen en el corte por top_k de VulnIdLookup.
    """
    if not finding:
        return []
    candidates: list[Any] = [finding.get("vulnerability"), finding.get("cve")]
    candidates += list(finding.get("aliases") or [])
    candidates += list(finding.get("cwe_ids") or [])
    out: list[str] = []
    for value in candidates:
        vid = str(value or "").strip().upper()
        if _VULN_ID_RE.match(vid) and vid not in out:
            out.append(vid)
    return out


def format_finding(finding: Mapping[str, Any]) -> str:
    """El hallazgo como líneas con etiqueta: un modelo chico las lee mejor que un JSON."""
    lines: list[str] = []

    identifier = finding.get("vulnerability") or finding.get("cve")
    aliases = [a for a in (finding.get("aliases") or []) if a and a != identifier]
    if identifier:
        suffix = f" (aliases: {', '.join(aliases)})" if aliases else ""
        lines.append(f"- Identifier: {identifier}{suffix}")

    package = finding.get("package")
    if package:
        installed = finding.get("installed_version") or "?"
        fixed = finding.get("fixed_version")
        fix = f"fixed in {fixed}" if fixed else "no fixed version published"
        lines.append(f"- Affected package: {package} {installed} ({fix})")

    scores = []
    if finding.get("cvss_score") is not None:
        scores.append(f"CVSS={finding['cvss_score']}")
    if finding.get("epss") is not None:
        scores.append(f"EPSS={finding['epss']}")
    if finding.get("kev") is not None:
        scores.append(f"in CISA KEV: {'yes' if finding['kev'] else 'no'}")
    if scores:
        lines.append(f"- {' | '.join(scores)}")

    cwe_ids = [str(c) for c in (finding.get("cwe_ids") or [])]
    if cwe_ids:
        lines.append(f"- Weakness: {', '.join(cwe_ids)}")

    advisory = "\n".join(t for t in (finding.get("summary"), finding.get("details")) if t).strip()
    if advisory:
        lines.append(f"- Advisory (OSV): {_trim(advisory, MAX_ADVISORY_CHARS)}")

    return "\n".join(lines)


def select_history(
    messages: Sequence[Mapping[str, Any]] | None,
    user_message: str,
    max_messages: int = 6,
    max_chars: int = 600,
) -> list[dict]:
    """Los últimos `max_messages` turnos user/assistant ANTERIORES al actual.

    Se descartan el `system` (va aparte, como hallazgo) y el turno actual, que el
    servidor de Pipelines repite al final de `messages` y ya entra como `question`.
    Cada mensaje se recorta a `max_chars`: lo que más crece son las respuestas viejas.
    """
    turns = [
        {"role": m.get("role"), "content": _message_text(m).strip()}
        for m in (messages or [])
        if m.get("role") in ("user", "assistant")
    ]
    if turns and turns[-1]["role"] == "user" and turns[-1]["content"] == user_message.strip():
        turns = turns[:-1]
    if max_messages <= 0:
        return []
    return [
        {"role": t["role"], "content": _trim(t["content"], max_chars)}
        for t in turns[-max_messages:]
        if t["content"]
    ]


def _trim(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " […]"
