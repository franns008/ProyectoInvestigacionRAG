"""
Hallazgos adjuntos e historial para el chat de la extensión de VSCode.

El chat de la extensión de VSCode manda los hallazgos que el usuario adjuntó como primer
mensaje, con `role: "system"`: no es algo que haya dicho el usuario, es el tema de la charla.

    [{"role": "system",    "content": '{"findings": [{"vulnerability": "CVE-...", ...}, ...]}'},
     {"role": "user",      "content": "¿qué tan grave es?"},
     {"role": "assistant", "content": "..."},
     {"role": "user",      "content": "¿cómo lo mitigo?"}]   ← el turno actual

También se acepta la forma vieja, un único hallazgo suelto en el `system`
(`'{"vulnerability": "CVE-...", "package": ...}'`), que era lo que mandaba el chat por
hallazgo antes del chat con contexto adjunto.

Cada request trae la conversación entera y el contexto vigente (los guarda el cliente), así
que el pipeline no guarda estado: sólo decide qué parte de `messages` entra al prompt.

Sin mensaje `system` (el chat normal de OpenWebUI) no hay hallazgos y todo sigue como antes.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

# Recorte de summary + details del advisory. Mismo criterio que explain/prompt.py.
MAX_ADVISORY_CHARS = 1200
# Con varios hallazgos adjuntos el presupuesto de advisory se reparte entre ellos, con un
# piso: el prompt entero tiene que entrar en el num_ctx del generador junto al corpus.
MAX_ADVISORY_CHARS_TOTAL = 3000
MIN_ADVISORY_CHARS = 300

_VULN_ID_RE = re.compile(r"^(CVE-\d{4}-\d{4,7}|CWE-\d{1,5})$")


def _message_text(msg: Mapping[str, Any]) -> str:
    """El content puede ser str o una lista de partes (multimodal)."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _is_finding(value: Any) -> bool:
    return isinstance(value, dict) and ("vulnerability" in value or "package" in value)


def _finding_key(finding: Mapping[str, Any]) -> tuple:
    return (
        finding.get("vulnerability") or finding.get("cve"),
        finding.get("package"),
        finding.get("installed_version"),
    )


def extract_findings_context(messages: Sequence[Mapping[str, Any]] | None) -> list[dict]:
    """Los hallazgos adjuntos en los mensajes `system` de la extensión, sin repetidos.

    Acepta `{"findings": [...]}` y también un hallazgo suelto (la forma vieja).
    """
    found: list[dict] = []
    seen: set[tuple] = set()
    for msg in messages or []:
        if msg.get("role") != "system":
            continue
        try:
            payload = json.loads(_message_text(msg))
        except (TypeError, json.JSONDecodeError):
            continue  # un system prompt de texto (p. ej. de OpenWebUI): no es un hallazgo
        if isinstance(payload, dict) and isinstance(payload.get("findings"), list):
            candidates = payload["findings"]
        else:
            candidates = [payload]
        for candidate in candidates:
            if _is_finding(candidate) and _finding_key(candidate) not in seen:
                seen.add(_finding_key(candidate))
                found.append(candidate)
    return found


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


def findings_vuln_ids(findings: Sequence[Mapping[str, Any]]) -> list[str]:
    """La unión de `finding_vuln_ids` de cada hallazgo, en orden de adjunto."""
    out: list[str] = []
    for finding in findings:
        for vid in finding_vuln_ids(finding):
            if vid not in out:
                out.append(vid)
    return out


def format_finding(finding: Mapping[str, Any], max_advisory_chars: int = MAX_ADVISORY_CHARS) -> str:
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
        lines.append(f"- Advisory (OSV): {_trim(advisory, max_advisory_chars)}")

    return "\n".join(lines)


def format_findings(findings: Sequence[Mapping[str, Any]]) -> str:
    """Todos los hallazgos adjuntos, numerados. Uno solo sale igual que `format_finding`."""
    if not findings:
        return ""
    if len(findings) == 1:
        return format_finding(findings[0])
    budget = max(MIN_ADVISORY_CHARS, MAX_ADVISORY_CHARS_TOTAL // len(findings))
    return "\n\n".join(
        f"[{i}]\n{format_finding(finding, budget)}" for i, finding in enumerate(findings, 1)
    )


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
