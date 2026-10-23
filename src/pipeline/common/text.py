"""Helpers de texto: contenido de mensajes, recorte, texto de un advisory e id de un hallazgo."""

from typing import Any, Mapping


def message_text(msg: Mapping[str, Any]) -> str:
    """El content puede ser str o una lista de partes (multimodal, formato OpenAI)."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def trim(text: str, limit: int) -> str:
    """Recorta en el último espacio antes de `limit` y marca el corte con `[…]`."""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " […]"


def advisory_text(finding: Mapping[str, Any]) -> str:
    """Summary y details del advisory de OSV, unidos."""
    return "\n".join(t for t in (finding.get("summary"), finding.get("details")) if t).strip()


def finding_identifier(finding: Mapping[str, Any], fallback: str = "?") -> str:
    """El CVE si el advisory lo declara; si no, el primer id de OSV (como `identifierOf` en la extensión)."""
    cve = finding.get("cve")
    if cve:
        return str(cve)
    osv_ids = finding.get("osv_ids") or []
    return str(osv_ids[0]) if osv_ids else fallback
