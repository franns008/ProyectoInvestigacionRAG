"""
Traer del corpus el contenido de un CVE o un CWE **por clave exacta**.

Es un `SELECT` por metadata, no un retrieval: cuando el escaneo de `deps/` termina, el
identificador ya se conoce (`CVE-2023-4863`, `CWE-787`). Usar embeddings + full-text + RRF
+ cross-encoder para encontrar un documento cuyo id exacto ya se tiene es aplicar una
herramienta probabilística a un problema determinístico — y el eval mide `recall_eff`
0.841, o sea que fallaría ~1 de cada 6 veces. Acá es exacto, o es un "no está" honesto.

Ver docs/explicacion_hallazgos.md.

Dos cosas que este módulo NO hace, a propósito:

- **No construye el store.** Lo recibe. Así los tests corren con un doble y sin levantar
  pgvector, igual que los de `deps/`.
- **No hace una query por hallazgo.** Junta los ids de los N hallazgos y hace como mucho
  dos llamadas (una por `source_type`). El join hallazgo ↔ documento se arma en memoria.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

logger = logging.getLogger("DepsExplain")

CVE_SOURCE_TYPE = "nvd_cve"
CWE_SOURCE_TYPE = "cwe_weakness"

_CWE_PREFIX = re.compile(r"^\s*cwe[\s\-_]*", re.IGNORECASE)


def cwe_key(raw: Any) -> int | None:
    """`"CWE-787"` → `787`. Devuelve None si no hay un número adentro.

    **Esta normalización no es cosmética.** El indexador guarda el número pelado como
    entero (`indexing/converters.py`: `"cwe_id": int(cwe_id)`), mientras que OSV entrega
    el string `"CWE-787"`. Un filtro `in ["CWE-787"]` contra una columna que tiene `787`
    no lanza error: devuelve **cero filas en silencio**, y el síntoma es "el corpus no
    tiene ese CWE" cuando sí lo tiene.
    """
    if isinstance(raw, bool):  # bool es subclase de int: no es un id de CWE
        return None
    if isinstance(raw, int):
        return raw
    if not isinstance(raw, str):
        return None
    digits = _CWE_PREFIX.sub("", raw).strip()
    return int(digits) if digits.isdigit() else None


@dataclass(frozen=True)
class ContextDoc:
    """Un documento del corpus, listo para entrar en un prompt."""

    doc_id: str
    kind: str          # CVE_SOURCE_TYPE | CWE_SOURCE_TYPE
    identifier: str    # "CVE-2023-4863" | "CWE-787"
    title: str
    content: str

    @property
    def source_label(self) -> str:
        """Cómo se cita esta fuente en la tarjeta."""
        return f"{self.identifier} ({'NVD' if self.kind == CVE_SOURCE_TYPE else 'MITRE'})"


@dataclass(frozen=True)
class ContextIndex:
    """Lo que el corpus devolvió, indexado para el join en memoria."""

    by_cve: dict[str, ContextDoc] = field(default_factory=dict)
    by_cwe: dict[int, ContextDoc] = field(default_factory=dict)

    def for_finding(self, cve: str | None, cwe_ids: Sequence[Any]) -> list[ContextDoc]:
        """Documentos que aplican a un hallazgo, del más específico al más general.

        Primero el CVE (esta vulnerabilidad concreta), después los CWE (la clase de
        debilidad). Sin repetidos y sin huecos.
        """
        docs: list[ContextDoc] = []
        seen: set[str] = set()

        candidates: list[ContextDoc | None] = [self.by_cve.get(cve) if cve else None]
        candidates.extend(self.by_cwe.get(key) for key in map(cwe_key, cwe_ids or ()) if key is not None)

        for doc in candidates:
            if doc is not None and doc.doc_id not in seen:
                seen.add(doc.doc_id)
                docs.append(doc)
        return docs

    def __bool__(self) -> bool:
        return bool(self.by_cve or self.by_cwe)


def fetch_context(store: Any, cve_ids: Iterable[str], cwe_ids: Iterable[Any]) -> ContextIndex:
    """Trae de una sola vez todos los documentos que necesitan los hallazgos.

    Como mucho dos llamadas al store; ninguna si no hay ids que pedir. Un fallo de la
    base **no rompe el escaneo**: se loguea y se devuelve lo que se haya podido traer,
    porque el contexto tiene fallback (ver explain/prompt.py).
    """
    wanted_cves = sorted({c for c in cve_ids if c})
    wanted_cwes = sorted({k for k in map(cwe_key, cwe_ids) if k is not None})

    by_cve: dict[str, ContextDoc] = {}
    for doc in _query(store, CVE_SOURCE_TYPE, "meta.cve_id", wanted_cves):
        identifier = str(doc.meta.get("cve_id", "")).strip()
        if identifier and identifier not in by_cve:
            by_cve[identifier] = _to_context_doc(doc, CVE_SOURCE_TYPE, identifier, identifier)

    by_cwe: dict[int, ContextDoc] = {}
    for doc in _query(store, CWE_SOURCE_TYPE, "meta.cwe_id", wanted_cwes):
        key = cwe_key(doc.meta.get("cwe_id"))
        if key is not None and key not in by_cwe:
            title = str(doc.meta.get("name") or f"CWE-{key}")
            by_cwe[key] = _to_context_doc(doc, CWE_SOURCE_TYPE, f"CWE-{key}", title)

    logger.info(
        "lookup: CVE %d/%d, CWE %d/%d",
        len(by_cve), len(wanted_cves), len(by_cwe), len(wanted_cwes),
    )
    return ContextIndex(by_cve=by_cve, by_cwe=by_cwe)


def _query(store: Any, source_type: str, field_name: str, values: list) -> list:
    """Un filtro por metadata. Sin store o sin ids, no se llama a nadie."""
    if store is None or not values:
        return []
    filters = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.source_type", "operator": "==", "value": source_type},
            {"field": field_name, "operator": "in", "value": values},
        ],
    }
    try:
        return list(store.filter_documents(filters=filters) or [])
    except Exception as error:  # noqa: BLE001 — la base caída degrada, no rompe
        logger.warning("lookup de %s falló (%s); sigo sin ese contexto", source_type, error)
        return []


def _to_context_doc(doc: Any, kind: str, identifier: str, title: str) -> ContextDoc:
    return ContextDoc(
        doc_id=str(getattr(doc, "id", identifier)),
        kind=kind,
        identifier=identifier,
        title=title,
        content=(getattr(doc, "content", "") or "").strip(),
    )
