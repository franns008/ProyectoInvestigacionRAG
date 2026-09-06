"""
Orquestación: hallazgos priorizados → una explicación por tarjeta.

**Una llamada al LLM por hallazgo**, no una sola con los tres adentro. Razones, por orden
de importancia:

- El mapeo hallazgo → explicación queda determinístico: no hay que adivinar qué párrafo
  corresponde a qué CVE.
- No hay contaminación cruzada. Con tres CVE en un mismo prompt el modelo mezcla productos
  y versiones, que es el error más caro posible acá.
- Una falla es una tarjeta sin explicación, no las tres.

El lookup, en cambio, es **una sola tanda para todos** (ver explain/lookup.py): dos queries
en total, no dos por hallazgo.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .lookup import fetch_context
from .prompt import build_context, build_prompt

logger = logging.getLogger("DepsExplain")

DEFAULT_TOP_N = 3

# Lo que el modelo tiene que responder cuando las fuentes no alcanzan (ver PROMPT_TEMPLATE).
NO_CONTEXT = "SIN CONTEXTO"

# Tope duro de la explicación. El prompt pide 2-3 frases; esto es la red por si no obedece.
MAX_EXPLANATION_CHARS = 700

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Explanation:
    """La explicación de un hallazgo, lista para mergear en su tarjeta."""

    identifier: str
    explanation: str
    citations: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "identifier": self.identifier,
            "explanation": self.explanation,
            "citations": list(self.citations),
            "kinds": list(self.kinds),
        }


def finding_identifier(finding: Mapping[str, Any]) -> str:
    """El id con el que la extensión reconoce a qué tarjeta pertenece la explicación."""
    cve = finding.get("cve")
    if cve:
        return str(cve)
    osv_ids: Sequence[str] = finding.get("osv_ids") or []
    return str(osv_ids[0]) if osv_ids else "?"


def explain_findings(
    findings: Sequence[Mapping[str, Any]],
    *,
    generate: Callable[[str], str],
    store: Any = None,
    top_n: int = DEFAULT_TOP_N,
) -> list[Explanation]:
    """Explica los primeros `top_n` hallazgos.

    `generate` y `store` se inyectan: este módulo no sabe de Haystack ni de pgvector, y
    por eso se testea con dobles sin levantar nada.

    Sólo se devuelven los hallazgos que efectivamente tienen explicación. Uno sin contexto
    —o cuya generación falló— simplemente no aparece, y su tarjeta se renderiza como hoy.
    """
    seleccionados = list(findings)[: max(top_n, 0)]
    if not seleccionados:
        return []

    index = fetch_context(
        store,
        cve_ids=[f.get("cve") for f in seleccionados],
        cwe_ids=[c for f in seleccionados for c in (f.get("cwe_ids") or [])],
    )

    explicaciones: list[Explanation] = []
    for finding in seleccionados:
        identifier = finding_identifier(finding)
        context = build_context(finding, index)
        if not context:
            logger.info("%s: sin contexto en ninguna fuente, no se explica", identifier)
            continue

        try:
            raw = generate(build_prompt(finding, context))
        except Exception as error:  # noqa: BLE001 — una tarjeta sin explicación, no un escaneo roto
            logger.warning("%s: la generación falló (%s)", identifier, error)
            continue

        texto = _clean(raw)
        if not texto:
            logger.info("%s: el modelo no devolvió una explicación utilizable", identifier)
            continue

        explicaciones.append(
            Explanation(
                identifier=identifier,
                explanation=texto,
                citations=list(context.citations),
                kinds=list(context.kinds),
            )
        )

    return explicaciones


def _clean(raw: Any) -> str:
    """Normaliza la salida del modelo y descarta las respuestas que no sirven.

    El modelo escribe prosa: se colapsa el espacio en blanco, se sacan comillas o rótulos
    que a veces agrega, y se corta en el último punto antes del tope para no dejar una
    frase por la mitad.
    """
    if not isinstance(raw, str):
        return ""

    texto = _WHITESPACE.sub(" ", raw).strip().strip('"').strip()
    texto = re.sub(r"^(explicaci[óo]n|respuesta)\s*:\s*", "", texto, flags=re.IGNORECASE).strip()

    if not texto or texto.upper().startswith(NO_CONTEXT):
        return ""

    if len(texto) > MAX_EXPLANATION_CHARS:
        recorte = texto[:MAX_EXPLANATION_CHARS]
        corte = recorte.rfind(". ")
        texto = (recorte[: corte + 1] if corte > 0 else recorte.rsplit(" ", 1)[0] + "…").strip()

    return texto
