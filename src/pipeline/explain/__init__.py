"""Explicación en lenguaje natural de los hallazgos del escaneo de dependencias.

A diferencia de `deps/` —que es lógica pura—, este paquete toca la base y el LLM. Los
recibe **inyectados** (`store`, `generate`), así que sigue siendo testeable sin levantar
nada. Ver docs/explicacion_hallazgos.md.
"""

from .lookup import ContextDoc, ContextIndex, cwe_key, fetch_context
from .prompt import FindingContext, build_context, build_prompt
from .summarize import Explanation, explain_findings

__all__ = [
    "ContextDoc",
    "ContextIndex",
    "cwe_key",
    "fetch_context",
    "FindingContext",
    "build_context",
    "build_prompt",
    "Explanation",
    "explain_findings",
]
