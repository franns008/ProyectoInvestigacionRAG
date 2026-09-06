"""
Validación de la entrada del pipeline de explicaciones.

Vive acá y no en `pipeline_dependencias.py` porque es lógica pura: no importa Haystack ni
pydantic, así que se testea con `pytest` sin levantar nada, igual que `deps/`. Del otro
lado del `:9099` hay una extensión de VSCode que tiene que poder mostrarle algo al
usuario, así que una entrada mala devuelve un mensaje legible, nunca una excepción cruda.
"""

from __future__ import annotations

import json

# Tope defensivo: la extensión manda 3, pero la entrada viene de afuera del proceso.
MAX_FINDINGS = 10


def parse_findings(user_message: str) -> list[dict]:
    """Acepta `{"findings": [...]}` o directamente una lista. Lanza ValueError con motivo."""
    try:
        payload = json.loads(user_message)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Este pipeline espera un JSON con los hallazgos del escaneo, no una pregunta "
            f"de chat ({error}). Para preguntar en lenguaje natural, usá el modelo del RAG."
        ) from error

    findings = payload.get("findings") if isinstance(payload, dict) else payload
    if not isinstance(findings, list):
        raise ValueError("El JSON tiene que traer una lista en 'findings'.")
    if not all(isinstance(f, dict) for f in findings):
        raise ValueError("Cada hallazgo tiene que ser un objeto JSON.")

    return findings[:MAX_FINDINGS]
