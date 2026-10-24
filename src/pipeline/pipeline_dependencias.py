"""
title: Explicación de dependencias vulnerables
author: Grupo Niños RAGtas
version: 1.0
requirements: haystack-ai, pgvector-haystack, ollama-haystack
"""

# Docstring corto a propósito: el servidor lo parsea como frontmatter (docs/arquitectura.md
# §4.5). Segundo modelo del :9099: explica los hallazgos que ya resolvió la extensión; el
# contrato de entrada y salida está en docs/explicacion_hallazgos.md.

import os
import sys

# El servidor no agrega este directorio a sys.path (docs/arquitectura.md §4.5).
_PIPELINES_DIR = os.path.dirname(os.path.abspath(__file__))
if _PIPELINES_DIR not in sys.path:
    sys.path.insert(0, _PIPELINES_DIR)

import json
from typing import Generator, Iterator, List, Union

from pydantic import BaseModel

from common.llm import DEFAULT_OLLAMA_LLM
from common.log import get_logger
from explain import explain_findings
from explain.payload import parse_findings

logger = get_logger("DepsExplain")


class Pipeline:

    class Valves(BaseModel):
        # Mismos nombres que los Valves del chat: build_generator() los lee por atributo.
        llm_model:   str   = DEFAULT_OLLAMA_LLM
        max_tokens:  int   = 220    # 2-3 frases; más que esto es el modelo desobedeciendo
        temperature: float = 0.2    # es un resumen de un texto dado, no escritura creativa

    def __init__(self):
        self.name = "Explicación de dependencias vulnerables"
        self.valves = self.Valves()
        # Se construyen en el arranque, no acá: el import de Haystack es caro y el
        # servidor instancia la clase antes de levantar.
        self.store = None
        self.generator = None

    async def on_startup(self):
        self._build()

    async def on_valves_updated(self):
        logger.info("Valves actualizados — reconstruyendo el generador...")
        self._build()

    def _build(self) -> None:
        """Store y generador, reusados del pipeline de chat.

        `get_document_store()` y `build_generator()` ya son funciones módulo-level, así
        que alcanza con importarlas: el pipeline de chat no se toca. El import es local
        para que este archivo se pueda importar fuera del container (tests) sin arrastrar
        torch.
        """
        try:
            from pipeline_ciberseguridad import build_generator, get_document_store

            self.store = get_document_store()
            self.generator = build_generator(self.valves)
            logger.info("Pipeline de explicaciones listo.")
        except Exception as error:  # noqa: BLE001
            # Sin store se explica igual desde el advisory; sin generador no hay nada que hacer,
            # pero el servidor tiene que levantar para poder decirlo en la respuesta.
            logger.error("No pude inicializar el pipeline de explicaciones: %s", error)

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:
        try:
            findings = parse_findings(user_message)
        except ValueError as error:
            logger.warning("entrada inválida: %s", error)
            return json.dumps({"error": str(error)}, ensure_ascii=False)

        if self.generator is None:
            self._build()
        if self.generator is None:
            return json.dumps(
                {"error": "El generador no está disponible: revisá LLM_MODEL y que el modelo "
                          "esté descargado en Ollama (el detalle está en el log de pipelines)."},
                ensure_ascii=False,
            )

        logger.info("explicando %d hallazgo(s), findings: %s", len(findings), ", ".join(str(f) for f in findings))
        explicaciones = explain_findings(
            findings,
            generate=self._generate,
            store=self.store,
        )
        return json.dumps(
            {"explanations": [e.as_dict() for e in explicaciones]},
            ensure_ascii=False,
        )

    def _generate(self, prompt: str) -> str:
        """Una llamada al LLM, con el prompt como único mensaje del usuario."""
        from haystack.dataclasses import ChatMessage

        replies = (self.generator.run(messages=[ChatMessage.from_user(prompt)]) or {}).get("replies") or []
        return (replies[0].text or "") if replies else ""

