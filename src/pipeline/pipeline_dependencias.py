"""
title: Explicación de dependencias vulnerables
author: Grupo Niños RAGtas
version: 1.0
requirements: haystack-ai, pgvector-haystack, ollama-haystack

Segundo modelo del servidor de Pipelines, al lado del chat. **No escanea nada**: recibe
los hallazgos que la extensión de VSCode ya resolvió de forma determinística y devuelve
una explicación corta por hallazgo.

Ese reparto es deliberado (docs/explicacion_hallazgos.md): el escaneo lo sigue haciendo
la CLI local en la máquina del usuario, así que si este servidor está caído **el escaneo
sigue funcionando** y sólo faltan los párrafos. Además este pipeline no necesita el dump
de OSV, ni parsear requirements, ni la lógica de rangos de versiones.

El camino de datos es un lookup por clave exacta contra pgvector — no hay retrieval, ni
embeddings, ni reranker. Ver explain/lookup.py.

Entrada (en el `content` del mensaje del usuario):

    {"findings": [{"package": "pillow", "installed_version": "5.2.0",
                   "cve": "CVE-2023-4863", "osv_ids": ["GHSA-..."],
                   "cwe_ids": ["CWE-787"], "summary": "...", "details": "..."}]}

Salida (string JSON):

    {"explanations": [{"identifier": "CVE-2023-4863", "explanation": "...",
                       "citations": ["CWE-787 (MITRE)", "GHSA-... (OSV)"],
                       "kinds": ["cwe_weakness", "osv_advisory"]}]}
"""

import json
import logging
from typing import Generator, Iterator, List, Union

from pydantic import BaseModel

from explain.payload import parse_findings

logger = logging.getLogger("DepsExplain")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"))
    logger.addHandler(handler)


class Pipeline:

    class Valves(BaseModel):
        # Mismos nombres que los Valves del chat: build_generator() los lee por atributo.
        llm_model:   str   = "meta-llama/llama-4-scout-17b-16e-instruct"
        max_tokens:  int   = 220    # 2-3 frases; más que esto es el modelo desobedeciendo
        temperature: float = 0.2    # es un resumen de un texto dado, no escritura creativa
        top_n:       int   = 3      # cuántas tarjetas se explican

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
        torch ni pedir GROQ_API_KEY.
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
                {"error": "El generador no está disponible (revisá LLM_PROVIDER y la API key)."},
                ensure_ascii=False,
            )

        from explain import explain_findings

        logger.info("explicando %d hallazgo(s)", len(findings))
        explicaciones = explain_findings(
            findings,
            generate=self._generate,
            store=self.store,
            top_n=self.valves.top_n,
        )
        return json.dumps(
            {"explanations": [e.as_dict() for e in explicaciones]},
            ensure_ascii=False,
        )

    def _generate(self, prompt: str) -> str:
        """Una llamada al LLM. La forma de la respuesta es la misma en Groq y en Ollama."""
        replies = (self.generator.run(prompt=prompt) or {}).get("replies") or []
        return replies[0] if replies else ""

