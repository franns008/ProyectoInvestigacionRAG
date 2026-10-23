"""Qué modelo de generación y qué ventana de contexto se usan de verdad.

Lo comparten `build_generator()` y los valves del chat, que publican la ventana efectiva
para el medidor de contexto de la extensión: los dos tienen que llegar al mismo número.
Sin Haystack: se testea sin stack.
"""

import json
import os
import re
import urllib.request
from functools import lru_cache
from typing import Any, Optional

# Modelo de generación cuando no hay `LLM_MODEL`: el default del valve `llm_model` de los
# dos pipelines. Vive acá y no en pipeline_ciberseguridad porque importar ese módulo carga
# torch, y pipeline_dependencias lo necesita al definir sus valves.
DEFAULT_OLLAMA_LLM = "qwen2.5:3b-instruct"

_NUM_CTX_RE = re.compile(r"^num_ctx\s+(\d+)\s*$", re.MULTILINE)


def llm_model(valves: Any) -> str:
    """`LLM_MODEL` le gana al valve `llm_model`: se cambia de modelo sin tocar los valves."""
    return os.getenv("LLM_MODEL") or valves.llm_model


def env_num_ctx() -> Optional[int]:
    """`LLM_NUM_CTX`: sin definir (o vacía) es None; 0 es "la del Modelfile"; N fija N."""
    raw = os.getenv("LLM_NUM_CTX", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"LLM_NUM_CTX tiene que ser un entero >= 0, no {raw!r}") from None
    if value < 0:
        raise ValueError(f"LLM_NUM_CTX tiene que ser un entero >= 0, no {raw!r}")
    return value


@lru_cache(maxsize=None)
def modelfile_num_ctx(model: str, ollama_url: str) -> int:
    """El `num_ctx` que fija el Modelfile del modelo, según `/api/show` de Ollama.

    Va cacheado: el Modelfile no cambia mientras corre el servidor, y los valves lo piden
    cada vez que la extensión los lee. Si se recrea el modelo, hay que reiniciar.
    """
    request = urllib.request.Request(
        f"{ollama_url}/api/show",
        data=json.dumps({"model": model}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        parameters = json.load(response).get("parameters") or ""
    match = _NUM_CTX_RE.search(parameters)
    if not match:
        raise ValueError(
            f"LLM_NUM_CTX=0 pide la ventana del Modelfile, pero el de {model!r} no fija "
            "num_ctx. Poné un número en LLM_NUM_CTX o usá un modelo que la fije."
        )
    return int(match.group(1))


def effective_num_ctx(valves: Any, ollama_url: str) -> Optional[int]:
    """La ventana con la que corre el generador: `LLM_NUM_CTX` le gana al valve `num_ctx`.

    None si nadie la fija (los valves de pipeline_dependencias no la traen): entonces
    Ollama usa su default.
    """
    value = env_num_ctx()
    if value is None:
        return getattr(valves, "num_ctx", None) or None
    if value == 0:
        return modelfile_num_ctx(llm_model(valves), ollama_url)
    return value
