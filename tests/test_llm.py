"""
Tests de la resolución de modelo y ventana (src/pipeline/common/llm.py). Corren sin stack:
`/api/show` de Ollama se reemplaza por una respuesta fija.
"""

import io
import json
from types import SimpleNamespace

import pytest

from common import llm

URL = "http://ollama:11434"
VALVES = SimpleNamespace(llm_model="qwen2.5:3b-instruct", num_ctx=8192)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_NUM_CTX", raising=False)
    llm.modelfile_num_ctx.cache_clear()
    yield
    llm.modelfile_num_ctx.cache_clear()


@pytest.fixture
def ollama_show(monkeypatch):
    """Hace que `/api/show` devuelva `parameters` y anota qué modelos se pidieron."""
    asked = []

    def install(parameters):
        def urlopen(request, timeout):
            asked.append(json.loads(request.data)["model"])
            return io.BytesIO(json.dumps({"parameters": parameters}).encode())

        monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
        return asked

    return install


# ======================================================================
# llm_model
# ======================================================================
def test_llm_model_sale_del_valve_sin_env():
    assert llm.llm_model(VALVES) == "qwen2.5:3b-instruct"


def test_llm_model_env_le_gana_al_valve(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gemma4-light")
    assert llm.llm_model(VALVES) == "gemma4-light"


# ======================================================================
# env_num_ctx
# ======================================================================
@pytest.mark.parametrize("raw", [None, "", "  "])
def test_env_num_ctx_sin_definir_es_none(monkeypatch, raw):
    if raw is not None:
        monkeypatch.setenv("LLM_NUM_CTX", raw)
    assert llm.env_num_ctx() is None


@pytest.mark.parametrize("raw, value", [("0", 0), ("32768", 32768), (" 4096 ", 4096)])
def test_env_num_ctx_entero(monkeypatch, raw, value):
    monkeypatch.setenv("LLM_NUM_CTX", raw)
    assert llm.env_num_ctx() == value


@pytest.mark.parametrize("raw", ["8k", "-1", "1.5"])
def test_env_num_ctx_invalida_falla(monkeypatch, raw):
    monkeypatch.setenv("LLM_NUM_CTX", raw)
    with pytest.raises(ValueError, match="LLM_NUM_CTX"):
        llm.env_num_ctx()


# ======================================================================
# effective_num_ctx
# ======================================================================
def test_sin_env_usa_el_valve():
    assert llm.effective_num_ctx(VALVES, URL) == 8192


def test_sin_env_ni_valve_es_none():
    valves = SimpleNamespace(llm_model="x")  # los valves de pipeline_dependencias
    assert llm.effective_num_ctx(valves, URL) is None


def test_env_le_gana_al_valve(monkeypatch):
    monkeypatch.setenv("LLM_NUM_CTX", "12288")
    assert llm.effective_num_ctx(VALVES, URL) == 12288


def test_env_cero_toma_el_modelfile_del_modelo_efectivo(monkeypatch, ollama_show):
    asked = ollama_show("top_k                          64\nnum_ctx                        32768\ntemperature 1")
    monkeypatch.setenv("LLM_MODEL", "gemma4-light")
    monkeypatch.setenv("LLM_NUM_CTX", "0")
    assert llm.effective_num_ctx(VALVES, URL) == 32768
    assert asked == ["gemma4-light"]


def test_modelfile_se_pide_una_sola_vez(monkeypatch, ollama_show):
    asked = ollama_show("num_ctx 32768")
    monkeypatch.setenv("LLM_NUM_CTX", "0")
    llm.effective_num_ctx(VALVES, URL)
    llm.effective_num_ctx(VALVES, URL)
    assert asked == ["qwen2.5:3b-instruct"]


@pytest.mark.parametrize("parameters", [None, "", "temperature 1\nnum_ctx_max 4"])
def test_env_cero_sin_num_ctx_en_el_modelfile_falla(monkeypatch, ollama_show, parameters):
    ollama_show(parameters)
    monkeypatch.setenv("LLM_NUM_CTX", "0")
    with pytest.raises(ValueError, match="no fija num_ctx"):
        llm.effective_num_ctx(VALVES, URL)
