"""Métricas del eval harness.

Tier 1 (retrieval): determinístico, sin LLM. El ground truth es una lista de doc
ids esperados (ids determinísticos del store: cwe-<n>, sha256(cve_id)).

Ver docs/eval/eval_harness.md.
"""
from __future__ import annotations

import math
from statistics import mean
from typing import Iterable, Sequence


# ── Tier 1: retrieval ───────────────────────────────────────────────────────

def hit_at_k(retrieved_ids: Sequence[str], expected_ids: Iterable[str]) -> float:
    """1.0 si al menos un id esperado aparece entre los recuperados, si no 0.0."""
    return 1.0 if set(expected_ids) & set(retrieved_ids) else 0.0


def recall_at_k(retrieved_ids: Sequence[str], expected_ids: Iterable[str]) -> float:
    """Fracción de ids esperados presentes entre los recuperados."""
    exp = set(expected_ids)
    if not exp:
        raise ValueError("recall_at_k no aplica cuando expected_doc_ids está vacío")
    return len(exp & set(retrieved_ids)) / len(exp)


def reciprocal_rank(retrieved_ids: Sequence[str], expected_ids: Iterable[str]) -> float:
    """1/posición (1-indexada) del primer id esperado en el orden recuperado; 0 si ninguno."""
    exp = set(expected_ids)
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in exp:
            return 1.0 / i
    return 0.0


def aggregate_retrieval(per_question: list[dict]) -> dict:
    """Promedia (macro) las métricas de retrieval sobre las preguntas que tienen
    ground truth (expected_doc_ids no vacío). Las negativas se ignoran acá.
    """
    scored = [q for q in per_question if q.get("recall") is not None]
    if not scored:
        return {"recall_at_k": None, "hit_rate": None, "mrr": None, "n": 0}
    return {
        "recall_at_k": round(mean(q["recall"] for q in scored), 4),
        "hit_rate":    round(mean(q["hit"] for q in scored), 4),
        "mrr":         round(mean(q["rr"] for q in scored), 4),
        "n":           len(scored),
    }


def aggregate_by_category(per_question: list[dict]) -> dict[str, dict]:
    """Igual que aggregate_retrieval pero desglosado por category."""
    cats: dict[str, list[dict]] = {}
    for q in per_question:
        cats.setdefault(q["category"], []).append(q)
    return {cat: aggregate_retrieval(qs) for cat, qs in sorted(cats.items())}


# ── Tier 1b: retrieval a nivel de fuente (agnóstico a la estrategia de chunking) ─
#
# El id de un chunk splittable (PDF/MD/DOCX/TXT) es un hash de su contenido: cambia
# con cada estrategia de chunking, así que no sirve de ground truth estable para
# COMPARAR estrategias entre sí. `meta.source` (el nombre del archivo original) sí
# es estable — se preserva igual sea cual sea el splitter. Ver docs/data_splitting.md.

def source_hit_at_k(retrieved_sources: Sequence[str], expected_sources: Iterable[str]) -> float:
    """1.0 si al menos una fuente esperada aparece entre las recuperadas, si no 0.0."""
    return 1.0 if set(expected_sources) & set(retrieved_sources) else 0.0


def source_recall_at_k(retrieved_sources: Sequence[str], expected_sources: Iterable[str]) -> float:
    """Fracción de fuentes esperadas presentes entre las recuperadas."""
    exp = set(expected_sources)
    if not exp:
        raise ValueError("source_recall_at_k no aplica cuando expected_sources está vacío")
    return len(exp & set(retrieved_sources)) / len(exp)


def source_reciprocal_rank(retrieved_sources: Sequence[str], expected_sources: Iterable[str]) -> float:
    """1/posición (1-indexada) de la primera fuente esperada; 0 si ninguna aparece.

    Complementa a source_recall_at_k, que con 1 fuente esperada por pregunta sólo puede
    valer 0 ó 1 y empata estrategias que recuperan la guía correcta en distinta posición.
    Con ranker_top_k=4, llegar 1º o 4º al prompt no es lo mismo: el orden es justamente
    lo que el reranker decide y lo que cambia el contexto que ve el LLM.
    """
    exp = set(expected_sources)
    for i, source in enumerate(retrieved_sources, start=1):
        if source in exp:
            return 1.0 / i
    return 0.0


def aggregate_source_retrieval(per_question: list[dict]) -> dict:
    """Promedia (macro) recall/hit/mrr de fuente sobre las preguntas con expected_sources."""
    scored = [q for q in per_question if q.get("source_recall") is not None]
    if not scored:
        return {"source_recall_at_k": None, "source_hit_rate": None,
                "source_mrr": None, "n": 0}
    con_rr = [q for q in scored if q.get("source_rr") is not None]
    return {
        "source_recall_at_k": round(mean(q["source_recall"] for q in scored), 4),
        "source_hit_rate":    round(mean(q["source_hit"] for q in scored), 4),
        "source_mrr":         round(mean(q["source_rr"] for q in con_rr), 4) if con_rr else None,
        "n":                  len(scored),
    }


# ── Tier 2: similitud de respuesta (SAS) ────────────────────────────────────

def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Coseno entre dos vectores de embedding. 0.0 si alguno es nulo."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def aggregate_sas(per_question: list[dict]) -> dict:
    """Promedia el SAS sobre las preguntas que lo tienen calculado."""
    scored = [q for q in per_question if q.get("sas") is not None]
    if not scored:
        return {"sas": None, "n": 0}
    return {"sas": round(mean(q["sas"] for q in scored), 4), "n": len(scored)}


# ── Abstención en preguntas negativas (H6 de docs/eval/mejoras_harness.md) ───
#
# Una pregunta con `expect_refusal: true` en el dataset es una trampa: el RAG NO
# tiene ese dato y lo correcto es que lo diga. Medirla con SAS no sirve —hay
# infinitas formas válidas de decir "no sé" y la `reference_answer` es sólo una,
# así que el SAS premia la que se parezca a esa redacción (hallazgo 18)—. El
# veredicto correcto es binario y tiene dos partes:
#
#     correct = se abstuvo  AND  no inventó identificadores
#
# Las dos son independientes: un modelo puede decir "no tengo información" y en
# la misma respuesta soltar tres CWE inventados.

# Guiones tipográficos → ASCII. Los LLM escriben `CVE‑2021‑44228` con U+2011 y el
# regex de ids del pipeline sólo entiende `-`: sin esto, un id inventado con
# guión "bonito" es invisible. Verificado (2026-08-18): `_extract_vuln_ids`
# devuelve `[]` sobre un texto con U+2011, y las respuestas de gpt-oss los usan.
_GUIONES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212"), "-")


def normalizar_guiones(texto: str) -> str:
    """Unifica los guiones a ASCII. Aplicar SIEMPRE antes de extraer ids."""
    return (texto or "").translate(_GUIONES)


# Marcadores de abstención sacados de las RESPUESTAS REALES del baseline
# (corrida 20260818T202351Z), no inventados de memoria — que es justo el error
# de la versión anterior: su lista no matcheaba ninguna de las abstenciones que
# el modelo produce de verdad, así que la métrica daba 0 siempre (hallazgo 19).
#
# `"no sé"` va CON tilde a propósito: sin ella matchearía el "no se" reflexivo
# ("no se recomienda", "no se debe") que aparece en respuestas normales.
REFUSAL_MARKERS = (
    "no lo sé",
    "no sé",
    "no dispongo de información",
    "no dispongo de datos",
    "no tengo información",
    "no tengo datos",
    "no hay información",
    "no se encuentra en el contexto",
    "no está en el contexto",
    "fuera del alcance",
    "no puedo responder",
    "no tengo suficiente información",
    "no es posible responder",
    "no puedo proporcionar una respuesta",
    "no hay información suficiente",
    "no puedo determinar",
    "no puedo contestar",
)


def is_refusal(answer: str) -> bool:
    """¿La respuesta es una abstención?

    Detección por marcadores, no por juez: tiene que ser determinística y gratis
    para poder correr en cada eval. La contra conocida es que una abstención
    redactada de una forma nueva no se detecta — si aparece, se agrega el
    marcador acá, tomándolo de la respuesta real.
    """
    if not answer:
        return False
    low = normalizar_guiones(str(answer)).lower()
    return any(marker in low for marker in REFUSAL_MARKERS)


def fabricated_ids(answer_ids: Iterable[str], question_ids: Iterable[str]) -> list[str]:
    """Ids que aparecen en la respuesta y NO venían en la pregunta.

    Restar los de la pregunta es lo que evita el falso positivo obvio: si te
    preguntan por CVE-2021-44228, repetirlo en la respuesta no es inventar nada.

    Recibe listas de ids ya extraídas (`rag._extract_vuln_ids` sobre el texto
    normalizado) en vez de los textos: así `metrics.py` no depende del pipeline.
    """
    de_la_pregunta = set(question_ids)
    vistos, out = set(), []
    for i in answer_ids:
        if i not in de_la_pregunta and i not in vistos:
            vistos.add(i)
            out.append(i)
    return out


def aggregate_abstention(per_question: list[dict]) -> dict:
    """Tasa de abstención correcta sobre las preguntas con `expect_refusal`.

    Devuelve `{"n": 0, "rate": None}` si el dataset no tiene ninguna: sin
    preguntas trampa no hay nada que promediar, y un 0.0 se leería como "falla
    siempre" en vez de "no se midió".
    """
    negativas = [q for q in per_question if q.get("expect_refusal")]
    if not negativas:
        return {"n": 0, "rate": None}
    ok = sum(1 for q in negativas if q.get("correct_rejection"))
    return {"n": len(negativas), "rate": ok / len(negativas)}
