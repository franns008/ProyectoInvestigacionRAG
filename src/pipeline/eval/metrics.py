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


def aggregate_source_retrieval(per_question: list[dict]) -> dict:
    """Promedia (macro) recall/hit de fuente sobre las preguntas con expected_sources."""
    scored = [q for q in per_question if q.get("source_recall") is not None]
    if not scored:
        return {"source_recall_at_k": None, "source_hit_rate": None, "n": 0}
    return {
        "source_recall_at_k": round(mean(q["source_recall"] for q in scored), 4),
        "source_hit_rate":    round(mean(q["source_hit"] for q in scored), 4),
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
