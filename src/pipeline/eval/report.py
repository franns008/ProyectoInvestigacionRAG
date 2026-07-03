"""Reporte de delta contra el baseline.

Compara un snapshot de resultados (el de la corrida actual) contra baseline.json
y muestra: delta de métricas globales, delta por categoría y la lista de preguntas
que regresaron. Lo usa run_eval.py al final de cada corrida.

Ver docs/eval_harness.md.
"""
from __future__ import annotations

import json
from pathlib import Path

# Caída de SAS (0..1) a partir de la cual se considera regresión de generación.
SAS_REGRESSION_THRESHOLD = 0.05


def load_snapshot(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def promote(snapshot: dict, baseline_path: Path) -> None:
    baseline_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _arrow(delta: float) -> str:
    if delta > 0:
        return "▲"
    if delta < 0:
        return "▼"
    return "="


def _fmt(cur: float | None, base: float | None) -> str:
    if cur is None and base is None:
        return "   n/a"
    if base is None:
        return f"{cur:.3f} (nuevo)"
    if cur is None:
        return f"n/a (baseline {base:.3f})"
    d = cur - base
    return f"{cur:.3f}  {_arrow(d)}{abs(d):.3f}"


def print_delta(current: dict, baseline: dict | None) -> None:
    print("\n" + "=" * 64)
    if baseline is None:
        print("DELTA vs BASELINE — no hay baseline.json todavía")
        print("=" * 64)
        print("  Promové esta corrida con:  run_eval.py --set-baseline")
        return

    print(f"DELTA vs BASELINE ({baseline.get('timestamp', '?')})")
    print("=" * 64)

    co, bo = current["overall"], baseline["overall"]
    cs = current.get("sas", {}) or {}
    bs = baseline.get("sas", {}) or {}
    print("  Global:")
    print(f"    recall@k : {_fmt(co.get('recall_at_k'), bo.get('recall_at_k'))}")
    print(f"    hit_rate : {_fmt(co.get('hit_rate'), bo.get('hit_rate'))}")
    print(f"    mrr      : {_fmt(co.get('mrr'), bo.get('mrr'))}")
    print(f"    sas      : {_fmt(cs.get('sas'), bs.get('sas'))}")

    # Delta de recall por categoría
    cbc, bbc = current["by_category"], baseline["by_category"]
    cats = sorted(set(cbc) | set(bbc))
    print("\n  recall@k por categoría:")
    for cat in cats:
        cur = (cbc.get(cat) or {}).get("recall_at_k")
        base = (bbc.get(cat) or {}).get("recall_at_k")
        if cur is None and base is None:
            continue
        print(f"    {cat:<16} {_fmt(cur, base)}")

    _print_regressions(current, baseline)


def _print_regressions(current: dict, baseline: dict) -> None:
    cur_by_id = {q["id"]: q for q in current["per_question"]}
    base_by_id = {q["id"]: q for q in baseline["per_question"]}

    retr_reg, sas_reg = [], []
    for qid, cq in cur_by_id.items():
        bq = base_by_id.get(qid)
        if bq is None:
            continue
        # Retrieval: recall bajó
        if cq.get("recall") is not None and bq.get("recall") is not None:
            if cq["recall"] < bq["recall"]:
                retr_reg.append((qid, bq["recall"], cq["recall"]))
        # Generación: SAS bajó más que el umbral
        if cq.get("sas") is not None and bq.get("sas") is not None:
            if bq["sas"] - cq["sas"] > SAS_REGRESSION_THRESHOLD:
                sas_reg.append((qid, bq["sas"], cq["sas"]))

    if not retr_reg and not sas_reg:
        print("\n  Sin regresiones por pregunta. 🎉")
        return
    if retr_reg:
        print("\n  ⚠ Regresiones de RETRIEVAL (recall bajó):")
        for qid, b, c in retr_reg:
            print(f"    {qid:<28} {b:.2f} → {c:.2f}")
    if sas_reg:
        print(f"\n  ⚠ Regresiones de GENERACIÓN (SAS bajó > {SAS_REGRESSION_THRESHOLD}):")
        for qid, b, c in sas_reg:
            print(f"    {qid:<28} {b:.3f} → {c:.3f}")
