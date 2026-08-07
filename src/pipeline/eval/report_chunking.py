#!/usr/bin/env python3
"""Reporte detallado de un experimento de chunking: por cada pregunta, QUÉ documentos
se esperaban y QUÉ documentos usó realmente el RAG, por estrategia.

La tabla comparativa que imprime run_chunking_experiment.py es agregada (un recall por
estrategia). Este reporte abre esa caja: muestra pregunta por pregunta el ground truth
contra lo que el reranker terminó pasando al prompt, que es lo que el LLM realmente ve.

Dos ejes, según el tipo de pregunta:
  · doc-id  (preguntas CWE)   — `expected_doc_ids` vs `ranked_ids`. El id de un CWE es
    determinístico (cwe-89), así que es estable entre estrategias.
  · fuente  (preguntas guía)  — `expected_sources` vs `ranked_sources`. El id de un chunk
    es hash de su contenido y cambia con cada estrategia; el nombre del archivo original
    no. Ver docs/data_splitting.md §6.

No recalcula nada: lee el JSON que ya persistió el experimento.

Uso (en el host, no necesita container):
    python src/pipeline/eval/report_chunking.py                  # último resultado
    python src/pipeline/eval/report_chunking.py --json <ruta>
    python src/pipeline/eval/report_chunking.py --solo-fallos    # solo lo que falló
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _fmt(x) -> str:
    return f"{x:.2f}" if isinstance(x, (int, float)) else "n/a"


def _etiqueta_id(doc_id: str, source: str | None) -> str:
    """Un id `cwe-89` se lee solo; un id de chunk es un hash del contenido y no dice
    nada, así que se lo muestra como `chunk(<archivo>)`. Esa diferencia ES el motivo
    de que exista la métrica por fuente (ver docstring del módulo)."""
    doc_id = str(doc_id)
    if doc_id.startswith("cwe-"):
        return doc_id
    corto = doc_id[:8]
    return f"chunk:{corto}…({source})" if source else f"{corto}…"


def _short(items, n=4, sources=None) -> str:
    """Recorta listas largas para que la salida siga siendo legible."""
    if not items:
        return "(ninguno)"
    sources = sources or []
    etiquetas = [_etiqueta_id(x, sources[i] if i < len(sources) else None)
                 for i, x in enumerate(items)]
    head = ", ".join(etiquetas[:n])
    return head + (f"  (+{len(etiquetas) - n} más)" if len(etiquetas) > n else "")


def _posicion(esperados, usados) -> str:
    """En qué posición del ranking apareció el primer documento esperado."""
    exp = set(esperados)
    for i, u in enumerate(usados, 1):
        if u in exp:
            return f"posición {i}"
    return "no aparece"


def _source_rr(q) -> float | None:
    """MRR por fuente. Se calcula post-hoc desde `ranked_sources` si el JSON no lo trae
    (corridas anteriores a metrics.source_reciprocal_rank): el orden ya está guardado,
    así que no hace falta re-correr el experimento."""
    if q.get("source_rr") is not None:
        return q["source_rr"]
    exp = set(q.get("expected_sources") or [])
    if not exp:
        return None
    for i, s in enumerate(q.get("ranked_sources") or [], start=1):
        if s in exp:
            return 1.0 / i
    return 0.0


def _agg_source_mrr(per_question: list[dict]) -> float | None:
    vals = [v for v in (_source_rr(q) for q in per_question) if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def print_comparison(results: dict) -> None:
    print("=" * 92)
    print("COMPARACIÓN DE ESTRATEGIAS")
    print("=" * 92)
    hdr = (f"{'estrategia':<18} {'chunks':>7} {'avg_w':>7} | {'recall@k':>9} {'mrr':>6} "
           f"| {'src_recall':>11} {'src_mrr':>8}")
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        s = r["summary"]
        print(f"{name:<18} {s['n_chunks']:>7} {s['avg_words']:>7.1f} | "
              f"{_fmt(s['retrieval'].get('recall_at_k')):>9} "
              f"{_fmt(s['retrieval'].get('mrr')):>6} | "
              f"{_fmt(s['source'].get('source_recall_at_k')):>11} "
              f"{_fmt(_agg_source_mrr(r['per_question'])):>8}")
    print("\n  CONTROL   recall@k / mrr : por doc-id, preguntas CWE (no se chunkean —")
    print("                              si bajan, la estrategia rompió algo).")
    print("  DECISIÓN  src_recall      : ¿recuperó la guía correcta? (0 ó 1 por pregunta)")
    print("            src_mrr         : ¿en qué POSICIÓN la recuperó? Desempata cuando")
    print("                              src_recall empata en 1.000.")


def print_detail(results: dict, solo_fallos: bool) -> None:
    for name, r in results.items():
        s = r["summary"]
        print("\n" + "=" * 78)
        print(f"DETALLE — {name}   ({s['n_chunks']} chunks, {s['avg_words']} palabras/chunk)")
        print("=" * 78)

        for q in r["per_question"]:
            if q.get("status") == "error":
                print(f"\n  ERROR  {q['id']}: {q.get('error')}")
                continue

            exp_ids, exp_src = q.get("expected_doc_ids") or [], q.get("expected_sources") or []
            if not exp_ids and not exp_src:
                continue  # pregunta negativa / sin ground truth

            # Eje que aplica a esta pregunta.
            src_usadas = q.get("ranked_sources") or []
            if exp_src:
                esperado, usado = exp_src, [s for s in src_usadas if s]
                metrica, valor, etiqueta = "source_recall", q.get("source_recall"), "fuente"
                usado_txt = _short(usado)
            else:
                esperado, usado = exp_ids, q.get("ranked_ids") or []
                metrica, valor, etiqueta = "recall", q.get("recall"), "doc-id"
                usado_txt = _short(usado, sources=src_usadas)

            ok = bool(valor)
            if solo_fallos and ok:
                continue

            print(f"\n  {'✓' if ok else '✗'} {q['id']}  [{q.get('category','?')}]  ({etiqueta})")
            print(f"      esperado : {_short(esperado)}")
            print(f"      usado    : {usado_txt}")
            print(f"      {metrica}={_fmt(valor)}   {_posicion(esperado, usado)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=None,
                    help="Resultado a reportar (default: el más reciente en results/).")
    ap.add_argument("--solo-fallos", action="store_true",
                    help="Mostrar únicamente las preguntas que fallaron.")
    args = ap.parse_args()

    path = args.json
    if path is None:
        candidates = sorted(RESULTS_DIR.glob("chunking_experiment_*.json"))
        if not candidates:
            raise SystemExit(f"No hay resultados en {RESULTS_DIR}. Correr primero "
                             f"run_chunking_experiment.py")
        path = candidates[-1]

    results = json.loads(path.read_text(encoding="utf-8"))
    print(f"Fuente: {path.name}\n")
    print_comparison(results)
    print_detail(results, args.solo_fallos)


if __name__ == "__main__":
    main()
