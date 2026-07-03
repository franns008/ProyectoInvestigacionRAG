#!/usr/bin/env python3
"""Eval harness — runner principal.

Corre todas las preguntas del dataset contra el RAG, mide retrieval (Tier 1)
contra el ground truth y guarda un snapshot de resultados. La respuesta generada
se captura para el Tier 2 (SAS), que se agrega en la Fase 2.

Ejecutar DENTRO del container `pipelines` (donde resuelven vdb/ollama/GROQ_API_KEY):

    docker compose exec pipelines python /app/pipelines/eval/run_eval.py

Ver docs/eval_harness.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

# El módulo del pipeline vive en el directorio padre (/app/pipelines).
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

import pipeline_ciberseguridad as rag  # noqa: E402
import metrics as m  # noqa: E402
import report as rep  # noqa: E402


def load_dataset(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("dataset.yaml debe ser una lista de preguntas")
    return data


class _Tee:
    """Duplica lo escrito a stdout hacia un archivo, para persistir el log de la corrida."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)

    def flush(self):
        for st in self._streams:
            st.flush()


def append_history(out_dir: Path, snapshot: dict) -> Path:
    """Agrega una fila resumen al history.csv acumulativo (evolución de las métricas)."""
    hist = out_dir / "history.csv"
    o = snapshot["overall"]
    s = snapshot.get("sas", {}) or {}
    c = snapshot["config"]
    row = {
        "timestamp":       snapshot["timestamp"],
        "top_k":           c["top_k"],
        "temperature":     c["temperature"],
        "llm_model":       c["llm_model"],
        "embedding_model": c["embedding_model"],
        "n_retrieval":     o["n"],
        "recall_at_k":     o["recall_at_k"],
        "hit_rate":        o["hit_rate"],
        "mrr":             o["mrr"],
        "n_sas":           s.get("n"),
        "sas":             s.get("sas"),
    }
    write_header = not hist.exists()
    with hist.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            w.writeheader()
        w.writerow(row)
    return hist


def run_question(pipeline, question: str):
    """Una corrida del pipeline; devuelve (emb_ids, kw_ids, joined_ids, answer)."""
    result = pipeline.run(
        {
            "text_embedder":     {"text": question},
            "keyword_retriever": {"query": rag.build_keyword_query(question)},
            "prompt_builder":    {"question": question},
        },
        include_outputs_from={"embedding_retriever", "keyword_retriever", "document_joiner"},
    )
    emb_ids    = [d.id for d in result["embedding_retriever"]["documents"]]
    kw_ids     = [d.id for d in result["keyword_retriever"]["documents"]]
    joined_ids = [d.id for d in result["document_joiner"]["documents"]]
    answer     = result.get("llm", {}).get("replies", [None])[0]
    return emb_ids, kw_ids, joined_ids, answer


def compute_sas(embedder, answer: str | None, reference: str | None) -> float | None:
    """SAS (Tier 2): coseno entre embeddings Ollama de la respuesta y la referencia.
    None si falta cualquiera de las dos."""
    if not answer or not reference:
        return None
    a = embedder.run(text=answer)["embedding"]
    r = embedder.run(text=reference)["embedding"]
    return round(m.cosine_similarity(a, r), 4)


def evaluate(item: dict, emb_ids, kw_ids, joined_ids, answer) -> dict:
    expected = item.get("expected_doc_ids") or []
    rec = {
        "id":               item["id"],
        "category":         item.get("category", "?"),
        "question":         item["question"],
        "expected_doc_ids": expected,
        "retrieved_ids":    joined_ids,
        "answer":           answer,
        "reference_answer": item.get("reference_answer"),
        "status":           "ok",
    }
    if expected:
        rec["recall"] = m.recall_at_k(joined_ids, expected)
        rec["hit"]    = m.hit_at_k(joined_ids, expected)
        rec["rr"]     = m.reciprocal_rank(joined_ids, expected)
        exp = set(expected)
        rec["hit_source"] = {          # qué retriever aportó algún esperado
            "embedding": bool(exp & set(emb_ids)),
            "keyword":   bool(exp & set(kw_ids)),
        }
    else:
        rec["recall"] = rec["hit"] = rec["rr"] = None
        rec["hit_source"] = None
    return rec


# ── impresión ───────────────────────────────────────────────────────────────

def _sas_str(rec: dict) -> str:
    return f" sas={rec['sas']:.3f}" if rec.get("sas") is not None else ""


def _print_row(i: int, rec: dict) -> None:
    if rec["status"] == "error":
        print(f"  {i:>2}. ✗ {rec['id']:<28} [{rec['category']}] ERROR: {rec['error']}")
        return
    if rec["recall"] is None:  # negativa / sin ground truth
        print(f"  {i:>2}. – {rec['id']:<28} [{rec['category']}] (negativa){_sas_str(rec)}")
        return
    mark = "✓" if rec["hit"] else "✗"
    src = "+".join(k for k, v in rec["hit_source"].items() if v) or "ninguno"
    print(f"  {i:>2}. {mark} {rec['id']:<28} [{rec['category']}] "
          f"recall={rec['recall']:.2f} rr={rec['rr']:.2f} via={src}{_sas_str(rec)}")


def _print_summary(overall: dict, by_cat: dict, sas_overall: dict) -> None:
    print("\n" + "=" * 64)
    print("RESUMEN")
    print("=" * 64)
    if sas_overall.get("sas") is not None:
        print(f"  Tier 2 SAS (n={sas_overall['n']}):  sas={sas_overall['sas']:.3f}")
    print("\n  Tier 1 RETRIEVAL — solo preguntas con ground truth:")
    if overall["n"] == 0:
        print("    Sin preguntas con ground truth.")
        return
    print(f"    overall (n={overall['n']}):  "
          f"recall@k={overall['recall_at_k']:.3f}  "
          f"hit_rate={overall['hit_rate']:.3f}  "
          f"mrr={overall['mrr']:.3f}")
    print("\n    Por categoría:")
    for cat, agg in by_cat.items():
        if agg["n"] == 0:
            continue
        print(f"      {cat:<16} n={agg['n']:<2}  recall={agg['recall_at_k']:.3f}  "
              f"hit={agg['hit_rate']:.3f}  mrr={agg['mrr']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval harness del RAG (Tier 1: retrieval)")
    ap.add_argument("--dataset", type=Path, default=EVAL_DIR / "dataset.yaml")
    ap.add_argument("--out", type=Path, default=EVAL_DIR / "results")
    ap.add_argument("--top-k", type=int, default=None, help="override de retriever_top_k")
    ap.add_argument("--temperature", type=float, default=0.0, help="temp de generación (0 = determinístico)")
    ap.add_argument("--limit", type=int, default=None, help="correr solo las primeras N preguntas")
    ap.add_argument("--set-baseline", action="store_true", help="promover esta corrida a baseline.json")
    ap.add_argument("--baseline", type=Path, default=EVAL_DIR / "baseline.json")
    args = ap.parse_args()

    # Persistencia: cada corrida deja un .log de texto (tee de la consola), un .json
    # completo y una fila en history.csv. Todo en results/ (bind-mount al host).
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_file = (args.out / f"{stamp}.log").open("w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    dataset = load_dataset(args.dataset)
    if args.limit:
        dataset = dataset[: args.limit]

    store  = rag.get_document_store()
    valves = rag.Pipeline.Valves(temperature=args.temperature)
    if args.top_k is not None:
        valves.retriever_top_k = args.top_k
    pipeline = rag.build_rag_pipeline(store, valves)
    embedder = rag.OllamaTextEmbedder(model=valves.embedding_model, url=rag.OLLAMA_URL)

    print(f"Corriendo {len(dataset)} preguntas "
          f"(top_k={valves.retriever_top_k}, temp={valves.temperature}, "
          f"docs_en_store={len(store.filter_documents())})\n")

    per_question: list[dict] = []
    for i, item in enumerate(dataset, 1):
        try:
            emb, kw, joined, answer = run_question(pipeline, item["question"])
            rec = evaluate(item, emb, kw, joined, answer)
        except Exception as e:  # noqa: BLE001 — queremos seguir con el resto
            rec = {
                "id": item["id"], "category": item.get("category", "?"),
                "question": item["question"],
                "expected_doc_ids": item.get("expected_doc_ids") or [],
                "answer": None, "reference_answer": item.get("reference_answer"),
                "recall": None, "hit": None, "rr": None,
                "status": "error", "error": repr(e),
            }
        rec["sas"] = compute_sas(embedder, rec.get("answer"), rec.get("reference_answer"))
        per_question.append(rec)
        _print_row(i, rec)

    overall = m.aggregate_retrieval(per_question)
    by_cat  = m.aggregate_by_category(per_question)
    sas     = m.aggregate_sas(per_question)
    _print_summary(overall, by_cat, sas)

    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "top_k": valves.retriever_top_k,
            "temperature": valves.temperature,
            "llm_model": valves.llm_model,
            "embedding_model": valves.embedding_model,
        },
        "overall": overall,
        "sas": sas,
        "by_category": by_cat,
        "per_question": per_question,
    }
    out_path = args.out / f"{stamp}.json"
    out_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    hist_path = append_history(args.out, snapshot)

    # Delta contra el baseline (y promoción opcional).
    baseline = rep.load_snapshot(args.baseline)
    rep.print_delta(snapshot, baseline)
    if args.set_baseline:
        rep.promote(snapshot, args.baseline)
        print(f"\nBaseline actualizado: {args.baseline}")

    print(f"\nResultados persistidos en {args.out}/:")
    print(f"  · {out_path.name}   (JSON completo)")
    print(f"  · {stamp}.log   (este log)")
    print(f"  · {hist_path.name}   (fila resumen acumulativa)")

    sys.stdout.flush()
    sys.stdout = sys.__stdout__   # restaurar antes de cerrar, si no el flush de shutdown falla
    log_file.close()


if __name__ == "__main__":
    main()
