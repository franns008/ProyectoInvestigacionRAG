#!/usr/bin/env python3
"""Eval harness — runner principal.

Corre todas las preguntas del dataset contra el RAG, mide retrieval (Tier 1) contra
el ground truth y **persiste los resultados crudos** en las tablas CSV acumulativas
(`results/runs.csv` + `results/questions.csv`). La respuesta generada se captura para
el Tier 2 (SAS).

Principio: este script **escribe crudo, no compara**. Todo el análisis (deltas contra
el baseline, regresiones, evolución) vive en `report.py`, que lee esas tablas. Al final
se invoca el reporte por cortesía, pero los datos se escriben ANTES de cualquier
análisis: si el reporte falla, la corrida no se pierde.

Ejecutar DENTRO del container `pipelines` (donde resuelven vdb/ollama/GROQ_API_KEY):

    docker compose exec pipelines python /app/pipelines/eval/run_eval.py

o vía el wrapper `scripts/eval.sh`, que además inyecta la metadata de git.

Ver docs/eval/refactor_resultados_csv.md y docs/eval/eval_harness.md.
"""
from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

# El módulo del pipeline vive en el directorio padre (/app/pipelines).
EVAL_DIR = Path(__file__).resolve().parent
META_PATH = EVAL_DIR / "eval_meta.yaml"
sys.path.insert(0, str(EVAL_DIR.parent))

import csv_store  # noqa: E402
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


def make_run_id(runs_csv: Path) -> str:
    """`run_id` = timestamp UTC compacto, único dentro de runs.csv.

    Dos corridas dentro del mismo segundo colisionarían; en ese caso se sufija
    `-2`, `-3`, … El orden lexicográfico de los run_id sigue siendo cronológico,
    que es de lo que depende `report.py` para saber cuál es "la última corrida".
    """
    base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    usados = set(csv_store.load_runs(runs_csv)["run_id"].dropna())
    if base not in usados:
        return base
    n = 2
    while f"{base}-{n}" in usados:
        n += 1
    return f"{base}-{n}"


def git_metadata() -> dict:
    """Metadata de git, inyectada por `scripts/eval.sh` como variables de entorno.

    El container **no** tiene el `.git` montado, así que acá nunca se corre `git`:
    si las variables no están (corrida a mano, sin el wrapper), quedan vacías y la
    corrida sigue igual.
    """
    return {
        "git_commit": os.getenv("GIT_COMMIT", ""),
        "git_branch": os.getenv("GIT_BRANCH", ""),
        "git_dirty":  os.getenv("GIT_DIRTY", ""),
    }


def set_baseline(meta_path: Path, run_id: str) -> None:
    """Apunta `baseline_run_id` a `run_id` en eval_meta.yaml.

    Reemplaza **esa línea sola** por regex sobre el texto crudo. No se usa
    `yaml.safe_dump` a propósito: PyYAML descarta los comentarios, y este archivo
    es casi todo comentarios más el historial de epochs.
    """
    texto = meta_path.read_text(encoding="utf-8")
    nuevo, n = re.subn(
        r"^baseline_run_id:.*$",
        f"baseline_run_id: {run_id}",
        texto,
        count=1,
        flags=re.MULTILINE,
    )
    if n == 0:
        raise ValueError(
            f"no se encontró la línea 'baseline_run_id:' en {meta_path}; "
            f"archivo sin tocar. Revisalo a mano."
        )
    meta_path.write_text(nuevo, encoding="utf-8")


def rank_first_hit(retrieved_ids: list[str], expected_ids: list[str]) -> int | None:
    """Posición 1-indexada del primer documento esperado. None si no hubo acierto.

    Equivale a `round(1/rr)` pero se calcula directo, sin pasar por el recíproco.
    """
    esperados = set(expected_ids or [])
    if not esperados:
        return None
    for i, doc_id in enumerate(retrieved_ids or [], 1):
        if doc_id in esperados:
            return i
    return None


def run_question(pipeline, question: str):
    """Una corrida del pipeline; devuelve
    (emb_ids, kw_ids, ranked_ids, ranked_sources, answer, document_ids).

    `ranked_ids`/`ranked_sources` son los docs que el reranker deja al prompt (lo que
    ve el LLM), no los candidatos fusionados del joiner. Las métricas Tier 1 se
    calculan sobre esto. `ranked_sources` es meta.source de esos mismos docs — ground
    truth agnóstico a la estrategia de chunking (ver metrics.source_recall_at_k).
    """
    result = pipeline.run(
        {
            "text_embedder":     {"text": question},
            "keyword_retriever": {"query": rag.build_keyword_query(question)},
            "ranker":            {"query": question},
            "prompt_builder":    {"question": question},
        },
        include_outputs_from={"embedding_retriever", "keyword_retriever", "document_joiner", "ranker"},
    )
    ranked_docs    = result["ranker"]["documents"]
    emb_ids        = [d.id for d in result["embedding_retriever"]["documents"]]
    document_ids   = [d.id for d in result["document_joiner"]["documents"]]
    kw_ids         = [d.id for d in result["keyword_retriever"]["documents"]]
    ranked_ids     = [d.id for d in ranked_docs]
    ranked_sources = [d.meta.get("source") for d in ranked_docs]
    answer         = result.get("llm", {}).get("replies", [None])[0]
    return emb_ids, kw_ids, ranked_ids, ranked_sources, answer, document_ids


def compute_sas(embedder, answer: str | None, reference: str | None) -> float | None:
    """SAS (Tier 2): coseno entre embeddings Ollama de la respuesta y la referencia.
    None si falta cualquiera de las dos."""
    if not answer or not reference:
        return None
    a = embedder.run(text=answer)["embedding"]
    r = embedder.run(text=reference)["embedding"]
    return round(m.cosine_similarity(a, r), 4)

def check_correct_rejection(answer:str ) -> bool:
    """
        Verifica si el LLM se abstuvo correctamente de responder una pregunta
        para la cual no tenía contexto.
    """
    if not answer:
        return False
    # Convert the answer to lowercase for case-insensitive comparison
    answer_lower = answer.lower()
    # Check for common phrases indicating a correct rejection
    rejection_phrases = [
        "no puedo responder",
        "no tengo suficiente información",
        "no es posible responder",
        "no puedo proporcionar una respuesta",
        "no hay información suficiente",
        "no puedo determinar la respuesta",
        "no puedo dar una respuesta precisa",
        "no puedo contestar",
        "no puedo ayudar con eso",
        "no tengo datos suficientes"
    ]
    return any(phrase in answer_lower for phrase in rejection_phrases)


def evaluate(item: dict, emb_ids, kw_ids, ranked_ids, ranked_sources, answer, document_ids=None) -> dict:
    """`document_ids` (candidatos del joiner) es opcional: el experimento de chunking
    corre en retrieval puro y sólo necesita ranked_ids/ranked_sources."""
    expected = item.get("expected_doc_ids") or []
    expected_sources = item.get("expected_sources") or []
    rec = {
        "id":               item["id"],
        "category":         item.get("category", "?"),
        "question":         item["question"],
        "expected_doc_ids": expected,
        "expected_sources": expected_sources,
        "ranked_ids":       ranked_ids,  # Renombrado de retrieved_ids/joined_ids
        "ranked_sources":   ranked_sources,
        "document_ids":     document_ids,
        "answer":           answer,
        "reference_answer": item.get("reference_answer"),
        "status":           "ok",
    }
    if expected:
        rec["recall"] = m.recall_at_k(ranked_ids, expected)
        rec["hit"]    = m.hit_at_k(ranked_ids, expected)
        rec["rr"]     = m.reciprocal_rank(ranked_ids, expected)
        exp = set(expected)
        rec["hit_source"] = {          # qué retriever aportó algún esperado
            "embedding": bool(exp & set(emb_ids)),
            "keyword":   bool(exp & set(kw_ids)),
        }
    else:
        rec["recall"] = rec["hit"] = rec["rr"] = None
        rec["hit_source"] = None
        rec["correct_rejection"] = check_correct_rejection(answer)  # Verifica si el LLM se abstuvo correctamente
        #FUncion a mejorar, debería ser más robusta y considerar más casos de abstención correcta.

    # Tier 1b: ground truth a nivel de fuente (agnóstico a la estrategia de chunking).
    if expected_sources:
        rec["source_recall"] = m.source_recall_at_k(ranked_sources, expected_sources)
        rec["source_hit"]    = m.source_hit_at_k(ranked_sources, expected_sources)
        rec["source_rr"]     = m.source_reciprocal_rank(ranked_sources, expected_sources)
    else:
        rec["source_recall"] = rec["source_hit"] = rec["source_rr"] = None
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


def _print_summary(overall: dict, by_cat: dict, sas_overall: dict, src_overall: dict | None = None) -> None:
    print("\n" + "=" * 64)
    print("RESUMEN")
    print("=" * 64)
    if sas_overall.get("sas") is not None:
        print(f"  Tier 2 SAS (n={sas_overall['n']}):  sas={sas_overall['sas']:.3f}")
    print("\n  Tier 1 RETRIEVAL — solo preguntas con ground truth:")
    if overall["n"] == 0:
        print("    Sin preguntas con ground truth.")
    else:
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

    # Tier 1b: recall a nivel de fuente — el eje que mide el chunking del splittable.
    if src_overall and src_overall.get("source_recall_at_k") is not None:
        print(f"\n  Tier 1b FUENTE (n={src_overall['n']}):  "
              f"source_recall@k={src_overall['source_recall_at_k']:.3f}  "
              f"source_hit={src_overall['source_hit_rate']:.3f}")


# ── armado de filas para las tablas CSV ─────────────────────────────────────

def question_row(run_id: str, rec: dict) -> dict:
    """Una fila de `questions.csv` a partir del `rec` que devuelve `evaluate()`.

    Los sufijos `_eff` son deliberados: recall/hit/rr se calculan sobre la salida
    del **reranker** (lo que ve el LLM), no sobre los candidatos del joiner. El
    nombre honesto evita el malentendido que describe H1 en mejoras_harness.md.
    `joined_ids` guarda esos candidatos, que es lo que habilita medir el techo del
    retriever cuando se implemente H1, sin migrar nada.
    """
    hit_source = rec.get("hit_source") or {}
    ranked_ids = rec.get("ranked_ids") or []
    expected   = rec.get("expected_doc_ids") or []
    return {
        "run_id":            run_id,
        "question_id":       rec["id"],
        "category":          rec.get("category"),
        "status":            rec.get("status"),
        "error":             rec.get("error"),
        "n_expected":        len(expected),
        "expected_ids":      expected,
        "retrieved_ids":     ranked_ids,
        "joined_ids":        rec.get("document_ids"),
        "rank_first_hit":    rank_first_hit(ranked_ids, expected),
        "recall_eff":        rec.get("recall"),
        "hit_eff":           rec.get("hit"),
        "rr_eff":            rec.get("rr"),
        "via_embedding":     hit_source.get("embedding"),
        "via_keyword":       hit_source.get("keyword"),
        "expected_sources":  rec.get("expected_sources"),
        "retrieved_sources": rec.get("ranked_sources"),
        "source_recall":     rec.get("source_recall"),
        "source_hit":        rec.get("source_hit"),
        "source_rr":         rec.get("source_rr"),
        "correct_rejection": rec.get("correct_rejection"),
        "sas":               rec.get("sas"),
        "faithfulness":      None,   # las llena Tier 3 (run_eval_llm.py)
        "context_relevance": None,
        "question":          rec.get("question"),
        "answer":            rec.get("answer"),
        "reference_answer":  rec.get("reference_answer"),
    }


def run_row(run_id: str, valves, meta: dict, per_question: list[dict],
            overall: dict, src_overall: dict, sas: dict,
            n_docs_store: int, dataset_n: int, duration_s: float,
            label: str) -> dict:
    """La fila resumen de `runs.csv` (una por corrida)."""
    sas_values = [q["sas"] for q in per_question if q.get("sas") is not None]
    return {
        "run_id":          run_id,
        "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
        "suite":           "retrieval",
        "epoch":           meta["epoch"],
        "label":           label,
        **git_metadata(),
        "llm_provider":    rag._llm_provider(),
        # Modelo EFECTIVO, no el default de los valves: ver csv_store.effective_llm_model.
        "llm_model":       csv_store.effective_llm_model(valves),
        "judge_model":     None,          # sólo suite=judge
        "temperature":     valves.temperature,
        "embedding_model": valves.embedding_model,
        "ranker_model":    valves.ranker_model,
        "retriever_top_k": valves.retriever_top_k,
        "ranker_top_k":    valves.ranker_top_k,
        "chunking":        "production",
        "n_docs_store":    n_docs_store,
        "dataset_n":       dataset_n,
        "n_errors":        sum(1 for q in per_question if q.get("status") == "error"),
        "duration_s":      round(duration_s, 1),
        "n_gt":            overall["n"],
        "recall_eff":      overall["recall_at_k"],
        "hit_eff":         overall["hit_rate"],
        "mrr_eff":         overall["mrr"],
        "n_source":        src_overall.get("n"),
        "source_recall":   src_overall.get("source_recall_at_k"),
        "source_hit":      src_overall.get("source_hit_rate"),
        "source_mrr":      src_overall.get("source_mrr"),
        "n_sas":           sas.get("n"),
        "sas_mean":        sas.get("sas"),
        "sas_std":         statistics.stdev(sas_values) if len(sas_values) >= 2 else None,
        "n_judge":         None,          # sólo suite=judge
        "faithfulness":    None,
        "context_relevance": None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval harness del RAG (Tier 1: retrieval)")
    ap.add_argument("--dataset", type=Path, default=EVAL_DIR / "dataset.yaml")
    ap.add_argument("--out", type=Path, default=EVAL_DIR / "results")
    ap.add_argument("--top-k", type=int, default=None, help="override de retriever_top_k")
    ap.add_argument("--temperature", type=float, default=0.0, help="temp de generación (0 = determinístico)")
    ap.add_argument("--limit", type=int, default=None, help="correr solo las primeras N preguntas")
    ap.add_argument("--label", type=str, default="", help="nota libre para identificar la corrida")
    ap.add_argument("--set-baseline", action="store_true",
                    help="apuntar baseline_run_id de eval_meta.yaml a esta corrida")
    args = ap.parse_args()

    # Persistencia: dos tablas acumulativas + un log por corrida. Ver
    # docs/eval/refactor_resultados_csv.md.
    args.out.mkdir(parents=True, exist_ok=True)
    runs_csv      = args.out / "runs.csv"
    questions_csv = args.out / "questions.csv"
    run_id        = make_run_id(runs_csv)
    meta          = csv_store.load_meta(META_PATH)

    log_dir = args.out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = (log_dir / f"{run_id}.log").open("w", encoding="utf-8")
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

    n_docs_store = len(store.filter_documents())
    print(f"Corriendo {len(dataset)} preguntas — run_id={run_id} epoch={meta['epoch']} "
          f"(top_k={valves.retriever_top_k}, temp={valves.temperature}, "
          f"docs_en_store={n_docs_store})\n")

    t0 = time.perf_counter()
    per_question: list[dict] = []
    for i, item in enumerate(dataset, 1):
        try:
            
            emb_ids, kw_ids, ranked_ids, ranked_sources, answer, document_ids = run_question(pipeline, item["question"])

            rec = evaluate(item, emb_ids, kw_ids, ranked_ids, ranked_sources, answer, document_ids)
        except Exception as e:  # noqa: BLE001 — queremos seguir con el resto
            rec = {
                "id": item["id"], "category": item.get("category", "?"),
                "question": item["question"],
                "expected_doc_ids": item.get("expected_doc_ids") or [],
                "answer": None, "reference_answer": item.get("reference_answer"),
                "recall": None, "hit": None, "rr": None,
                "source_recall": None, "source_hit": None,
                "status": "error", "error": repr(e),
            }
        rec["sas"] = compute_sas(embedder, rec.get("answer"), rec.get("reference_answer"))
        per_question.append(rec)
        _print_row(i, rec)
        # Se persiste pregunta por pregunta, no al final: si la corrida muere en la
        # 20, las 19 anteriores ya están guardadas.
        csv_store.append_row(questions_csv, question_row(run_id, rec),
                             csv_store.QUESTIONS_COLUMNS)

    duration_s  = time.perf_counter() - t0
    overall     = m.aggregate_retrieval(per_question)
    by_cat      = m.aggregate_by_category(per_question)   # sólo para imprimir; no se persiste
    sas         = m.aggregate_sas(per_question)
    src_overall = m.aggregate_source_retrieval(per_question)
    _print_summary(overall, by_cat, sas, src_overall)

    # `by_category` deja de persistirse: report.py lo recalcula con un groupby sobre
    # questions.csv, así no hay dos fuentes de verdad para el mismo número.
    csv_store.append_row(
        runs_csv,
        run_row(run_id, valves, meta, per_question, overall, src_overall, sas,
                n_docs_store, len(dataset), duration_s, args.label),
        csv_store.RUNS_COLUMNS,
    )

    print(f"\nResultados persistidos en {args.out}/:")
    print(f"  · runs.csv        (fila de la corrida {run_id})")
    print(f"  · questions.csv   ({len(per_question)} filas de detalle)")
    print(f"  · logs/{run_id}.log")

    if args.set_baseline:
        set_baseline(META_PATH, run_id)
        print(f"\nBaseline actualizado: baseline_run_id = {run_id} ({META_PATH.name})")

    # Cortesía: el reporte va DESPUÉS de persistir, y si falla no se pierde nada.
    try:
        rep.compare(run_id=run_id, results_dir=args.out, meta_path=META_PATH)
    except Exception as e:  # noqa: BLE001
        print(f"\n⚠ reporte falló ({e!r}); los datos quedaron persistidos igual")

    sys.stdout.flush()
    sys.stdout = sys.__stdout__   # restaurar antes de cerrar, si no el flush de shutdown falla
    log_file.close()


if __name__ == "__main__":
    main()
