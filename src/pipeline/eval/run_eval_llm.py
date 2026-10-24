#!/usr/bin/env python3
"""Eval harness, Tier 3: el Ollama local como juez de faithfulness y context relevance.
Lento: correr a mano y sobre subsets. Escribe en las mismas tablas con `suite="judge"`.
Ver docs/eval/eval_harness.md.

    scripts/eval_llm.sh --limit 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parent
META_PATH = EVAL_DIR / "eval_meta.yaml"
sys.path.insert(0, str(EVAL_DIR.parent))

import csv_store  # noqa: E402
import pipeline_ciberseguridad as rag  # noqa: E402
from runner import Tee, git_metadata, load_dataset, make_run_id  # noqa: E402

from haystack_integrations.components.generators.ollama import OllamaChatGenerator  # noqa: E402
from haystack.components.evaluators import (  # noqa: E402
    FaithfulnessEvaluator,
    ContextRelevanceEvaluator,
)


# Esquemas de salida de cada evaluador, como structured output: sin esto el juez
# pega los scores en un solo número (`[10]` en vez de `[1, 0]`). `boolean` y no
# `integer` con enum, que el modelo igual viola. Ver bitácora, hallazgo 16.
SCHEMA_FAITHFULNESS = {
    "type": "array", "items": {"type": "boolean"},
}
JSON_SCHEMAS = {
    "faithfulness": {
        "statements": {"type": "array", "items": {"type": "string"}},
        "statement_scores": SCHEMA_FAITHFULNESS,
    },
    "context_relevance": {
        "relevant_statements": {"type": "array", "items": {"type": "string"}},
    },
}


def _response_format(nombre: str) -> dict:
    """El JSON Schema que se le pasa a Ollama como `response_format`.

    Ollama espera el schema **pelado** (`{"type": "object", ...}`), a diferencia del
    envoltorio `{"type": "json_schema", "json_schema": {...}}` que pedía la API de
    OpenAI/Groq cuando el juez corría por ahí.
    """
    props = JSON_SCHEMAS[nombre]
    return {"type": "object", "properties": props,
            "required": list(props), "additionalProperties": False}


def make_judge(model: str, schema: str | None = None):
    """Juez LLM determinístico y en modo JSON sobre el Ollama local.

    `schema` es la clave de JSON_SCHEMAS que corresponde al evaluador que va a usar
    este juez, y va por `response_format` para forzar la FORMA de la salida, no sólo
    que sea JSON válido: sin eso reaparece el hallazgo 16 (`statement_scores: [10]`
    en vez de `[true, false]`, que dispara la métrica a cientos de miles).

    Sin `schema` cae en `"json"` a secas, que garantiza sintaxis pero no forma.
    """
    return OllamaChatGenerator(
        model=model,
        url=rag.OLLAMA_URL,
        timeout=120,
        response_format=_response_format(schema) if schema else "json",
        generation_kwargs={"temperature": 0, "seed": 42},
    )


def run_pipeline(pipeline, question: str):
    """Devuelve (contexts: list[str], answer: str) — lo que necesita el juez."""
    result = pipeline.run(
        rag.build_query(question).inputs(),
        include_outputs_from={"ranker"},
    )
    # Contexto = la salida del ranker, que es lo que lee el LLM, no la del joiner
    # (bitácora, hallazgo 15).
    contexts = [d.content for d in result["ranker"]["documents"] if d.content]
    replies  = result.get("llm", {}).get("replies") or []
    answer   = (replies[0].text if replies else "") or ""
    return contexts, answer


def question_row(run_id: str, item: dict, question: str, answer: str,
                 faithfulness, context_relevance) -> dict:
    """La fila de detalle de `questions.csv` para una pregunta juzgada.

    Las columnas de Tier 1 quedan **vacías a propósito**: este runner no calcula
    recall/mrr (no compara contra los `expected_doc_ids` del dataset), así que no
    hay de dónde sacarlas honestamente. Para eso está `run_eval.py`; inventarlas
    acá sería peor que dejarlas vacías.
    """
    return {
        "run_id":            run_id,
        "question_id":       item["id"],
        "category":          item.get("category"),
        "status":            "ok",
        "faithfulness":      faithfulness,
        "context_relevance": context_relevance,
        "question":          question,
        "answer":            answer,
    }


def run_row(run_id: str, valves, meta: dict, judge_model: str, dataset_n: int,
            n_judge: int, faithfulness, context_relevance, duration_s: float,
            label: str) -> dict:
    """La fila resumen de `runs.csv`, con `suite="judge"`.

    Las columnas de Tier 1 y Tier 2 quedan vacías: esta suite no las mide. Por eso
    `report.py` elige "la última corrida" sólo entre las de `suite="retrieval"` —
    una fila judge no tiene recall que comparar.
    """
    return {
        "run_id":            run_id,
        "timestamp_utc":     datetime.now(timezone.utc).isoformat(),
        "suite":             "judge",
        "epoch":             meta["epoch"],
        "label":             label,
        **git_metadata(),
        "llm_provider":      "ollama",
        "llm_model":         csv_store.effective_llm_model(valves),
        "judge_model":       judge_model,
        "temperature":       0,
        "embedding_model":   valves.embedding_model,
        "ranker_model":      valves.ranker_model,
        "retriever_top_k":   valves.retriever_top_k,
        "ranker_top_k":      valves.ranker_top_k,
        "chunking":          "production",
        "dataset_n":         dataset_n,
        "duration_s":        round(duration_s, 1),
        "n_judge":           n_judge,
        "faithfulness":      faithfulness,
        "context_relevance": context_relevance,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval harness Tier 3 (juez LLM local sobre Ollama)")
    ap.add_argument("--dataset", type=Path, default=EVAL_DIR / "dataset.yaml")
    ap.add_argument("--out", type=Path, default=EVAL_DIR / "results")
    ap.add_argument("--judge-model", type=str, default=None, help="modelo juez (default: el del pipeline)")
    ap.add_argument("--limit", type=int, default=None, help="correr solo las primeras N preguntas (recomendado)")
    ap.add_argument("--label", type=str, default="", help="nota libre para identificar la corrida")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    runs_csv      = args.out / "runs.csv"
    questions_csv = args.out / "questions.csv"
    run_id        = make_run_id(runs_csv)
    meta          = csv_store.load_meta(META_PATH)

    log_dir = args.out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = (log_dir / f"{run_id}.log").open("w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)

    t0 = time.perf_counter()
    dataset = load_dataset(args.dataset)
    if args.limit:
        dataset = dataset[: args.limit]

    store  = rag.get_document_store()
    valves = rag.Pipeline.Valves(temperature=0.0)
    pipeline = rag.build_rag_pipeline(store, valves)

    # Modelo del juez: --judge-model gana; si no, el mismo que usa el generador,
    # resuelto con la misma precedencia que build_generator (LLM_MODEL > valve).
    judge_model = args.judge_model or os.getenv("LLM_MODEL") or valves.llm_model

    print(f"Tier 3 — run_id={run_id} epoch={meta['epoch']}  |  juez={judge_model}  "
          f"|  {len(dataset)} preguntas\n")

    # 1) Correr el RAG y juntar (pregunta, contexto, respuesta).
    questions, contexts, answers, ids = [], [], [], []
    for item in dataset:
        ctx, ans = run_pipeline(pipeline, item["question"])
        questions.append(item["question"])
        contexts.append(ctx)
        answers.append(ans)
        ids.append(item["id"])
        print(f"  · RAG corrido: {item['id']}")

    # 2) Juzgar. raise_on_failure=False → una pregunta que no parsee no tumba el batch.
    print("\n  Juzgando faithfulness...")
    faith = FaithfulnessEvaluator(chat_generator=make_judge(judge_model, "faithfulness"),
                                  raise_on_failure=False, progress_bar=False)
    fres = faith.run(questions=questions, contexts=contexts, predicted_answers=answers)

    print("  Juzgando context relevance...")
    ctxrel = ContextRelevanceEvaluator(chat_generator=make_judge(judge_model, "context_relevance"), 
                                       raise_on_failure=False, progress_bar=False) # Indica qué tan relevantes son los contextos para cada pregunta.
    cres = ctxrel.run(questions=questions, contexts=contexts) # cres es un dict con 'score' y 'individual_scores' para cada pregunta.

    f_ind = fres.get("individual_scores", [])
    c_ind = cres.get("individual_scores", [])

    print("\n" + "=" * 64)
    print("TIER 3 — JUEZ LLM (por pregunta)")
    print("=" * 64)
    n_judge = 0
    for i, (item, qid) in enumerate(zip(dataset, ids)):
        f = f_ind[i] if i < len(f_ind) else None
        c = c_ind[i] if i < len(c_ind) else None
        if f is not None or c is not None:
            n_judge += 1
        fs = f"{f:.2f}" if isinstance(f, (int, float)) else "n/a"
        cs = f"{c:.2f}" if isinstance(c, (int, float)) else "n/a"
        print(f"  {qid:<28} faithfulness={fs}  context_relevance={cs}")

        # Fila por fila, igual que run_eval.py: si el proceso muere en la
        # pregunta 20, las 19 anteriores ya están guardadas.
        csv_store.append_row(
            questions_csv,
            question_row(run_id, item, questions[i], answers[i], f, c),
            csv_store.QUESTIONS_COLUMNS,
        )

    print("\n" + "=" * 64)
    print(f"  GLOBAL:  faithfulness={fres.get('score')}   context_relevance={cres.get('score')}")
    print("=" * 64)

    csv_store.append_row(
        runs_csv,
        run_row(run_id, valves, meta, judge_model, len(dataset), n_judge,
                fres.get("score"), cres.get("score"),
                time.perf_counter() - t0, args.label),
        csv_store.RUNS_COLUMNS,
    )

    print(f"\nResultados persistidos en {args.out}/:")
    print(f"  · runs.csv        (fila suite=judge de la corrida {run_id})")
    print(f"  · questions.csv   ({len(dataset)} filas de detalle)")
    print(f"  · logs/{run_id}.log")
    print("\nVer con:  python report.py runs --all")

    sys.stdout.flush()
    sys.stdout = sys.__stdout__   # restaurar antes de cerrar, si no el flush de shutdown falla
    log_file.close()


if __name__ == "__main__":
    main()
