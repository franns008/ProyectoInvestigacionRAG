#!/usr/bin/env python3
"""Eval harness — Tier 3: juez LLM (MANUAL, rate-limited).

Mide la calidad de la RESPUESTA con evaluators nativos de Haystack usando Groq
como juez (endpoint compatible con OpenAI, modo JSON):

  - Faithfulness       : ¿las afirmaciones de la respuesta están fundadas en los
                         documentos recuperados? (anti-alucinación)
  - Context Relevance  : ¿los documentos recuperados son relevantes a la pregunta?

Multiplica llamadas a Groq (varias por pregunta) → correr a mano y sobre subsets
(--limit) para no comerse los 429. NO va en cada iteración ni en CI.

    scripts/eval_llm.sh --limit 8

Persistencia: igual que `run_eval.py`, escribe en las tablas acumulativas
`results/runs.csv` (con `suite="judge"`) y `results/questions.csv`. Ya no deja un
`tier3_<stamp>.json` suelto. Ver docs/eval/refactor_resultados_csv.md (Paso 6).

Ver docs/eval/eval_harness.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

EVAL_DIR = Path(__file__).resolve().parent
META_PATH = EVAL_DIR / "eval_meta.yaml"
sys.path.insert(0, str(EVAL_DIR.parent))

import csv_store  # noqa: E402
import pipeline_ciberseguridad as rag  # noqa: E402

from haystack.components.generators.chat import OpenAIChatGenerator  # noqa: E402
from haystack_integrations.components.generators.ollama import OllamaChatGenerator  # noqa: E402
from haystack.components.evaluators import (  # noqa: E402
    FaithfulnessEvaluator,
    ContextRelevanceEvaluator,
)


def load_dataset(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("dataset.yaml debe ser una lista de preguntas")
    return data


# Los tres helpers de abajo son gemelos de los de `run_eval.py`. Se duplican a
# propósito y no se importan de ahí: importar `run_eval` arrastraría todo el
# harness Tier 1 (metrics, report) y ataría los dos runners entre sí. Si alguno
# cambia, cambian los dos — son 20 líneas.

class _Tee:
    """Duplica lo escrito a stdout hacia un archivo, para persistir el log."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)

    def flush(self):
        for st in self._streams:
            st.flush()


def make_run_id(runs_csv: Path) -> str:
    """`run_id` = timestamp UTC compacto, único dentro de runs.csv."""
    base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    usados = set(csv_store.load_runs(runs_csv)["run_id"].dropna())
    if base not in usados:
        return base
    n = 2
    while f"{base}-{n}" in usados:
        n += 1
    return f"{base}-{n}"


def git_metadata() -> dict:
    """Metadata inyectada por `scripts/eval_llm.sh`. El container no tiene el
    `.git` montado, así que acá nunca se corre `git`: sin las variables, vacío."""
    return {
        "git_commit": os.getenv("GIT_COMMIT", ""),
        "git_branch": os.getenv("GIT_BRANCH", ""),
        "git_dirty":  os.getenv("GIT_DIRTY", ""),
    }


# Esquemas de salida de cada evaluador (sus `.outputs`), como structured output.
#
# Sin esto el juez devuelve JSON válido pero con la forma equivocada: ante dos
# afirmaciones responde `statement_scores: [10]` — pega el 1 y el 0 en un solo
# número en vez de dar un score por afirmación. Con 7 afirmaciones sale 1111000 y
# el "promedio" termina en cientos de miles. Ver bitácora, hallazgo 16.
#
# `boolean` y NO `integer` con `enum: [0,1]`: con el enum el modelo igual genera
# `10`, Groq lo rechaza contra el esquema y devuelve 400. Con booleanos genera
# `[true, false]` bien, y Haystack los promedia igual (True==1).
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
    props = JSON_SCHEMAS[nombre]
    return {"type": "json_schema", "json_schema": {
        "name": nombre, "strict": True, "schema": {
            "type": "object", "properties": props,
            "required": list(props), "additionalProperties": False}}}


def make_judge(model: str, schema: str | None = None):
    """Juez LLM determinístico y en modo JSON. Sigue LLM_PROVIDER (ver docs/modos_llm.md):
    - ollama : OllamaChatGenerator local → Tier-3 corre sin key de Groq (modo GPU).
    - groq   : OpenAIChatGenerator contra el endpoint OpenAI-compatible de Groq.

    `schema` es la clave de JSON_SCHEMAS que corresponde al evaluador que va a usar
    este juez. Sólo se aplica en el modo groq; el modo ollama sigue con `format:
    json` a secas y **no está validado** contra el problema del hallazgo 16.
    """
    if rag._llm_provider() == "ollama":
        return OllamaChatGenerator(
            model=model,
            url=rag.OLLAMA_URL,
            timeout=120,
            generation_kwargs={"format": "json", "temperature": 0},
        )
    formato = _response_format(schema) if schema else {"type": "json_object"}
    return OpenAIChatGenerator(
        api_key=rag.Secret.from_env_var("GROQ_API_KEY"),
        api_base_url=rag.GROQ_BASE_URL,
        model=model,
        generation_kwargs={"response_format": formato, "seed": 42, "temperature": 0},
    )


def run_pipeline(pipeline, question: str):
    """Devuelve (contexts: list[str], answer: str) — lo que necesita el juez."""
    result = pipeline.run(
        {
            "text_embedder":     {"text": question},
            "keyword_retriever": {"query": rag.build_keyword_query(question)},
            # El `ranker` es obligatorio desde que entró el cross-encoder. Sin esta
            # línea el pipeline ni arranca ("Missing mandatory input 'query'").
            "ranker":            {"query": question},
            "prompt_builder":    {"question": question},
        },
        include_outputs_from={"ranker"},
    )
    # Contexto = la salida del RANKER, no la del document_joiner.
    #
    # El plan (§Paso 6) dice document_joiner, pero eso fue escrito antes de que el
    # cross-encoder entrara al pipeline. Hoy el cableado es
    # `ranker → prompt_builder → llm`: el LLM ve los 4 documentos rerankeados, no
    # los 15 del joiner. Juzgar contra los 15 rompe las dos métricas en direcciones
    # opuestas: `faithfulness` se vuelve indulgente (una afirmación inventada puede
    # quedar "respaldada" por un documento que el LLM nunca leyó) y
    # `context_relevance` se diluye (promedia 11 documentos que no se usaron).
    # Ver bitacora_refactor_csv.md, hallazgo 15.
    contexts = [d.content for d in result["ranker"]["documents"] if d.content]
    answer   = result.get("llm", {}).get("replies", [""])[0] or ""
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
        "llm_provider":      rag._llm_provider(),
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
    ap = argparse.ArgumentParser(description="Eval harness Tier 3 (juez LLM sobre Groq)")
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
    sys.stdout = _Tee(sys.__stdout__, log_file)

    t0 = time.perf_counter()
    dataset = load_dataset(args.dataset)
    if args.limit:
        dataset = dataset[: args.limit]

    store  = rag.get_document_store()
    valves = rag.Pipeline.Valves(temperature=0.0)
    pipeline = rag.build_rag_pipeline(store, valves)

    # Modelo del juez: --judge-model gana; si no, el mismo que usa el generador activo.
    # En modo ollama, valves.llm_model es un nombre de modelo Groq (no existe en Ollama),
    # así que se resuelve con LLM_MODEL / DEFAULT_OLLAMA_LLM igual que build_generator.
    if args.judge_model:
        judge_model = args.judge_model
    elif rag._llm_provider() == "ollama":
        judge_model = os.getenv("LLM_MODEL") or rag.DEFAULT_OLLAMA_LLM
    else:
        judge_model = os.getenv("LLM_MODEL") or valves.llm_model

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
