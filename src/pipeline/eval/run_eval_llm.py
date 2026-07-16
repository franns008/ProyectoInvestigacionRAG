#!/usr/bin/env python3
"""Eval harness — Tier 3: juez LLM (MANUAL, rate-limited).

Mide la calidad de la RESPUESTA con evaluators nativos de Haystack usando Groq
como juez (endpoint compatible con OpenAI, modo JSON):

  - Faithfulness       : ¿las afirmaciones de la respuesta están fundadas en los
                         documentos recuperados? (anti-alucinación)
  - Context Relevance  : ¿los documentos recuperados son relevantes a la pregunta?

Multiplica llamadas a Groq (varias por pregunta) → correr a mano y sobre subsets
(--limit) para no comerse los 429. NO va en cada iteración ni en CI.

    docker compose exec pipelines python /app/pipelines/eval/run_eval_llm.py --limit 8

Ver docs/eval/eval_harness.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

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


def make_judge(model: str):
    """Juez LLM determinístico y en modo JSON. Sigue LLM_PROVIDER (ver docs/modos_llm.md):
    - ollama : OllamaChatGenerator local → Tier-3 corre sin key de Groq (modo GPU).
    - groq   : OpenAIChatGenerator contra el endpoint OpenAI-compatible de Groq.
    """
    if rag._llm_provider() == "ollama":
        return OllamaChatGenerator(
            model=model,
            url=rag.OLLAMA_URL,
            timeout=120,
            generation_kwargs={"format": "json", "temperature": 0},
        )
    return OpenAIChatGenerator(
        api_key=rag.Secret.from_env_var("GROQ_API_KEY"),
        api_base_url=rag.GROQ_BASE_URL,
        model=model,
        generation_kwargs={"response_format": {"type": "json_object"}, "seed": 42, "temperature": 0},
    )


def run_pipeline(pipeline, question: str):
    """Devuelve (contexts: list[str], answer: str) — lo que necesita el juez."""
    result = pipeline.run(
        {
            "text_embedder":     {"text": question},
            "keyword_retriever": {"query": rag.build_keyword_query(question)},
            "prompt_builder":    {"question": question},
        },
        include_outputs_from={"document_joiner"},
    )
    contexts = [d.content for d in result["document_joiner"]["documents"] if d.content]
    answer   = result.get("llm", {}).get("replies", [""])[0] or ""
    return contexts, answer


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval harness Tier 3 (juez LLM sobre Groq)")
    ap.add_argument("--dataset", type=Path, default=EVAL_DIR / "dataset.yaml")
    ap.add_argument("--out", type=Path, default=EVAL_DIR / "results")
    ap.add_argument("--judge-model", type=str, default=None, help="modelo juez (default: el del pipeline)")
    ap.add_argument("--limit", type=int, default=None, help="correr solo las primeras N preguntas (recomendado)")
    args = ap.parse_args()

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

    print(f"Tier 3 — juez={judge_model}  |  {len(dataset)} preguntas\n")

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
    faith = FaithfulnessEvaluator(chat_generator=make_judge(judge_model),
                                  raise_on_failure=False, progress_bar=False)
    fres = faith.run(questions=questions, contexts=contexts, predicted_answers=answers)

    print("  Juzgando context relevance...")
    ctxrel = ContextRelevanceEvaluator(chat_generator=make_judge(judge_model),
                                       raise_on_failure=False, progress_bar=False)
    cres = ctxrel.run(questions=questions, contexts=contexts)

    f_ind = fres.get("individual_scores", [])
    c_ind = cres.get("individual_scores", [])

    print("\n" + "=" * 64)
    print("TIER 3 — JUEZ LLM (por pregunta)")
    print("=" * 64)
    per_question = []
    for i, qid in enumerate(ids):
        f = f_ind[i] if i < len(f_ind) else None
        c = c_ind[i] if i < len(c_ind) else None
        per_question.append({"id": qid, "faithfulness": f, "context_relevance": c})
        fs = f"{f:.2f}" if isinstance(f, (int, float)) else "n/a"
        cs = f"{c:.2f}" if isinstance(c, (int, float)) else "n/a"
        print(f"  {qid:<28} faithfulness={fs}  context_relevance={cs}")

    print("\n" + "=" * 64)
    print(f"  GLOBAL:  faithfulness={fres.get('score')}   context_relevance={cres.get('score')}")
    print("=" * 64)

    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "judge_model": judge_model,
        "faithfulness_overall": fres.get("score"),
        "context_relevance_overall": cres.get("score"),
        "per_question": per_question,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out / f"tier3_{stamp}.json"
    out_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultados guardados en {out_path}")


if __name__ == "__main__":
    main()
