#!/usr/bin/env python3
"""Experimento de chunking: indexa el corpus con cada estrategia candidata en una
tabla pgvector SEPARADA y corre el eval contra cada una, para comparar estrategias
de forma controlada sin tocar la tabla de producción (`ciberseguridad_docs`).

Cada estrategia se materializa en `ciberseguridad_docs_chunk_<estrategia>`. Los
documentos atómicos (CWE, y opcionalmente CVE) NO cambian entre estrategias — se
embeben una sola vez y se reutilizan en cada tabla; solo se re-chunkea el corpus
splittable (guías INCIBE). Ver docs/data_splitting.md.

Retrieval puro: arma el pipeline SIN LLM (build_rag_pipeline(include_llm=False)), así
NO exige GROQ_API_KEY ni paga la latencia de generación. El chunking se juzga por
retrieval, que es lo que mueve; la generación (SAS) se evalúa aparte con run_eval.py.

Ejecutar como container EFÍMERO que saltea el server de Open WebUI (para NO disparar la
auto-indexación de producción, que embebería los ~76k CVE de NVD). En Fedora/SELinux los
bind-mounts necesitan `:z`. NO se monta el volumen marker_cache (el reranker ya está en
la imagen). Desde infrastructure/:

    docker run --rm --network infrastructure_default -e TORCH_DEVICE=cpu \
      -v "$(cd .. && pwd)/src/pipeline":/app/pipelines:z \
      -v "$(cd .. && pwd)/data/raw":/app/pipelines/rawdata:z \
      --entrypoint python infrastructure-pipelines:latest \
      /app/pipelines/eval/run_chunking_experiment.py
      # opcional: --strategies current_word200 recursive_500   |   --include-cve

Métricas por estrategia (retrieval puro, sin LLM):
  · Tier 1  (recall@k por doc-id)  — solo válido para preguntas CWE (id estable).
  · Tier 1b (source_recall@k)      — el eje que mide el chunking del corpus splittable.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

import pipeline_ciberseguridad as rag  # noqa: E402
import chunking_strategies as cstrat  # noqa: E402
import metrics as m  # noqa: E402
import run_eval as re_  # noqa: E402
from haystack import Document  # noqa: E402
from haystack.components.converters import (  # noqa: E402
    DOCXToDocument, MarkdownToDocument, TextFileToDocument,
)
from haystack.document_stores.types import DuplicatePolicy  # noqa: E402


def load_splittable_corpus() -> tuple[list[Document], list[Document]]:
    """Devuelve (rendered, raw_markdown) del corpus splittable.

    - rendered:     texto pasado por MarkdownToDocument (headers '#' renderizados a
                    texto plano). Es lo que consume el pipeline de producción.
    - raw_markdown: el markdown crudo (sintaxis '#' intacta), necesario para
                    MarkdownHeaderSplitter (needs_raw_markdown=True). Ver
                    docs/data_splitting.md §3.8.

    Ambas listas comparten el mismo meta.source (nombre del archivo original), que es
    el ground truth de Tier 1b.
    """
    input_dir = rag.INPUT_DIR
    converted_dir = rag.CONVERTED_DIR

    rendered: list[Document] = []
    raw_markdown: list[Document] = []

    # PDFs → markdown cacheado en _converted_md (marker no se re-corre en el experimento).
    for pdf_path in sorted(input_dir.glob("*.pdf")):
        md_path = converted_dir / (pdf_path.stem + ".md")
        if not md_path.exists():
            print(f"  [warn] sin conversión cacheada para {pdf_path.name}; se saltea "
                  f"(correr el pipeline una vez para generar {md_path.name})")
            continue
        text = md_path.read_text(encoding="utf-8")
        meta = {"file_path": str(pdf_path), "source": pdf_path.name}
        rendered.extend(_render_markdown(md_path, meta))
        raw_markdown.append(Document(content=text, meta=dict(meta)))

    # DOCX / MD nativo / TXT en INPUT_DIR (no en _converted_md).
    simple = [
        ("*.docx", DOCXToDocument()),
        ("*.md",   MarkdownToDocument()),
        ("*.txt",  TextFileToDocument()),
    ]
    for pattern, converter in simple:
        files = [f for f in input_dir.glob(pattern) if f.parent == input_dir]
        if not files:
            continue
        for f in files:
            docs = converter.run(sources=[f])["documents"]
            for d in docs:
                d.meta.setdefault("source", f.name)
            rendered.extend(docs)
            # Para el raw path: .md se usa crudo; el resto no tiene headers markdown → texto plano.
            if pattern == "*.md":
                raw_markdown.append(Document(content=f.read_text(encoding="utf-8"),
                                             meta={"source": f.name, "file_path": str(f)}))
            else:
                raw_markdown.extend([Document(content=d.content, meta=dict(d.meta)) for d in docs])

    return rendered, raw_markdown


def _render_markdown(md_path: Path, meta: dict) -> list[Document]:
    docs = MarkdownToDocument().run(sources=[md_path])["documents"]
    for d in docs:
        d.meta.update(meta)
    return docs


def load_atomic_corpus(include_cve: bool = False) -> list[Document]:
    """CWE (XML) + opcionalmente CVE (JSON) → Documents atómicos, idénticos entre
    estrategias.

    CVE está OFF por defecto: los CVE son atómicos (no se chunkean), así que no
    aportan nada a la comparación de chunking, y el corpus NVD son ~76k CVEs cuyo
    embedding local en CPU tarda horas. Las preguntas CWE (que sí controlan el
    recall por doc-id) sólo necesitan el catálogo CWE. Ver docs/data_splitting.md.
    """
    input_dir = rag.INPUT_DIR
    atomic: list[Document] = []
    xml_files = [f for f in input_dir.glob("*.xml") if f.parent == input_dir]
    if xml_files:
        atomic.extend(rag.XMLCWEConverter().run(sources=xml_files)["documents"])
    if include_cve:
        cve_files = sorted(input_dir.rglob("cves_page_*.json"))
        if cve_files:
            atomic.extend(rag.NVDJsonConverter().run(sources=cve_files)["documents"])
    # Dedup por id (mismo criterio que el pipeline).
    seen, out = set(), []
    for d in atomic:
        if d.id in seen:
            continue
        seen.add(d.id)
        out.append(d)
    return out


def chunk_for_strategy(strat: cstrat.ChunkingStrategy,
                       rendered: list[Document],
                       raw_markdown: list[Document]) -> list[Document]:
    source = raw_markdown if strat.needs_raw_markdown else rendered
    splitter = strat.build()
    chunks = splitter.run(documents=[Document(content=d.content, meta=dict(d.meta)) for d in source])["documents"]
    if strat.needs_raw_markdown:
        chunks = [cstrat.prepend_header_context(c) for c in chunks]
    return chunks


def index_strategy(table: str, chunks: list[Document], embedded_atomic: list[Document],
                   embedding_model: str) -> rag.PgvectorDocumentStore:
    """Crea/recrea la tabla del experimento y escribe chunks (recién embebidos) +
    atómicos (embeddings reutilizados). Devuelve el store."""
    store = rag.get_document_store(
        table_name=table,
        keyword_index_name=f"{table}_kw_idx",
    )
    # Vaciar la tabla para una corrida limpia (idempotente entre re-ejecuciones).
    existing = store.filter_documents()
    if existing:
        store.delete_documents([d.id for d in existing])

    embedder = rag.OllamaDocumentEmbedder(model=embedding_model, url=rag.OLLAMA_URL, batch_size=32)
    embedded_chunks = embedder.run(chunks)["documents"] if chunks else []
    store.write_documents(embedded_chunks + embedded_atomic, policy=DuplicatePolicy.OVERWRITE)
    return store


def retrieval_only_question(pipeline, question: str):
    """Corre el pipeline SIN LLM (hasta el reranker) y devuelve
    (emb_ids, kw_ids, ranked_ids, ranked_sources). Ver build_rag_pipeline(include_llm=False)."""
    result = pipeline.run(
        {
            "text_embedder":     {"text": question},
            "keyword_retriever": {"query": rag.build_keyword_query(question)},
            "ranker":            {"query": question},
        },
        include_outputs_from={"embedding_retriever", "keyword_retriever", "document_joiner", "ranker"},
    )
    ranked_docs = result["ranker"]["documents"]
    return (
        [d.id for d in result["embedding_retriever"]["documents"]],
        [d.id for d in result["keyword_retriever"]["documents"]],
        [d.id for d in ranked_docs],
        [d.meta.get("source") for d in ranked_docs],
    )


def run_eval_on_store(store, dataset: list[dict], valves) -> list[dict]:
    # Retrieval puro: sin LLM (no exige GROQ_API_KEY ni paga latencia de generación).
    # El chunking se juzga por recall@k (doc-id CWE) y source_recall (fuente de las guías).
    pipeline = rag.build_rag_pipeline(store, valves, include_llm=False)
    per_question: list[dict] = []
    for item in dataset:
        try:
            emb, kw, joined, joined_sources = retrieval_only_question(pipeline, item["question"])
            rec = re_.evaluate(item, emb, kw, joined, joined_sources, answer=None)
        except Exception as e:  # noqa: BLE001
            rec = {"id": item["id"], "category": item.get("category", "?"),
                   "recall": None, "source_recall": None,
                   "status": "error", "error": repr(e)}
        per_question.append(rec)
    return per_question


def summarize(per_question: list[dict]) -> dict:
    return {
        "retrieval":  m.aggregate_retrieval(per_question),
        "source":     m.aggregate_source_retrieval(per_question),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Experimento comparativo de estrategias de chunking")
    ap.add_argument("--dataset", type=Path, default=EVAL_DIR / "dataset.yaml")
    ap.add_argument("--out", type=Path, default=EVAL_DIR / "results")
    ap.add_argument("--strategies", nargs="*", default=list(cstrat.STRATEGIES),
                    help="subconjunto de estrategias a comparar (default: todas)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--include-cve", action="store_true",
                    help="indexar también los ~76k CVE de NVD (lento; irrelevante para chunking)")
    args = ap.parse_args()

    unknown = [s for s in args.strategies if s not in cstrat.STRATEGIES]
    if unknown:
        ap.error(f"estrategias desconocidas: {unknown}. Disponibles: {list(cstrat.STRATEGIES)}")

    args.out.mkdir(parents=True, exist_ok=True)
    dataset = yaml.safe_load(args.dataset.read_text(encoding="utf-8"))
    if args.limit:
        dataset = dataset[: args.limit]

    valves = rag.Pipeline.Valves(temperature=args.temperature)

    print("Cargando corpus...")
    rendered, raw_markdown = load_splittable_corpus()
    atomic = load_atomic_corpus(include_cve=args.include_cve)
    print(f"  splittable: {len(rendered)} docs (rendered) / {len(raw_markdown)} (raw md)  |  atómicos: {len(atomic)}")

    print("Embebiendo documentos atómicos una sola vez (se reutilizan por estrategia)...")
    embedded_atomic = (rag.OllamaDocumentEmbedder(model=valves.embedding_model, url=rag.OLLAMA_URL, batch_size=32)
                       .run(atomic)["documents"]) if atomic else []

    results: dict[str, dict] = {}
    for name in args.strategies:
        strat = cstrat.STRATEGIES[name]
        table = f"ciberseguridad_docs_chunk_{name}"
        print(f"\n=== Estrategia: {name} ===")
        print(f"  {strat.description}")
        chunks = chunk_for_strategy(strat, rendered, raw_markdown)
        words = [len(c.content.split()) for c in chunks]
        avg = sum(words) / len(words) if words else 0
        print(f"  chunks={len(chunks)}  avg_words={avg:.1f}  → tabla {table}")
        store = index_strategy(table, chunks, embedded_atomic, valves.embedding_model)
        per_question = run_eval_on_store(store, dataset, valves)
        summary = summarize(per_question)
        summary["n_chunks"] = len(chunks)
        summary["avg_words"] = round(avg, 1)
        results[name] = {"summary": summary, "per_question": per_question}

    _print_comparison(results)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out / f"chunking_experiment_{stamp}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultados persistidos en {out_path}")


def _print_comparison(results: dict[str, dict]) -> None:
    print("\n" + "=" * 84)
    print("COMPARACIÓN DE ESTRATEGIAS DE CHUNKING")
    print("=" * 84)
    hdr = f"{'estrategia':<18} {'chunks':>7} {'avg_w':>7} {'recall@k':>9} {'src_recall':>11}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        s = r["summary"]
        rec = s["retrieval"].get("recall_at_k")
        src = s["source"].get("source_recall_at_k")
        print(f"{name:<18} {s['n_chunks']:>7} {s['avg_words']:>7.1f} "
              f"{_fmt(rec):>9} {_fmt(src):>11}")
    print("\nLeyenda (retrieval puro, sin LLM): recall@k = doc-id (CWE); "
          "src_recall = fuente (guías INCIBE).")


def _fmt(x: float | None) -> str:
    return f"{x:.3f}" if x is not None else "  n/a"


if __name__ == "__main__":
    main()
