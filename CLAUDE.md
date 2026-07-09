# CLAUDE.md

Guía rápida del repo para asistentes y contribuidores. RAG de ciberseguridad sobre
**Haystack** + **OpenWebUI Pipelines** + **pgvector**. Embeddings locales (`bge-m3`
en Ollama); generación intercambiable (Groq API o Ollama local).

## Documentación crucial (leer primero)

- **[`docs/modos_llm.md`](docs/modos_llm.md) — CRUCIAL.** Cómo el repo levanta la
  generación en dos modos (API Key Groq ↔ Ollama local/Nvidia) con una sola variable
  `LLM_PROVIDER`. Leer antes de tocar la generación, el arranque de la infra o el eval.
- [`docs/arquitectura_groq.md`](docs/arquitectura_groq.md) — arquitectura general y
  ejecución (CPU por defecto / GPU Nvidia opcional).
- [`docs/eval_harness.md`](docs/eval_harness.md) — harness de evaluación (Tiers 1-3).

## Mapa rápido

- `src/pipeline/pipeline_ciberseguridad.py` — el RAG (pipeline OpenWebUI). La
  generación la arma `build_generator(valves)` según `LLM_PROVIDER`.
- `src/pipeline/eval/` — harness de evaluación (corre dentro del container `pipelines`).
- `infrastructure/` — stack Docker Compose. `docker-compose.yml` (base, CPU) +
  `docker-compose.nvidia.yml` (overlay GPU). Config en `.env` (plantilla `.env.example`).

## Convenciones

- **No** inspeccionar archivos `.env` reales ni sus valores.
- Los embeddings (`bge-m3`) son siempre locales y no cambian entre modos de LLM.
