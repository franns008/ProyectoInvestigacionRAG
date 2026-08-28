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
- [`docs/eval/`](docs/eval/) — todo sobre el harness de evaluación:
  [`eval_harness.md`](docs/eval/eval_harness.md) (diseño vigente, Tiers 1-3) y
  [`mejoras_harness.md`](docs/eval/mejoras_harness.md) (plan de mejoras priorizado, propuesto).
- [`docs/escaneo_dependencias_setup.md`](docs/escaneo_dependencias_setup.md) — **puesta en
  marcha** del escaneo de `requirements.txt` y su extensión de VSCode en una máquina nueva.
  No necesita el stack: ni Docker, ni pgvector, ni LLM. El diseño está en
  [`escaneo_dependencias.md`](docs/escaneo_dependencias.md).
- [`docs/reranker_cross_encoder.md`](docs/reranker_cross_encoder.md) — reranker cross-encoder
  (`bge-reranker-v2-m3`) tras el retrieval híbrido: retrieve-and-rerank, implicancias (modelo
  local en CPU, latencia) y cambios en pipeline/Dockerfile/eval.

## Mapa rápido

- `src/pipeline/pipeline_ciberseguridad.py` — el RAG (pipeline OpenWebUI). La
  generación la arma `build_generator(valves)` según `LLM_PROVIDER`.
- `src/pipeline/eval/` — harness de evaluación (corre dentro del container `pipelines`).
- `src/pipeline/deps/` — escaneo de dependencias (resolver + priorización + CLI). Lógica
  pura: no importa Haystack ni toca la base. Se testea con `pytest` sin levantar nada.
- `extension/` — extensión de VSCode que muestra el escaneo. Cliente del `deps.cli` hoy,
  del pipeline RAG cuando exista (`src/scan/types.ts` es el contrato compartido).
- `infrastructure/` — stack Docker Compose. `docker-compose.yml` (base, CPU) +
  `docker-compose.nvidia.yml` (overlay GPU). Config en `.env` (plantilla `.env.example`).

## Convenciones

- **No** inspeccionar archivos `.env` reales ni sus valores.
- Los embeddings (`bge-m3`) son siempre locales y no cambian entre modos de LLM.
