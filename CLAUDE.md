# CLAUDE.md

Guía rápida del repo para asistentes y contribuidores. RAG de ciberseguridad sobre
**Haystack** + **OpenWebUI Pipelines** + **pgvector**. Todo local sobre Ollama:
embeddings (`qwen3-embedding:4b`) y generación.

## Documentación crucial (leer primero)

- **[`docs/arquitectura.md`](docs/arquitectura.md) — CRUCIAL.** Arquitectura general,
  puesta en marcha y ejecución (CPU por defecto / GPU Nvidia opcional). Leer antes de
  tocar la generación, el arranque de la infra o el eval.
- [`docs/eval/`](docs/eval/) — todo sobre el harness de evaluación:
  [`eval_harness.md`](docs/eval/eval_harness.md) (diseño vigente, Tiers 1-3) y
  [`mejoras_harness.md`](docs/eval/mejoras_harness.md) (plan de mejoras priorizado, propuesto).
- [`docs/escaneo_dependencias_setup.md`](docs/escaneo_dependencias_setup.md) — **puesta en
  marcha** del escaneo de `requirements.txt` y su extensión de VSCode en una máquina nueva.
  No necesita el stack: ni Docker, ni pgvector, ni LLM. El diseño está en
  [`escaneo_dependencias.md`](docs/escaneo_dependencias.md) y el guion de la demo en
  [`escaneo_dependencias_demo.md`](docs/escaneo_dependencias_demo.md).
  **El escaneo vive en la rama `feature/requirements`, no en `main`.** Un clon nuevo cae
  en `main`, que todavía no lo tiene: hay que hacer `git checkout feature/requirements`.
- [`docs/ingestion_fetchers.md`](docs/ingestion_fetchers.md) — cómo bajar los datos crudos
  (OSV, EPSS, KEV, CWE, NVD) con `src/ingestion/fetch_all.py`. El detalle de NVD, que es el
  único con API key y checkpoint, está en
  [`ingestion_nvd_setup.md`](docs/ingestion_nvd_setup.md).
- [`docs/explicacion_hallazgos.md`](docs/explicacion_hallazgos.md) — cómo se explican los 3 primeros hallazgos del escaneo trayendo el CVE/CWE del
  corpus **por clave exacta** (filtro por metadata, sin retrieval) y mandándoselo al LLM como
  contexto. **Implementado**; incluye la medición de qué hay realmente en los datos.
- [`docs/reranker_cross_encoder.md`](docs/reranker_cross_encoder.md) — reranker cross-encoder
  (`bge-reranker-v2-m3`) tras el retrieval híbrido: retrieve-and-rerank, implicancias (modelo
  local en CPU, latencia) y cambios en pipeline/Dockerfile/eval.

## Mapa rápido

- `src/pipeline/pipeline_ciberseguridad.py` — el RAG (pipeline OpenWebUI). La
  generación la arma `build_generator(valves)` sobre el Ollama local.
- `src/pipeline/eval/` — harness de evaluación (corre dentro del container `pipelines`).
- `src/pipeline/deps/` — escaneo de dependencias (resolver + priorización + CLI). Lógica
  pura: no importa Haystack ni toca la base. Se testea con `pytest` sin levantar nada.
- `src/pipeline/explain/` — explicación de los 3 primeros hallazgos: trae el CVE/CWE del
  corpus **por clave exacta** (sin retrieval) y arma el prompt. Recibe el store y el
  generador inyectados, así que también se testea sin stack. Lo expone
  `pipeline_dependencias.py` como segundo modelo del `:9099`.
- `extension/` — extensión de VSCode que muestra el escaneo. Cliente del `deps.cli` hoy,
  del pipeline RAG cuando exista (`src/scan/types.ts` es el contrato compartido).
- `infrastructure/` — stack Docker Compose. `docker-compose.yml` (base, CPU) +
  `docker-compose.nvidia.yml` (overlay GPU NVIDIA, vía CUDA) +
  `docker-compose.amd.yml` (overlay GPU AMD, vía Vulkan; sólo acelera Ollama).
  Config en `.env` (plantilla `.env.example`).

## Convenciones

- **No** inspeccionar archivos `.env` reales ni sus valores.
- Los embeddings (`bge-m3`) son siempre locales y no cambian entre modos de LLM.
