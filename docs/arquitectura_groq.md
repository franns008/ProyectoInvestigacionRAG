# Arquitectura del RAG — Generación por API (Groq) + Embeddings locales

Este documento describe la arquitectura del RAG de ciberseguridad tras migrar la
**generación** a la API de **Groq**, manteniendo los **embeddings** locales en Ollama.

## 1. Idea general

El sistema es un RAG (Retrieval-Augmented Generation) sobre documentos de
ciberseguridad. La responsabilidad se reparte en dos planos:

- **Local (dentro de Docker):** ingesta de documentos, embeddings, base vectorial,
  recuperación y orquestación del pipeline.
- **Nube (API):** únicamente la **generación** del texto de respuesta, delegada a
  Groq mediante su endpoint compatible con OpenAI.

> **Decisión clave:** Groq **no ofrece API de embeddings**, así que los embeddings
> siguen calculándose localmente con Ollama (`bge-m3`, 1024 dimensiones). Por eso
> los vectores ya almacenados en pgvector **no requieren re-indexado** al cambiar
> el proveedor de generación.

## 2. Diagrama de componentes

```
                          ┌─────────────────────────────────────────────┐
                          │                  Docker host                 │
                          │                                              │
   Usuario ──HTTP──▶ ┌────┴──────────┐        ┌──────────────────────┐   │
   (navegador)       │  open-webui   │──────▶ │      pipelines       │   │
                     │  (frontend)   │ OpenAI │  (OpenWebUI Pipelines │   │
                     │  :8180        │  API   │   + Haystack RAG)     │   │
                     └────┬──────────┘ compat │   :9099              │   │
                          │                    └───┬───────────┬──────┘   │
                          │                        │           │          │
                          │            embeddings  │           │ retrieval│
                          │                        ▼           ▼          │
                          │                 ┌──────────┐  ┌──────────┐    │
                          │                 │  ollama  │  │   vdb    │    │
                          │                 │ bge-m3   │  │ pgvector │    │
                          │                 │ :11434   │  │ :5433    │    │
                          │                 └──────────┘  └──────────┘    │
                          │                                              │
                          │  ┌──────────┐                                │
                          │  │   db     │  (postgres para OpenWebUI)     │
                          │  │ :5432    │                                │
                          │  └──────────┘                                │
                          └──────────────────────────────────────────────┘
                                          │
                                          │ generación (HTTPS)
                                          ▼
                              ┌───────────────────────────┐
                              │   Groq API (nube)          │
                              │   api.groq.com/openai/v1   │
                              │   llama-3.1-8b-instant     │
                              └───────────────────────────┘
```

## 3. Servicios (docker-compose)

| Servicio     | Imagen                         | Rol                                                        | Puerto |
|--------------|--------------------------------|------------------------------------------------------------|--------|
| `open-webui` | `ghcr.io/open-webui/open-webui`| Frontend tipo ChatGPT; habla con `pipelines` vía API OpenAI| 8180   |
| `pipelines`  | build de `Dockerfile.pipelines`| Servidor OpenWebUI Pipelines que hostea el RAG en Haystack | 9099   |
| `ollama`     | `ollama/ollama`                | Embeddings locales (`bge-m3`)                              | 11434  |
| `vdb`        | `ankane/pgvector`              | Base vectorial (documentos + embeddings)                  | 5433   |
| `db`         | `postgres:15-alpine`           | Persistencia interna de OpenWebUI                         | 5432   |

## 4. El pipeline RAG (Haystack)

Definido en [`src/pipeline/pipeline_ciberseguridad.py`](../src/pipeline/pipeline_ciberseguridad.py).
Es una clase `Pipeline` que el servidor OpenWebUI Pipelines carga y expone como un
"modelo" más.

### 4.1 Componentes del pipeline de consulta

| Componente            | Clase Haystack                 | Función                                              |
|-----------------------|--------------------------------|------------------------------------------------------|
| `text_embedder`       | `OllamaTextEmbedder`           | Embeda la pregunta (bge-m3, local)                   |
| `embedding_retriever` | `PgvectorEmbeddingRetriever`   | Búsqueda por similitud vectorial                     |
| `keyword_retriever`   | `PgvectorKeywordRetriever`     | Búsqueda léxica (BM25/keyword)                       |
| `document_joiner`     | `DocumentJoiner` (RRF)         | Fusiona ambos retrievers (Reciprocal Rank Fusion)    |
| `prompt_builder`      | `PromptBuilder`                | Arma el prompt con el contexto recuperado            |
| `llm`                 | `OpenAIGenerator` → **Groq**   | Genera la respuesta                                  |

La recuperación es **híbrida**: combina retrieval semántico (embeddings) con
retrieval por palabras clave, unificados con Reciprocal Rank Fusion.

### 4.2 Configuración de la generación (Groq)

El componente `llm` usa `OpenAIGenerator` de Haystack apuntado al endpoint
compatible de Groq:

```python
OpenAIGenerator(
    api_key=Secret.from_env_var("GROQ_API_KEY"),
    api_base_url="https://api.groq.com/openai/v1",
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    generation_kwargs={"max_tokens": 512, "temperature": 0.5},
)
```

- Groq expone una API **compatible con OpenAI**, por eso se reutiliza el
  `OpenAIGenerator` estándar de Haystack sin integración específica.
- La API key se lee de la variable de entorno `GROQ_API_KEY` (nunca hardcodeada).
- Parámetros de Groq: `max_tokens` y `temperature`. (Los parámetros de Ollama como
  `num_ctx`/`num_predict` **no aplican** a la API de Groq.)

### 4.3 Valves (parámetros ajustables desde OpenWebUI)

| Valve             | Default               | Descripción                              |
|-------------------|-----------------------|------------------------------------------|
| `llm_model`       | `meta-llama/llama-4-scout-17b-16e-instruct`| Modelo de generación en Groq (TPM 30K)|
| `embedding_model` | `bge-m3`              | Modelo de embeddings en Ollama (1024 dim)|
| `retriever_top_k` | `3`                   | Documentos por retriever                 |
| `max_tokens`      | `512`                 | Longitud máxima de la respuesta          |
| `temperature`     | `0.5`                 | Creatividad de la generación             |
| `split_length`    | `200`                 | Tamaño de chunk (palabras)               |
| `split_overlap`   | `20`                  | Solape entre chunks                      |

## 5. Flujos de datos

### 5.1 Indexación (al arrancar / `on_startup`)

```
PDFs (data/raw) ──marker-pdf──▶ Markdown (caché _converted_md)
                                       │
                                       ▼
                            DocumentSplitter (chunks)
                                       │
                                       ▼
                   OllamaDocumentEmbedder (bge-m3, local)
                                       │
                                       ▼
                         pgvector (vdb)  ← deduplicación por (file_path, split_id)
```

- La conversión PDF→Markdown con **marker-pdf** se cachea en
  `data/raw/_converted_md/`; si el `.md` ya existe, se reutiliza (evita reconvertir).
- La indexación es **incremental**: sólo se embeban y escriben los chunks nuevos.

### 5.2 Consulta (`pipe`)

```
Pregunta del usuario
   │
   ├─▶ OllamaTextEmbedder ──▶ PgvectorEmbeddingRetriever ─┐
   │                                                      ├─▶ DocumentJoiner (RRF)
   └─▶ PgvectorKeywordRetriever ──────────────────────────┘        │
                                                                   ▼
                                                            PromptBuilder
                                                                   │
                                                                   ▼
                                                     OpenAIGenerator → Groq (nube)
                                                                   │
                                                                   ▼
                                                              Respuesta
```

## 6. Configuración y secretos

- Los secretos viven en `infrastructure/.env` (git-ignored). Plantilla en
  [`infrastructure/.env.example`](../infrastructure/.env.example).
- Variables relevantes:
  - `GROQ_API_KEY` — key de Groq ([console.groq.com/keys](https://console.groq.com/keys)),
    inyectada al servicio `pipelines`.
  - `PIPESERVER_KEY` — key compartida entre OpenWebUI y el servidor de pipelines
    (`OPENAI_API_KEY` en open-webui = `PIPELINES_API_KEY` en pipelines).
- OpenWebUI se conecta al servidor de pipelines vía `OPENAI_API_BASE_URL=http://pipelines:9099`.

## 7. Ejecución (CPU por defecto / GPU opcional)

El grupo tiene hardware mixto (AMD y NVIDIA), por lo que el compose base corre
**todo en CPU** (funciona en cualquier máquina) y la GPU NVIDIA es un override opcional.

```bash
cd infrastructure

# CPU (por defecto, cualquier máquina):
docker compose up -d --build

# NVIDIA (requiere nvidia-container-toolkit):
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up -d --build

# Modelo de embeddings (una vez):
docker compose exec ollama ollama pull bge-m3
```

- Como la **generación pesada corre en Groq**, lo único local es embeddar y
  (eventualmente) convertir PDFs — viable en CPU para este proyecto.
- El `Dockerfile.pipelines` usa `ARG TORCH_INDEX_URL` (default CPU; el override
  NVIDIA pasa la build CUDA `cu126`).

### 7.1 El override `docker-compose.nvidia.yml`

**Para qué sirve:** acelera con GPU NVIDIA las tareas locales que corren sobre
PyTorch — los **embeddings** (Ollama con `bge-m3`) y la **conversión de PDFs**
(marker-pdf, que hace OCR/layout). En CPU esas tareas funcionan, pero son lentas
(sobre todo la conversión de PDFs, minutos por archivo); en GPU se reducen a
segundos. **No** afecta a la generación, que siempre va por la API de Groq.

**Por qué se creó (hardware mixto):** el grupo trabaja con máquinas distintas —
algunos integrantes tienen GPU **AMD** (que no usa CUDA) y otros **NVIDIA**. Si el
compose exigiera GPU NVIDIA de forma fija, no arrancaría en las máquinas AMD ni en
las que no tienen `nvidia-container-toolkit`. La solución es separar
responsabilidades:

- **`docker-compose.yml` (base):** corre **todo en CPU**, sin reservas de GPU.
  Funciona en cualquier máquina del grupo sin instalar nada extra.
- **`docker-compose.nvidia.yml` (override):** capa opcional que **sólo** agregan
  quienes tienen NVIDIA. No es un Dockerfile aparte; es un archivo de Compose que
  se fusiona con el base y le suma tres cosas:
  1. Reservas de GPU (`deploy.resources.reservations.devices: driver: nvidia`) en
     los servicios `ollama` y `pipelines` → Docker les pasa la placa.
  2. `TORCH_DEVICE: cuda` en `pipelines` → marker/torch usan la GPU.
  3. El build-arg `TORCH_INDEX_URL=…/cu126` → se buildea la imagen con torch CUDA
     en vez de la de CPU.

> Un único `Dockerfile.pipelines` sirve para ambos casos gracias al
> `ARG TORCH_INDEX_URL`. El override sólo cambia con qué argumento se buildea.

**Cómo se usa:**

```bash
cd infrastructure

# Máquina CPU / AMD / sin toolkit → sólo el base:
docker compose up -d --build

# Máquina NVIDIA (requiere nvidia-container-toolkit instalado en el host):
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up -d --build
```

Al pasar los dos `-f`, Compose **fusiona** ambos archivos: toma todo el base y le
superpone lo del override. Si se omite el segundo `-f`, se corre en CPU. Requisito
para la variante NVIDIA: tener instalado el **nvidia-container-toolkit** en el host;
si falla con `could not select device driver "nvidia"`, es que falta ese paquete.

## 8. Resumen de decisiones de arquitectura

- **Generación en Groq, embeddings locales:** aprovecha la velocidad/gratuidad de
  Groq para el LLM sin re-indexar (Groq no da embeddings).
- **Recuperación híbrida (vector + keyword) con RRF:** mejor cobertura que sólo
  búsqueda semántica.
- **CPU por defecto:** portabilidad sobre todo el grupo; GPU como optimización.
- **Caché de conversión marker-pdf:** evita el costo alto de OCR/layout en CPU.
```
