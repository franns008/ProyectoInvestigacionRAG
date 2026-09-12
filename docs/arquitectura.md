# Arquitectura del RAG — todo local sobre Ollama

Este documento describe la arquitectura del RAG de ciberseguridad. **Todo corre
localmente**: embeddings y generación en Ollama, base vectorial en pgvector, y el
reranker cross-encoder en el propio container de `pipelines`. No hay APIs externas
ni claves que gestionar.

> **Nota histórica.** Hasta 2026-09 la generación se delegaba a la API de Groq y el
> repo soportaba los dos modos mediante `LLM_PROVIDER`. Groq se removió: quedó un
> solo camino. El registro de esa etapa vive en [`eval/`](eval/) (bitácoras) y en las
> filas viejas de `runs.csv`.

## 1. Idea general

El sistema es un RAG (Retrieval-Augmented Generation) sobre documentos de
ciberseguridad. Todo el cómputo ocurre dentro de Docker:

- **Ingesta y recuperación:** embeddings, base vectorial, retrieval híbrido y
  reranking.
- **Generación:** un modelo de Ollama produce la respuesta a partir del contexto
  recuperado.

> **Implicancia práctica:** el modelo de generación se cambia con una variable
> (`LLM_MODEL`) y un `ollama pull`, sin re-indexar. Los vectores ya almacenados en
> pgvector dependen del modelo de **embeddings**, que es un eje independiente:
> cambiar ese sí obliga a reindexar.

## 2. Diagrama de componentes

```
                          ┌──────────────────────────────────────────────┐
                          │                 Docker host                  │
                          │                                              │
   Usuario ──HTTP──▶ ┌────┴──────────┐        ┌──────────────────────┐   │
   (navegador)       │  open-webui   │──────▶ │      pipelines       │   │
                     │  (frontend)   │ OpenAI │ (OpenWebUI Pipelines │   │
                     │  :8180        │  API   │  + Haystack RAG      │   │
                     └────┬──────────┘ compat │  + reranker en CPU)  │   │
                          │                   │  :9099               │   │
                          │                   └───┬───────────┬──────┘   │
                          │                       │           │          │
                          │   embeddings +        │           │ retrieval│
                          │   generación          ▼           ▼          │
                          │              ┌────────────────┐  ┌─────────┐ │
                          │              │     ollama     │  │   vdb   │ │
                          │              │ qwen3-embedding│  │pgvector │ │
                          │              │ + LLM de gen.  │  │ :5433   │ │
                          │              │ :11434         │  └─────────┘ │
                          │              └────────────────┘              │
                          │                                              │
                          │  ┌──────────┐                                │
                          │  │   db     │  (postgres para OpenWebUI)     │
                          │  │ :5432    │                                │
                          │  └──────────┘                                │
                          └──────────────────────────────────────────────┘
```

## 3. Servicios (docker-compose)

| Servicio     | Imagen                         | Rol                                                        | Puerto |
|--------------|--------------------------------|------------------------------------------------------------|--------|
| `open-webui` | `ghcr.io/open-webui/open-webui`| Frontend tipo ChatGPT; habla con `pipelines` vía API OpenAI| 8180   |
| `pipelines`  | build de `Dockerfile.pipelines`| Servidor OpenWebUI Pipelines que hostea el RAG en Haystack | 9099   |
| `ollama`     | `ollama/ollama`                | Embeddings **y** generación, ambos locales                 | 11434  |
| `vdb`        | `ankane/pgvector`              | Base vectorial (documentos + embeddings)                  | 5433   |
| `db`         | `postgres:15-alpine`           | Persistencia interna de OpenWebUI                         | 5432   |

## 4. El pipeline RAG (Haystack)

Definido en [`src/pipeline/pipeline_ciberseguridad.py`](../src/pipeline/pipeline_ciberseguridad.py).
Es una clase `Pipeline` que el servidor OpenWebUI Pipelines carga y expone como un
"modelo" más.

### 4.1 Componentes del pipeline de consulta

| Componente            | Clase Haystack                 | Función                                              |
|-----------------------|--------------------------------|------------------------------------------------------|
| `text_embedder`       | `OllamaTextEmbedder`           | Embeda la pregunta (local)                           |
| `embedding_retriever` | `PgvectorEmbeddingRetriever`   | Búsqueda por similitud vectorial                     |
| `keyword_retriever`   | `PgvectorKeywordRetriever`     | Búsqueda léxica (BM25/keyword)                       |
| `id_lookup`           | `VulnIdLookup`                 | Canal determinístico por CWE/CVE en metadata         |
| `document_joiner`     | `DocumentJoiner` (RRF)         | Fusiona los tres canales (Reciprocal Rank Fusion)    |
| `ranker`              | `SentenceTransformersSimilarityRanker` | Reranker cross-encoder (local, CPU)          |
| `prompt_builder`      | `PromptBuilder`                | Arma el prompt con el contexto recuperado            |
| `llm`                 | `OllamaGenerator` → **local**  | Genera la respuesta                                  |

La recuperación es **híbrida**: combina retrieval semántico (embeddings), retrieval
por palabras clave y un canal determinístico por identificador, unificados con
Reciprocal Rank Fusion y reordenados por el cross-encoder. Ver
[`reranker_cross_encoder.md`](reranker_cross_encoder.md).

### 4.2 Configuración de la generación

El componente `llm` usa el `OllamaGenerator` de Haystack contra el servicio `ollama`
del compose:

```python
OllamaGenerator(
    model=model,                       # LLM_MODEL, o el valve llm_model
    url="http://ollama:11434",
    timeout=120,
    generation_kwargs={"num_predict": 512, "temperature": 0.5},
)
```

- El modelo se resuelve con la precedencia `LLM_MODEL` (env) **>** valve `llm_model`
  (cuyo default es `DEFAULT_OLLAMA_LLM`, hoy `qwen2.5:3b-instruct`). La variable de
  entorno gana para poder cambiar de modelo sin tocar la UI.
- Cualquier modelo que se configure tiene que estar **`ollama pull`-eado** en el
  container de Ollama, o la generación falla.
- Parámetros de Ollama: `num_predict` (longitud de la respuesta) y `temperature`.

### 4.3 Valves (parámetros ajustables desde OpenWebUI)

| Valve             | Default                    | Descripción                                   |
|-------------------|----------------------------|-----------------------------------------------|
| `llm_model`       | `qwen2.5:3b-instruct`      | Modelo de generación en Ollama (`LLM_MODEL` le gana) |
| `embedding_model` | `qwen3-embedding:4b`       | Modelo de embeddings en Ollama (2560 dim)     |
| `retriever_top_k` | `15`                       | Documentos por retriever                      |
| `ranker_model`    | `BAAI/bge-reranker-v2-m3`  | Cross-encoder del reranking                   |
| `ranker_top_k`    | `4`                        | Documentos que llegan al LLM tras rerankear   |
| `id_lookup_top_k` | `10`                       | Tope del canal determinístico por CWE/CVE     |
| `max_tokens`      | `512`                      | Longitud máxima de la respuesta               |
| `temperature`     | `0.5`                      | Creatividad de la generación                  |

> `split_length` / `split_overlap` ya no son valves: el chunking se decide en el
> script de indexación. Ver [`data_splitting.md`](data_splitting.md).

## 5. Flujos de datos

### 5.1 Indexación

```
PDFs (data/raw) ──marker-pdf──▶ Markdown (caché _converted_md)
                                       │
                                       ▼
                            DocumentSplitter (chunks)
                                       │
                                       ▼
                    OllamaDocumentEmbedder (local)
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
   │                                                      │
   ├─▶ PgvectorKeywordRetriever ───────────────────────────┼─▶ DocumentJoiner (RRF)
   │                                                      │        │
   └─▶ VulnIdLookup (CWE/CVE por metadata) ────────────────┘        ▼
                                                            Ranker (cross-encoder)
                                                                   │
                                                                   ▼
                                                            PromptBuilder
                                                                   │
                                                                   ▼
                                                     OllamaGenerator (local)
                                                                   │
                                                                   ▼
                                                              Respuesta
```

## 6. Configuración y secretos

- La config vive en `infrastructure/.env` (git-ignored). Plantilla en
  [`infrastructure/.env.example`](../infrastructure/.env.example).
- **No hay claves de API que gestionar**: nada del RAG sale a internet en runtime.
  Las únicas credenciales son internas del stack.
- Variables relevantes:
  - `LLM_MODEL` — modelo de generación en Ollama. Vacío = el default del pipeline.
  - `PIPESERVER_KEY` — key compartida entre OpenWebUI y el servidor de pipelines
    (`OPENAI_API_KEY` en open-webui = `PIPELINES_API_KEY` en pipelines).
- OpenWebUI se conecta al servidor de pipelines vía `OPENAI_API_BASE_URL=http://pipelines:9099`.

## 7. Puesta en marcha

Todos los comandos se corren **parados en `infrastructure/`**:

```bash
cd infrastructure
```

### 7.1 Paso a paso

1. **Crear el `.env`** (si no existe):
   ```bash
   cp .env.example .env
   ```
   Para un arranque de desarrollo no hace falta completar nada: todo tiene defaults
   en el compose.

2. **Levantar el stack** (la primera vez buildea la imagen, tarda):
   ```bash
   # CPU (cualquier máquina):
   docker compose up -d --build

   # NVIDIA (requiere nvidia-container-toolkit en el host):
   docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up -d --build

   # AMD sin ROCm, vía Vulkan (requiere Mesa/RADV en el host):
   docker compose -f docker-compose.yml -f docker-compose.amd.yml up -d --build
   ```

3. **Descargar los modelos** en Ollama (una sola vez): embeddings **y** el LLM de
   generación.
   ```bash
   docker compose exec ollama ollama pull qwen3-embedding:4b
   docker compose exec ollama ollama pull qwen2.5:3b-instruct
   ```
   > Si configuraste otro `LLM_MODEL` en el `.env`, descargá **ese** en vez de
   > `qwen2.5:3b-instruct`.

4. **Usar el RAG**: abrí OpenWebUI en `http://localhost:8180` y elegí el modelo
   *"RAG ciberseguridad"* en el selector.

### 7.2 Operación

- **Cambiar el modelo de generación**: editá `LLM_MODEL` en `.env`, asegurate de
  haberlo pulleado, y **recreá** el stack (`docker compose up -d`). Las env vars se
  leen al crear el container: editar el `.env` a secas no alcanza.
- **Ver logs de la generación**: `docker compose logs -f pipelines` o el archivo
  `src/pipeline/logCiberseguridad.txt`.
- **Parar** todo sin borrar datos: `docker compose down`.
- **Reset total** (borra la base y hay que reindexar): `docker compose down -v`.

> **Notas**
> - "Una sola vez" = el `ollama pull` persiste en el volumen `ollama_data`; no hay
>   que repetirlo salvo que borres el volumen.
> - `ollama` sin GPU funciona, pero la generación es lenta. Si no tenés GPU, usá un
>   modelo chico vía `LLM_MODEL`.
> - Si `8180` está ocupado, cambiá `OWEBUI_PORT` en `.env` (ver puertos opcionales).

### 7.3 Troubleshooting

- **Respuesta vacía / error de modelo** → falta `ollama pull <LLM_MODEL>`.
- **Error de modelo inexistente con un nombre que no configuraste** (p. ej.
  `meta-llama/llama-4-...`) → OpenWebUI **persiste los valves** en
  `<volumen pipelines>/<nombre_pipeline>/valves.json` y los restaura al arrancar, así
  que puede estar reinyectando el modelo que se usaba en la etapa Groq. Se resuelve
  seteando `LLM_MODEL` en el `.env` (le gana al valve) o borrando ese `valves.json`.
- **Cambiaste `.env` y no toma efecto** → recreá el stack (`docker compose up -d`).
- **El pipeline aparece en `failed/`** → OpenWebUI no pudo cargar el `.py` (suele ser
  una versión de dependencia). Ver el hallazgo 17 de
  [`eval/bitacora_refactor_csv.md`](eval/bitacora_refactor_csv.md).
- **Generación muy lenta** → estás en CPU (sin el overlay de GPU que corresponda a tu
  placa) o el modelo no entra en VRAM; probá uno más chico vía `LLM_MODEL`.
- **`vulkaninfo` no lista tu placa AMD en el host** → falta el driver Vulkan de Mesa
  (`mesa-vulkan-drivers`). Ojo: `vulkaninfo --summary | grep -A3 GPU0` **no** alcanza
  para verificar, corta antes de `deviceName`; usá
  `vulkaninfo --summary | grep -E 'deviceName|driverName'`. Los warnings del loader
  sobre `libvulkan_dzn.so` son ruido esperado (ICD Dozen, no aplica en Linux).
- **Levantaste con el overlay AMD y Ollama sigue en CPU** → revisá
  `docker compose logs ollama | grep -i vulkan`. Si no hay mención de Vulkan, la imagen
  `ollama/ollama` es anterior a 0.12.11 (`docker compose exec ollama ollama --version`);
  si la hay pero no encuentra device, faltan los `/dev/dri` en el container.
- **`VK_ERROR_DEVICE_LOST` con una GPU Polaris** → bug conocido del backend Vulkan en
  gfx803. El fallback es correr sin el overlay (CPU).
- **Levantaste un solo servicio y perdiste la GPU** → `docker compose up -d pipelines`
  arrastra `ollama` por `depends_on` y lo recrea **sin** el overlay. Restaurarlo
  pasando los dos `-f` y apuntando sólo a `ollama`. Ver el detalle en
  [`eval/epoch_qwen3_reranker_v1.md`](eval/epoch_qwen3_reranker_v1.md).

## 8. Ejecución en CPU vs GPU

El grupo tiene hardware mixto (AMD y NVIDIA), por lo que el compose base corre
**todo en CPU** (funciona en cualquier máquina) y cada familia de GPU tiene su override
opcional: `docker-compose.nvidia.yml` (§8.1) y `docker-compose.amd.yml` (§8.2).

- El `Dockerfile.pipelines` usa `ARG TORCH_INDEX_URL` (default CPU; el override
  NVIDIA pasa la build CUDA `cu126`).
- Ahora que la generación también es local, la GPU pesa **más** que antes: acelera
  embeddings, reranker, conversión de PDFs **y** generación.
- Los dos overrides **no** cubren lo mismo: el de NVIDIA acelera todo lo anterior,
  mientras que el de AMD acelera **sólo Ollama** (generación y embeddings). El motivo
  está en §8.2.

### 8.1 El override `docker-compose.nvidia.yml`

**Para qué sirve:** acelera con GPU NVIDIA las tareas locales — **generación** y
**embeddings** (Ollama), el **reranker** cross-encoder y la **conversión de PDFs**
(marker-pdf, que hace OCR/layout). En CPU todo eso funciona, pero es lento.

**Por qué se creó (hardware mixto):** el grupo trabaja con máquinas distintas —
algunos integrantes tienen GPU **AMD** (que no usa CUDA) y otros **NVIDIA**. Si el
compose exigiera GPU NVIDIA de forma fija, no arrancaría en las máquinas AMD ni en
las que no tienen `nvidia-container-toolkit`. La solución es separar
responsabilidades:

> Las máquinas AMD ya no están condenadas a CPU: desde que Ollama trae el backend
> **Vulkan**, tienen su propio override (§8.2). Pero la separación base/override sigue
> siendo necesaria por la misma razón de siempre.

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

Al pasar los dos `-f`, Compose **fusiona** ambos archivos: toma todo el base y le
superpone lo del override. Si se omite el segundo `-f`, se corre en CPU. Requisito
para la variante NVIDIA: tener instalado el **nvidia-container-toolkit** en el host;
si falla con `could not select device driver "nvidia"`, es que falta ese paquete.

### 8.2 El override `docker-compose.amd.yml`

**Para qué sirve:** acelerar **Ollama** (generación y embeddings) en placas AMD, incluidas
las que ROCm ya no soporta. Se probó en una **Radeon RX 570** (Polaris10 / gfx803, 4 GB).

**Por qué Vulkan y no ROCm:** ROCm cubre oficialmente RDNA1 (RX 5000) en adelante, así que
Polaris queda afuera; los workarounds tipo `HSA_OVERRIDE_GFX_VERSION` no sirven porque el
silicio no tiene los bloques que esos binarios esperan. **Vulkan** sí: el driver **RADV**
de Mesa soporta GCN/Polaris desde hace años, y desde **Ollama 0.12.11** el backend Vulkan
viene **incluido en la imagen oficial `ollama/ollama`** y se activa solo cuando el
container ve los devices de GPU. No hace falta una imagen alternativa ni un fork.

**Qué agrega el override** (sólo al servicio `ollama`):

1. `devices: /dev/dri:/dev/dri` → el container ve el nodo de render. **No** se pasa
   `/dev/kfd`: ese device es para ROCm/HSA, no para Vulkan.
2. `group_add` con los GIDs de `render` y `video` → permisos sobre esos nodos en hosts
   donde no son 666. Van como **números**, no como nombres: con nombres Docker los busca
   en el `/etc/group` del *container*, y la imagen de Ollama no tiene grupo `render`
   (falla con `unable to find group render`). Los defaults son los de Fedora
   (render=105, video=39); si tu distro usa otros (`getent group render video`),
   overrideálos con `RENDER_GID` / `VIDEO_GID` en el `.env`.
3. `OLLAMA_VULKAN: "1"` → explícito (redundante en ≥ 0.12.11, pero documenta la intención).
4. `OLLAMA_KEEP_ALIVE: "5m"` y `OLLAMA_NUM_PARALLEL: "1"` → ver "presupuesto de VRAM".

**Qué NO toca (y por qué):** a diferencia del override NVIDIA, **no toca `pipelines`**, no
usa build-args y **no requiere rebuildear la imagen**. El reranker cross-encoder corre con
torch, y torch no tiene backend Vulkan; ROCm tampoco es opción en gfx803. Así que el
reranker y la conversión de PDFs con marker siguen en **CPU**. Es una limitación real, no
un olvido.

**Presupuesto de VRAM (4 GB):** `qwen3-embedding:4b` pesa ~2.5 GB y el LLM de generación
por defecto (`qwen2.5:3b-instruct`) ~1.9 GB: **no entran juntos**. Se decidió no cambiar el
modelo de embeddings —`EMBEDDING_DIMENSION = 2560` está fijo en el pipeline y en la
indexación, y cambiarlo obligaría a recrear la tabla de pgvector y **reindexar todo el
corpus**, rompiendo compatibilidad con la base del resto del grupo. La alternativa elegida
es dejar que Ollama **descargue un modelo para cargar el otro**, a costa de unos segundos
de recarga por consulta. Para que eso sea posible, el override pisa dos valores de
`env/ollama.env` que están calculados para 12 GB de VRAM: `OLLAMA_KEEP_ALIVE=-1` (que
significa "no descargar nunca") y `OLLAMA_NUM_PARALLEL=2`. Como en Compose `environment`
le gana a `env_file`, no hace falta tocar ese archivo compartido.

**Rendimiento medido** (RX 570 4 GB + i3-10100F, Ollama 0.34, Vulkan/RADV). Ollama reporta
`total=4.0 GiB available=3.2 GiB`: el resto se lo lleva el escritorio, porque el i3-10100**F**
no tiene iGPU y la misma placa maneja el display.

| | VRAM en runtime | Reparto | Medición |
|---|---|---|---|
| `qwen2.5:3b-instruct` (generación) | 2.2 GB | **100% GPU** | 54-63 tok/s |
| `qwen3-embedding:4b` (embeddings) | 6.7 GB | 48%/52% CPU/GPU | 0,33 s por chunk en lote |

Dos cosas que sorprenden y conviene tener presentes:

- **El embedder no entra en 4 GB.** Pesa 2.5 GB en disco pero pide **6.7 GB** en runtime:
  la diferencia son los buffers de cómputo, que escalan con el contexto. Bajándole
  `num_ctx` a 512 baja a 2.9 GB y 80% GPU, pero sigue sin entrar del todo. **No** se
  arregla con `OLLAMA_CONTEXT_LENGTH`, que es global y le recortaría el contexto al LLM
  de generación (que necesita ~1600 tokens: 4 chunks + pregunta + 512 de respuesta).
  Se convive con el split; en la práctica no duele, por lo que sigue.
- **Embeber de a uno vs en lote cambia todo**: 1,95 s por chunk suelto contra 0,33 s en
  lotes de 32. La indexación ya usa lotes (`EMBED_BATCH_SIZE=64` en `run_indexing.py`),
  así que el número que importa es el segundo. En consulta se embebe un solo texto corto:
  **0,13 s**, irrelevante.

**Costo del swap:** ~4,7 s cada vez que Ollama tiene que descargar un modelo para cargar el
otro. Se paga una vez por consulta si venís de indexar, y no se paga en consultas seguidas.

**Requisito del host:** Mesa con RADV. Verificar con
`vulkaninfo --summary | grep -E 'deviceName|driverName'` (ver §7.3 para los falsos
positivos). Si el modelo elegido no entra en VRAM, `ollama ps` lo muestra como split
`x%/y%` CPU/GPU: la palanca sin tocar código es `LLM_MODEL` en el `.env` con uno más chico.

## 9. Resumen de decisiones de arquitectura

- **Todo local sobre Ollama:** sin dependencia de APIs externas, sin claves, sin
  cuotas ni rate limits. El costo se paga en latencia y hardware.
- **Recuperación híbrida (vector + keyword + IDs) con RRF y reranking:** mejor
  cobertura que sólo búsqueda semántica.
- **CPU por defecto:** portabilidad sobre todo el grupo; GPU como optimización, con un
  override por familia de placa (NVIDIA vía CUDA, AMD vía Vulkan).
- **Caché de conversión marker-pdf:** evita el costo alto de OCR/layout en CPU.
