# Modos de LLM: API Key (Groq) ↔ Ollama local (Nvidia)

> **Doc crucial.** Explica cómo el mismo repo levanta la generación del RAG de dos
> maneras según el hardware de cada integrante. Leer antes de tocar la generación,
> el arranque de la infra o el harness de evaluación.

## Por qué existe

El equipo tiene hardware mixto: algunos integrantes corren modelos locales con GPU
sin problema y otros dependen de una API Key. Para que **el mismo repositorio** sirva
a ambos, la **generación** del LLM es intercambiable entre dos proveedores. Los
**embeddings** (`bge-m3` en Ollama) son siempre locales y **no cambian** entre modos.

| Modo | `LLM_PROVIDER` | Generación | Hardware | Necesita |
|------|----------------|------------|----------|----------|
| **API Key** (default) | `groq` | Groq (nube, endpoint OpenAI-compatible) | cualquiera (CPU) | `GROQ_API_KEY` |
| **Local** | `ollama` | Ollama (local) | GPU Nvidia recomendada | modelo `ollama pull`-eado |

Dos ejes **ortogonales** e independientes:
- **Proveedor del LLM** → variable de entorno `LLM_PROVIDER` (+ `LLM_MODEL`).
- **Hardware CPU/GPU** → overlay `docker-compose.nvidia.yml` (ya existía).

No hay Makefile ni scripts: el modo se elige con una variable en `.env`, así funciona
igual en **Windows y Linux** (`docker compose` es idéntico en ambos).

## Configuración del modo (variables)

El modo se controla con dos variables en `infrastructure/.env` (plantilla en
`infrastructure/.env.example`):

```dotenv
LLM_PROVIDER=groq        # o "ollama"
LLM_MODEL=               # vacío = default por proveedor
```

- `LLM_PROVIDER=groq` (default): generación por API. Requiere `GROQ_API_KEY`.
- `LLM_PROVIDER=ollama`: generación local. `GROQ_API_KEY` puede quedar vacío.
- `LLM_MODEL` overridea el modelo. Vacío usa el default:
  - groq → el del pipeline (`meta-llama/llama-4-scout-17b-16e-instruct`).
  - ollama → `qwen2.5:3b-instruct` (~2 GB VRAM, bilingüe ES/EN).

## Levantar la infra — paso a paso

Todos los comandos se corren **parados en `infrastructure/`**:

```bash
cd infrastructure
```

Elegí UNO de los dos modos.

---

### 🅰️ Modo API Key (Groq) — cualquier máquina, CPU

Para quienes no corren modelos locales. La generación va a Groq; sólo los embeddings
corren local (CPU alcanza).

1. **Crear el `.env`** (si no existe):
   ```bash
   cp .env.example .env
   ```
2. **Editar `.env`** y dejar:
   ```dotenv
   LLM_PROVIDER=groq
   GROQ_API_KEY=<tu key de https://console.groq.com/keys>
   ```
   (El resto de credenciales usan defaults del compose — no hace falta tocarlas.)
3. **Levantar el stack** (la primera vez buildea la imagen, tarda):
   ```bash
   docker compose up -d --build
   ```
4. **Descargar el modelo de embeddings** en Ollama (una sola vez):
   ```bash
   docker compose exec ollama ollama pull bge-m3
   ```
5. **Usar el RAG**: abrí OpenWebUI en `http://localhost:8180` y elegí el modelo
   *"RAG ciberseguridad"* en el selector.

---

### 🅱️ Modo Ollama local (Nvidia) — GPU

Para quienes tienen GPU Nvidia. La generación **y** los embeddings corren locales;
no se usa Groq. Requiere `nvidia-container-toolkit` instalado en el host.

1. **Crear el `.env`** (si no existe):
   ```bash
   cp .env.example .env
   ```
2. **Editar `.env`** y dejar:
   ```dotenv
   LLM_PROVIDER=ollama
   LLM_MODEL=qwen2.5:3b-instruct
   # GROQ_API_KEY podés dejarla vacía o sin poner
   ```
3. **Levantar el stack con el overlay de GPU** (primera vez buildea con CUDA, tarda):
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up -d --build
   ```
4. **Descargar los modelos** en Ollama (una sola vez): embeddings **y** el LLM:
   ```bash
   docker compose exec ollama ollama pull bge-m3
   docker compose exec ollama ollama pull qwen2.5:3b-instruct
   ```
   > Si en el paso 2 pusiste otro `LLM_MODEL`, descargá **ese** en vez de
   > `qwen2.5:3b-instruct`.
5. **Usar el RAG**: abrí OpenWebUI en `http://localhost:8180` y elegí el modelo
   *"RAG ciberseguridad"* en el selector.

> El overlay `docker-compose.nvidia.yml` reserva la GPU para `ollama` y `pipelines`
> y buildea torch con CUDA. Detalles en [`docs/arquitectura_groq.md`](arquitectura_groq.md) §7.1.

---

### Cambiar de modo o parar

- **Cambiar de modo** (ej. de groq a ollama): editá `LLM_PROVIDER` en `.env` y
  **recreá** el stack con el `up` del modo destino (las env vars se leen al arrancar):
  ```bash
  docker compose up -d           # (o con el overlay -f nvidia según el modo)
  ```
- **Ver logs de la generación**: `docker compose logs -f pipelines` o el archivo
  `src/pipeline/logCiberseguridad.txt`.
- **Parar** todo sin borrar datos: `docker compose down`.
- **Reset total** (borra la base y hay que reindexar): `docker compose down -v`.

> **Notas**
> - "Una sola vez" = el `ollama pull` persiste en el volumen `ollama_data`; no hay
>   que repetirlo salvo que borres el volumen.
> - `ollama` sin GPU funciona pero es lento; el modo local está pensado para Nvidia.
>   Si no tenés GPU, usá el modo API Key.
> - Si `8180` está ocupado, cambiá `OWEBUI_PORT` en `.env` (ver puertos opcionales).

## Cómo funciona por dentro

- `src/pipeline/pipeline_ciberseguridad.py`
  - `build_generator(valves)` lee `LLM_PROVIDER` en tiempo de build y devuelve un
    `OpenAIGenerator` (Groq) o un `OllamaGenerator` (local). `GROQ_API_KEY` **sólo** se
    lee en la rama Groq → el modo ollama no la exige.
  - `build_rag_pipeline(...)` usa `build_generator(v)` como componente `llm`; el resto
    del grafo (retrievers híbridos + RRF + prompt) no cambia.
  - `DEFAULT_OLLAMA_LLM` fija el modelo local por defecto.
- `infrastructure/docker-compose.yml` pasa `LLM_PROVIDER` y `LLM_MODEL` al servicio
  `pipelines` (con defaults `groq` / vacío).

## Evaluación (harness)

- **Tier 1+2** (`eval/run_eval.py`): retrieval + SAS. No usa LLM de generación → corre
  igual en ambos modos, sin cambios.
- **Tier 3** (`eval/run_eval_llm.py`): el **juez** LLM también sigue `LLM_PROVIDER`.
  En modo ollama usa `OllamaChatGenerator` local → los integrantes con GPU pueden
  correr el Tier 3 **sin key de Groq**:

  ```bash
  docker compose exec pipelines python /app/pipelines/eval/run_eval_llm.py --limit 8
  ```

Ver [`docs/eval_harness.md`](eval_harness.md) para el detalle del harness.

## Checklist de troubleshooting

- **`ollama` mode, respuesta vacía / error de modelo** → falta `ollama pull <LLM_MODEL>`.
- **`groq` mode, error de auth** → `GROQ_API_KEY` no seteada en `.env`.
- **Cambiaste `.env` y no toma efecto** → recreá el stack (`docker compose up -d`); las
  env vars se leen al construir el pipeline.
- **Modelo local muy lento** → estás en CPU (sin el overlay `docker-compose.nvidia.yml`)
  o el modelo no entra en VRAM; probá uno más chico vía `LLM_MODEL`.
