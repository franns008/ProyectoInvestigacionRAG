# Data Splitting (Chunking) — Análisis, estrategias candidatas y cómo evaluarlas

Este documento describe cómo se parte (chunkea) el corpus antes de indexarlo en el
RAG de ciberseguridad, analiza críticamente la configuración actual, cataloga las
estrategias candidatas (con **mediciones reales sobre el corpus**) y define el
**harness para compararlas objetivamente** en performance de recuperación y calidad de
respuesta.

> **Estado (2026-07-17):** análisis + andamiaje de experimentación **implementados** y el
> experimento **corre end-to-end** (modo retrieval puro). El barrido completo de las 4
> estrategias quedó **a medio correr por RAM** (el proceso se mató embebiendo la 2ª
> estrategia con el stack + otros procesos activos, ver §7.3); se retoma **una estrategia
> a la vez**. El runtime de producción **no cambió**: el pipeline sigue usando
> `DocumentSplitter(word/200/20)`. Todo lo nuevo vive aparte y en tablas pgvector separadas.

> **Recordatorio de arquitectura:** los **embeddings** se calculan **local** en Ollama
> (`bge-m3`, 1024 dims) y sólo la **generación** va a **Groq**. El chunking **no consume
> tokens de Groq durante la indexación**; sólo influye en Groq por la cantidad de
> contexto que termina en el prompt de cada consulta. Ver
> [arquitectura_groq.md](arquitectura_groq.md).

---

## 1. Dónde vive el chunking

Todo el splitting del corpus **splittable** (PDF/DOCX/MD/TXT) está en un único punto:
[`src/pipeline/pipeline_ciberseguridad.py`](../src/pipeline/pipeline_ciberseguridad.py),
método `_index_new_documents()`:

```python
splitter = DocumentSplitter(
    split_by="word",
    split_length=self.valves.split_length,   # 200
    split_overlap=self.valves.split_overlap,  # 20
)
docs = splitter.run(documents=splittable_docs)["documents"]
```

Los documentos **atómicos** (CWE desde XML, CVE desde JSON) **no** se chunkean: un
CWE/CVE = un `Document` con id determinístico. El chunking sólo afecta a las guías
INCIBE.

| Parámetro       | Valor actual | Significado                         |
|-----------------|--------------|-------------------------------------|
| `split_by`      | `"word"`     | Unidad de corte: palabras           |
| `split_length`  | `200`        | Palabras por chunk (~280 tokens)    |
| `split_overlap` | `20`         | Solapamiento (10%)                  |

---

## 2. El corpus real (medido)

3 estudios técnicos de INCIBE (PDF → Markdown vía marker, cacheados en
`data/raw/_converted_md/`) + el catálogo CWE (XML atómico). Palabras y estructura
**medidas** sobre el markdown convertido:

| Documento (PDF)        | Palabras | Headers `#` (h1/h2/h3/h4) |
|------------------------|----------|---------------------------|
| Cloud SCI              | ~3.900   | 1 / 3 / 6 / 9             |
| XSS persistente        | ~7.850   | 7 / 10 / 1 / 9            |
| Redes privadas 5G      | ~22.380  | 11 / 13 / 1 / 117         |

La estructura (`3.1 Definición de XSS`, `7.2 Programación segura`, `8.1 Aplicaciones
SCADA`, tablas de configuración) es **señal semántica** que la configuración actual
ignora.

---

## 3. Análisis crítico de la configuración actual

1. **El corte por palabras destruye la estructura.** Un límite fijo de 200 palabras
   parte secciones a la mitad, corta tablas dejando filas sin encabezado y trocea
   ejemplos de código. La guía 5G tiene 117 headings de nivel 4 que hoy se fragmentan.

2. **200 palabras es chico para texto expositivo.** `bge-m3` admite 8192 tokens y se le
   dan ~280. Medido sobre el corpus, la config actual produce **255 chunks de ~148
   palabras de media** (bastante por debajo del target de 200 por los cortes en
   secciones cortas). Para docs técnicos lo habitual es 400-700 palabras.

3. **El heading queda huérfano de su contenido.** El título es contexto útil para
   retrieval, pero como es "palabras más" en el stream, cae en un chunk distinto al de
   su contenido.

4. **Corta a mitad de oración.** `split_by="word"` no respeta límites de frase.

5. **Idioma no configurado.** El corpus es español y el tokenizador de oraciones de
   Haystack usa inglés por defecto.

6. **Cambiar `split_length` en runtime no re-chunkea lo indexado.** `on_valves_updated`
   sólo reconstruye el pipeline de query, no re-indexa.

7. **Inconsistencias código ↔ runtime.** El `valves.json` de runtime puede diferir del
   código en `retriever_top_k` y `max_tokens`.

8. **⚠ Bug latente descubierto: `MarkdownToDocument` borra los headers.** El pipeline
   convierte cada PDF→markdown con marker (que **sí** deja `#`, `##`…) pero luego lo
   pasa por `MarkdownToDocument`, que **renderiza el markdown a texto plano y elimina la
   sintaxis de headers**. Verificado: sobre el corpus, `MarkdownToDocument` deja **0
   matches** del patrón de header ATX. Consecuencia: cualquier splitter *markdown-aware*
   (como `MarkdownHeaderSplitter`) alimentado con esa salida **no encuentra ningún
   header y no parte por sección**. Para aprovechar la estructura hay que darle el
   **markdown crudo** (los `.md` de `_converted_md/`), no la salida de
   `MarkdownToDocument`. Este es el hallazgo clave que habilita la Estrategia D.

---

## 4. Impacto en consumo de tokens (Groq)

El chunking **no** toca la cuota de embeddings (local). Impacta en Groq **sólo** por el
contexto del prompt: `nº_docs_al_prompt × tamaño_chunk` + plantilla + pregunta + salida.
Hoy el **techo duro** de docs al prompt lo pone el **reranker** (`ranker_top_k=4`), no el
joiner (ver [reranker_cross_encoder.md](reranker_cross_encoder.md)). Por eso subir el
tamaño de chunk sí sube tokens/query, pero acotado a 4 piezas.

Estimación (~1,4 tokens/palabra en el tokenizer de Llama), con `ranker_top_k=4`:

| Palabras/chunk | Tokens/chunk | 4 docs al prompt | + plantilla/pregunta | + salida (512) | Total/query |
|----------------|--------------|------------------|----------------------|----------------|-------------|
| 148 (actual)   | ~207         | ~830             | ~150                 | 512            | **~1.500**  |
| 250 (sentence) | ~350         | ~1.400           | ~150                 | 512            | **~2.060**  |
| 490 (recursive)| ~686         | ~2.744           | ~150                 | 512            | **~3.400**  |

Trade-off: chunks más grandes dan más contexto por pieza (mejor para respuestas
"explicá y prevení") a cambio de más tokens/query, lo que baja el techo de queries/minuto
antes del 429 de Groq. El **overlap se paga doble** si dos chunks solapados entran ambos
al prompt.

---

## 5. Estrategias candidatas (con medición real)

Registradas en
[`src/pipeline/chunking_strategies.py`](../src/pipeline/chunking_strategies.py). Números
**medidos** sobre los 3 PDFs INCIBE (34.130 palabras en total):

Números **medidos en la corrida real** (A, B) y en validación local sobre el mismo corpus
(C, D):

| Estrategia | Componente Haystack | n_chunks | avg_words | Notas |
|---|---|---|---|---|
| **A. `current_word200`** (baseline) | `DocumentSplitter(word/200/20)` | 255 | 148.1 | La de producción. Sin `language`, corta a mitad de oración. |
| **B. `sentence_es`** | `DocumentSplitter(word/220/30, respect_sentence_boundary, language=es)` | 148 | 278.1 | No corta oraciones; tokenizador ES. Cambio mínimo, mismo componente. |
| **C. `recursive_500`** | `RecursiveDocumentSplitter(500/75, sep=[¶,\n,sentence, ])` | 82 | 489.4 | Respeta jerarquía párrafo→línea→oración. Chunks grandes, autocontenidos. |
| **D. `markdown_header`** | `MarkdownHeaderSplitter(h1-3) + secondary word/400/40` + breadcrumb | 151 | 264.9 | **Requiere markdown crudo** (§3.8). Preserva `header`/`parent_headers` en meta y los antepone al contenido. |

Notas de implementación:

- **B** usa `split_by="word"` + `respect_sentence_boundary` (no `split_by="period"`:
  Haystack ignora `respect_sentence_boundary` con `period` y emite un warning). Requiere
  `nltk` (tokenizador de oraciones).
- **C** (`RecursiveDocumentSplitter`) también depende de `nltk` para el separador
  `"sentence"`.
- **D** (`MarkdownHeaderSplitter`) **sólo funciona con el markdown crudo**. El
  experimento le pasa los `.md` de `_converted_md/` sin re-renderizar. `secondary_split`
  usa `"word"` (no `"period"`) porque el splitter interno no expone `language` y
  tokenizaría oraciones en inglés sobre texto español. `prepend_header_context()`
  antepone el breadcrumb `[Sección padre > Sección]` al `content` (Contextual chunking,
  Mejora 4 del análisis previo) usando `dataclasses.replace` (Document es inmutable).

> **Dependencias:** `nltk` (para B y C) y `markdown-it-py`/`mdit-plain` (para el
> converter) están **horneadas en la imagen** del container `pipelines`
> ([Dockerfile.pipelines](../infrastructure/Dockerfile.pipelines)): `nltk` + los datos
> `punkt_tab` se instalan en el build, y el stack Haystack incluye el converter de
> markdown. No hay que instalar nada aparte para correr el experimento.

---

## 6. Cómo se evalúan — el problema de medición y su solución

### 6.1 Por qué el ground truth por doc-id NO sirve para comparar chunking

El harness Tier 1 ([eval_harness.md](eval/eval_harness.md)) mide `recall@k` comparando
los **ids** de los docs recuperados contra `expected_doc_ids`. Eso funciona para CWE/CVE
(id determinístico: `cwe-89`, `sha256(cve_id)`) pero **rompe para el corpus splittable**:
el id de un chunk es un **hash de su contenido**, así que **cambia con cada estrategia de
chunking**. No hay un id estable que anclar.

### 6.2 La solución: métrica a nivel de FUENTE

Se agrega **Tier 1b — `source_recall@k`** ([metrics.py](../src/pipeline/eval/metrics.py)):
en vez de comparar ids de chunk, compara `meta.source` (el **nombre del archivo
original**, que se preserva sea cual sea el splitter) contra un nuevo campo del dataset
`expected_sources`. Así "¿recuperó el chunk correcto de la guía XSS?" se vuelve medible y
**estable entre estrategias**.

Se agregaron 3 preguntas de categoría **`guia_incibe`** a
[`dataset.yaml`](../src/pipeline/eval/dataset.yaml), ancladas a contenido que **sólo vive
en las guías** (no en el catálogo CWE), una por documento:

| id | fuente esperada | qué estresa |
|---|---|---|
| `xss-programacion-segura`      | XSS.pdf   | recuperar la sección "7.2 Programación segura" |
| `cloud-honeypots-industriales` | Cloud.pdf | recuperar "8.2 Honeypots Industriales" |
| `5g-riesgo-mitm`               | 5G.pdf    | recuperar la sección de riesgos MITM en 5G |

Estas son las preguntas donde el chunking **realmente** mueve la aguja; las de CWE sirven
de control de "no rompí el retrieval de lo atómico".

### 6.3 Métricas que reporta la comparación

- **Tier 1 `recall@k`** (doc-id): control sobre las preguntas CWE.
- **Tier 1b `source_recall@k`** (fuente): **el eje** que mide el chunking del splittable.

> **Retrieval puro (sin LLM).** El experimento arma el pipeline con
> `build_rag_pipeline(include_llm=False)` — retrievers + joiner + reranker, sin
> `prompt_builder` ni generador. Así **no exige `GROQ_API_KEY` ni paga la latencia de
> generación**, y el chunking se juzga por lo que realmente mueve: el retrieval. La
> **Tier 2 `SAS`** (calidad de respuesta generada) se mide aparte con
> [run_eval.py](../src/pipeline/eval/run_eval.py) una vez elegida la estrategia, no en
> este barrido.

---

## 7. El harness de experimentación

[`src/pipeline/eval/run_chunking_experiment.py`](../src/pipeline/eval/run_chunking_experiment.py)
compara todas las estrategias en una sola corrida, **sin tocar la tabla de producción**:

1. Carga el corpus splittable en dos variantes: *rendered* (vía `MarkdownToDocument`,
   como producción) y *raw markdown* (crudo, para la Estrategia D — ver §3.8).
2. Embeba los **documentos atómicos una sola vez** (idénticos entre estrategias) y los
   reutiliza en cada tabla → ahorra re-embeber el catálogo CWE por estrategia. Por
   defecto indexa **sólo CWE** (control de recall por doc-id); los ~76k CVE de NVD están
   **OFF** (`--include-cve` para activarlos) porque son atómicos, no aportan a la
   comparación de chunking y su embedding local en CPU tarda horas.
3. Por cada estrategia: chunkea el splittable, embeba sólo esos chunks, y escribe
   `chunks + atómicos` en una tabla dedicada `ciberseguridad_docs_chunk_<estrategia>`.
4. Corre el dataset completo contra cada tabla y reporta una **tabla comparativa**.

Habilitado por `get_document_store(table_name=..., keyword_index_name=...)`
([pipeline_ciberseguridad.py](../src/pipeline/pipeline_ciberseguridad.py)), ahora
parametrizable (defaults = store de producción, runtime sin cambios).

### 7.1 El flag de estrategia

El tipo de chunking se elige **al lanzar**, con `--strategies` (uno o varios nombres, que
son las claves de `STRATEGIES` en `chunking_strategies.py`). No es dinámico en caliente:
cada corrida indexa y evalúa las estrategias pedidas. Para **sumar** un tipo nuevo, se
agrega una entrada al diccionario `STRATEGIES` y queda disponible en el flag sin tocar el
orquestador.

### 7.2 Cómo correrlo (container efímero, sin auto-indexar producción)

No se usa `docker compose exec pipelines`, porque eso levanta el **server de Open WebUI**,
que en su arranque auto-indexa el corpus de producción — incluidos los **~76k CVE de NVD**
(horas de embeddings). En su lugar, un container **efímero** que saltea el server y corre
Python directo. En **Fedora/SELinux** los bind-mounts requieren `:z`; **no** se monta el
volumen `marker_cache` (el reranker ya vive en la imagen). Desde `infrastructure/`:

```bash
docker run --rm --network infrastructure_default -e TORCH_DEVICE=cpu \
  -v "$(cd .. && pwd)/src/pipeline":/app/pipelines:z \
  -v "$(cd .. && pwd)/data/raw":/app/pipelines/rawdata:z \
  --entrypoint python infrastructure-pipelines:latest \
  /app/pipelines/eval/run_chunking_experiment.py
```

- `--strategies A B …` — subconjunto de estrategias (default: todas).
- `--include-cve` — indexar también los ~76k CVE (lento; **off** por defecto, ver paso 2
  del flujo: son atómicos y no aportan a comparar chunking).

Salida (formato real, retrieval puro):

```
estrategia          chunks   avg_w  recall@k  src_recall
current_word200        255   148.1       ...         ...
sentence_es            148   278.1       ...         ...
recursive_500           82   489.4       ...         ...
markdown_header        151   264.9       ...         ...
```

Los `n_chunks`/`avg_w` son reales; las columnas `recall@k`/`src_recall` quedan pendientes
del barrido completo (§7.3). El JSON por corrida se persiste en
`src/pipeline/eval/results/chunking_experiment_<timestamp>.json` (gitignored).

### 7.3 Nota operativa: RAM

El experimento carga el reranker (~2 GB residentes) + embeddings + pgvector. En una máquina
con ~15 GB de RAM y otros procesos activos (navegador, otro stack Docker), la primera
corrida completa se **quedó sin memoria** (OOM, exit 137) embebiendo la 2ª estrategia. Para
que entre, correr **una estrategia por vez** (`--strategies recursive_500`, luego otra) y/o
liberar RAM (cerrar apps, `docker stop` de otros proyectos). No es un problema de disco.

---

## 8. Recomendación priorizada

| Prioridad | Acción | Beneficio | Riesgo/costo |
|-----------|--------|-----------|--------------|
| 1 | Correr `run_chunking_experiment.py` (una estrategia por vez) y leer `source_recall` | Decisión basada en datos, no en teoría | Requiere la imagen buildeada + vdb/ollama arriba; RAM (§7.3) |
| 2 | Adoptar la ganadora (probable **C** o **D**) como splitter de producción | Mejor contexto por chunk | Reindexar la tabla de producción |
| 3 | Si gana **D**: cambiar el pipeline para pasar **markdown crudo** al splitter markdown-aware (evitar `MarkdownToDocument` para PDFs) | Aprovecha la estructura de sección | Toca `_load_local_documents` |
| 4 | Re-indexar al cambiar los valves de chunking | Config utilizable de verdad | Cambio en `on_valves_updated` |
| 5 | Alinear parámetros código ↔ runtime | Consistencia | — |

**Hipótesis a validar con el experimento:** las estrategias que respetan estructura (C
recursivo y D markdown-aware) deberían **subir `source_recall`** en las preguntas
`guia_incibe` sin degradar el `recall@k` de CWE (que no se chunkean). El breadcrumb de la
D es el candidato de mayor impacto para preguntas conceptuales sobre las guías. La calidad
de respuesta (SAS) de la ganadora se confirma después con `run_eval.py` (§6.3).

---

## 9. Relación con otras mejoras

- [enriquecimiento_de_chunks.md](enriquecimiento_de_chunks.md) — Contextual Retrieval vía
  `meta_fields_to_embed`. **Ortogonal** al chunking: se aplica *después* de partir, sobre
  cualquiera de estas estrategias. El breadcrumb de la Estrategia D es una versión
  barata y determinística de la misma idea (contexto de sección sin llamar al LLM).
- [reranker_cross_encoder.md](reranker_cross_encoder.md) — el `ranker_top_k=4` es hoy el
  techo real de tokens al prompt; acota el costo de subir el tamaño de chunk.
- [eval/eval_harness.md](eval/eval_harness.md) — harness base (Tiers 1-3) que este
  experimento extiende con la métrica de fuente.
- [eval/mejoras_harness.md](eval/mejoras_harness.md) — H10 (nDCG) y el graded de H3 harían
  la comparación de chunking aún más fina en preguntas multi-doc.
