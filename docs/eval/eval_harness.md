# Plan de diseño: harness de evaluación del RAG

> **Estado (2026-07-03): IMPLEMENTADO y verificado.** El harness vive en
> [../src/pipeline/eval/](../../src/pipeline/eval/) y corre dentro del container `pipelines`.
> Las tres fases (0–3) están operativas. Baseline inicial (27 preguntas, `top_k=3`, `temp=0`):
> **recall@k=0.841, hit_rate=0.864, MRR=0.693** (Tier 1, n=22 con ground truth) y
> **SAS=0.755** (Tier 2, n=14). Tier 3 (juez Groq) probado sobre subsets.
>
> **Hallazgos del baseline:** las categorías débiles son `concepto_es` (recall 0.333) y
> `multi_doc` (0.250) — la brecha es→en descrita en
> [optimizacion_keyword_retrieval.md](../optimizacion_keyword_retrieval.md). Subir `top_k` a 8 **no**
> las mejora → el cuello de botella es el preprocesado de la query, no `k`.
>
> **Corpus actual:** 710 CWE, **0 CVE** (la ingesta NVD aún no se corrió). Por eso las preguntas
> de CVE del dataset son hoy tests de robustez (el RAG debe decir "no sé"); cuando se indexen
> CVEs, cambiarles `expected_doc_ids` al `sha256(cve_id)` real.

## Contexto y objetivo

Hoy, cuando se cambia una pieza del pipeline (retriever, chunking, modelo de embeddings,
prompt, LLM, valves), la única forma de saber si mejoró o empeoró es leer a ojo los logs de
[pipeline_ciberseguridad.py](../../src/pipeline/pipeline_ciberseguridad.py) y "sentir" las
respuestas. Eso no escala ni es reproducible.

El objetivo de este plan es definir un **eval harness**: un dataset de preguntas con ground
truth y un flujo automático que, ante cualquier cambio, corra todas las preguntas contra el
RAG, calcule un conjunto de métricas y muestre el **delta contra un baseline** — es decir,
convertir "hice un cambio" en un número: *cuánto* mejoró/empeoró y *en qué* preguntas.

**Principio rector:** el harness debe cubrir **todo el stack**, no solo el retriever. Una sola
corrida por pregunta produce todo (documentos recuperados + respuesta generada) y de ahí se
derivan todas las métricas.

## Qué cambio mueve qué métrica

Cualquier cambio cae en esta grilla. El harness mide todas las filas siempre; según lo que
tocaste, mirás la fila relevante y las demás como control de "no rompí nada".

| Cambiás… | Métrica que lo captura | Tier | Costo |
|---|---|---|---|
| Retriever (embedding/keyword, `top_k`, RRF) | Recall@k, Hit@k, MRR, contribución por retriever | 1 | gratis, determinístico |
| Chunking (`split_length`, `split_overlap`) | Recall@k + context precision | 1 / 3 | gratis / juez |
| Modelo de embeddings (bge-m3 → otro) | Recall@k (requiere reindexar) | 1 | gratis |
| Converters (CWE/CVE → Document) | Recall@k (¿el doc esperado existe y matchea?) | 1 | gratis |
| `build_keyword_query` (preprocesado) | Hit del keyword retriever, Recall@k | 1 | gratis |
| Prompt template | Faithfulness, answer relevancy, SAS | 2 / 3 | local / juez |
| LLM (modelo, temperatura, `max_tokens`) | SAS, faithfulness, correctness | 2 / 3 | local / juez |

## Arquitectura de ejecución: dentro del container `pipelines`

El pipeline hardcodea hostnames de la red interna de Docker —
`vdb:5432` y `ollama:11434` ([pipeline_ciberseguridad.py](../../src/pipeline/pipeline_ciberseguridad.py#L65-L66))—
y `GROQ_API_KEY` se inyecta desde `env/pipelines.env`. Desde el host esos nombres no resuelven
y el secreto no está a mano.

Decisión: **el harness se ejecuta con `docker compose exec pipelines …`**, donde `vdb`,
`ollama` y `GROQ_API_KEY` ya resuelven. Como el compose monta `../src/pipeline/ →
/app/pipelines`
([docker-compose.yml](../../infrastructure/docker-compose.yml#L103-L108)), si el harness vive en
`src/pipeline/eval/` aparece en `/app/pipelines/eval/` **sin rebuild ni copy**.

> **A verificar (Fase 0):** que el server de Open WebUI Pipelines no intente auto-cargar los
> `.py` de subdirectorios como pipelines (busca una clase `Pipeline`). Los scripts de eval no
> la tienen. Si los escanea y falla, se resuelve poniendo el harness en un dir con guion bajo
> (`_eval/`) o apuntando `PIPELINES_DIR` fuera de él.

El harness usa el store **ya poblado** por la operación normal — **no re-indexa**. Solo lee de
`ciberseguridad_docs` y ejecuta el pipeline de query.

## Refactor mínimo para instanciar el pipeline headless

Hoy no se puede instanciar el pipeline sin arrastrar todo el runtime: `__init__` carga
marker-pdf e indexa documentos
([pipeline_ciberseguridad.py](../../src/pipeline/pipeline_ciberseguridad.py#L449-L468)). Para el
eval no queremos nada de eso.

Refactor propuesto (sin cambiar comportamiento de runtime): extraer a **funciones
módulo-level** lo que hoy son métodos de instancia:

- `get_document_store() -> PgvectorDocumentStore`  ← de `_get_document_store`
- `build_rag_pipeline(store, valves) -> HaystackPipeline`  ← de `_build_rag_pipeline`
- `build_keyword_query(...)` **ya es módulo-level** y se reutiliza tal cual.

Los métodos de la clase `Pipeline` pasan a delegar en estas funciones. Como `marker` se importa
*lazy* dentro de `__init__` (no en el top del módulo), `import pipeline_ciberseguridad` desde el
harness **no** dispara la carga pesada. El harness arma sus propias `Valves` (con
`temperature = 0`, ver abajo), construye el store y el pipeline, y corre pregunta por pregunta.

## Los tres tiers de métricas

La clave para poder correrlo seguido sin morir en el rate limit de Groq es escalonar por costo.

### Tier 1 — Retrieval (determinístico, gratis, segundos)

Aprovecha que los documentos atómicos tienen **id determinístico**: CWE →
`cwe-89`
([pipeline_ciberseguridad.py](../../src/pipeline/pipeline_ciberseguridad.py#L150)), CVE →
`sha256(cve_id)`
([pipeline_ciberseguridad.py](../../src/pipeline/pipeline_ciberseguridad.py#L281)). El ground
truth de una pregunta es, literalmente, una **lista de ids esperados**. No hace falta fuzzy
match ni juez.

Métricas sobre los documentos que el `DocumentJoiner` pasó al prompt:
- **Recall@k / Hit@k:** ¿aparecieron los `expected_doc_ids` en el top-k?
- **MRR:** ¿qué tan arriba aparecieron?
- **Contribución por retriever:** embedding vs keyword (ya se expone en el output del pipeline
  vía `include_outputs_from`, igual que en
  [`_log_retrieved_docs`](../../src/pipeline/pipeline_ciberseguridad.py#L512)).

### Tier 2 — Similitud de respuesta (SAS, local, gratis, rápido)

Aprovecha que **ya corre Ollama con bge-m3 local y gratis**: se mide **SAS (Semantic Answer
Similarity)** = coseno entre el embedding de la respuesta generada y el de la `reference_answer`
del dataset. Captura regresiones de prompt/LLM **sin gastar una sola llamada a Groq de juez**.

> Se calcula con `OllamaTextEmbedder` directo (reusa bge-m3) en vez del `SASEvaluator` de
> Haystack, para no descargar un modelo sentence-transformers nuevo.

Este es el **gate diario** para cambios de generación.

### Tier 3 — Juez LLM (caro, rate-limited, profundo)

Faithfulness (¿la respuesta está fundada en los chunks recuperados?), answer relevancy y
context relevance, usando **Groq como juez** reusando el `OpenAIGenerator` ya cableado
([pipeline_ciberseguridad.py](../../src/pipeline/pipeline_ciberseguridad.py#L747-L757)).

> **Recomendación:** usar los evaluators nativos de Haystack (`FaithfulnessEvaluator`,
> `ContextRelevanceEvaluator`) en vez de agregar Ragas como dependencia — reusan el wiring de
> Groq existente y evitan una capa extra. Ragas queda como alternativa si más adelante se
> quieren sus métricas específicas.

Multiplica llamadas (varias por pregunta y métrica), así que con ~10 preguntas ya se rozan los
429 de Groq. Se corre **a mano antes de un merge grande**, no en cada iteración.

**Regla operativa:** Tier 1 + 2 en cada cambio (gratis, ~segundos, 1 sola generación por
pregunta). Tier 3 puntual.

## Determinismo: forzar `temperature = 0` en eval

El pipeline usa `temperature = 0.5`
([pipeline_ciberseguridad.py](../../src/pipeline/pipeline_ciberseguridad.py#L446)). Con eso la
misma pregunta da respuestas distintas en cada corrida, y el delta de SAS/faithfulness sería
**ruido, no señal del cambio**. El harness debe construir las `Valves` con `temperature = 0`.
Si se quisiera medir con la temperatura real de producción, habría que correr N veces y
promediar — pero para *comparar cambios*, temp 0 es lo correcto.

(Nota: temp 0 reduce la varianza pero no la elimina del todo con la API; alcanza para el
propósito de comparar.)

## Diseño del dataset

### Esquema (`dataset.yaml`)

```yaml
- id: sqli-por-numero            # slug estable, se usa en el reporte por-pregunta
  category: id_cwe               # ver taxonomía
  question: "Explicame la vulnerabilidad CWE-89 y cómo prevenirla."
  expected_doc_ids: ["cwe-89"]   # Tier 1. Vacío = pregunta solo-generación.
  reference_answer: |            # Tier 2 (SAS) y Tier 3 (juez). Opcional.
    CWE-89 es SQL Injection: entrada no sanitizada que se interpola en una
    consulta SQL y altera su lógica. Se previene con consultas parametrizadas
    / prepared statements y validación de entrada.
  eval_focus: [retrieval, generation]   # qué tiers pesan para esta pregunta
  tags: [injection, es]
```

Campos:
- `expected_doc_ids` — ground truth de retrieval. Debe **curarse contra el corpus realmente
  indexado**: correr la pregunta una vez, inspeccionar el store / los logs y fijar los ids
  correctos a mano. Para preguntas puramente conceptuales puede quedar vacío.
- `expected_sources` — ground truth a nivel de **fuente** (nombre del archivo original,
  `meta.source`). Habilita **Tier 1b — `source_recall@k`**: a diferencia de un chunk-id (hash
  del contenido, cambia con cada estrategia de chunking), el nombre de archivo es estable, así
  que es el ground truth correcto para preguntas cuyo contenido vive en el corpus splittable
  (guías INCIBE) y para **comparar estrategias de chunking**. Ver
  [../data_splitting.md](../data_splitting.md). Vacío/ausente = no aplica.
- `reference_answer` — respuesta modelo escrita a mano. Alimenta SAS y el juez. Puede omitirse
  en preguntas solo-retrieval.
- `eval_focus` — permite reportar por foco: un cambio de retriever se juzga por las de
  `retrieval`; uno de prompt/LLM por las de `generation`.

### Taxonomía abarcativa de preguntas

El dataset debe cubrir estas categorías para que *cualquier* tipo de cambio tenga preguntas que
lo estresen. La columna "estresa" indica qué parte del sistema pone a prueba.

| Categoría (`category`) | Ejemplo | Ground truth | Estresa |
|---|---|---|---|
| **`id_cve`** — CVE por número | "¿Qué es CVE-2023-1234?" | `sha256(CVE-2023-1234)` | fast-path de IDs + keyword exacto |
| **`id_cwe`** — CWE por número | "Explicame CWE-89" | `cwe-89` | fast-path de IDs + keyword exacto |
| **`id_variantes`** — formato no canónico | "que es cwe 89", "cve 2023 1234" | mismo id | normalización de `build_keyword_query` |
| **`nombre_cve`** — CVE por producto/nombre | "vulnerabilidades de Log4Shell / Apache Struts" | CVEs cuyos `vendors`/`products` matchean | embeddings + metadata de CVE |
| **`nombre_cwe`** — CWE por nombre conceptual | "¿Qué es un buffer overflow?" | `cwe-120` / `cwe-787` | embeddings concepto→doc |
| **`concepto_es`** — concepto en español | "inyección SQL", "credenciales en el código" | `cwe-89` / `cwe-798` | brecha es→en (docs CWE en inglés) |
| **`desarrollar`** — respuesta larga/explicativa | "¿Qué es XSS y cómo se previene?" | CWE(s) de XSS + foco en generación | faithfulness, SAS, cobertura |
| **`comparativa`** — contraste multi-concepto | "¿Diferencia entre XSS reflejado y almacenado?" | `cwe-79`, `cwe-80`, … | recall multi-doc, `top_k`, joiner |
| **`multi_doc`** — requiere varios docs | "principales vulnerabilidades de inyección" | conjunto de CWE de inyección | Recall@k, RRF, `top_k` |
| **`procedimental`** — mitigación/pasos | "¿Cómo mitigar CSRF paso a paso?" | `cwe-352` | calidad y fidelidad de la respuesta |
| **`bilingue`** — misma pregunta ES/EN | "what is SQL injection" vs "qué es inyección SQL" | mismo id | consistencia idiomática (prompt exige responder en el idioma de la pregunta) |
| **`fuera_dominio`** — negativa esperada | "¿Cuál es la capital de Francia?" | `[]`, referencia = "no sé / fuera de alcance" | anti-alucinación (el prompt manda decir "no sé") |
| **`id_inexistente`** — ID que no está | "¿Qué es CVE-9999-99999?" | `[]`, referencia = "no encontrado" | robustez, no inventar |
| **`regresion`** — conceptuales puras estables | XSS / 5G / cloud sin ID | CWE/PDF esperados | que optimizaciones de keyword no degraden lo conceptual |

Las categorías `fuera_dominio` e `id_inexistente` son las que atrapan alucinaciones y se miden
sobre todo con Tier 3 (faithfulness) y con una `reference_answer` que expresa el "no sé".

### Tamaño y crecimiento

Arrancar con **~2-3 preguntas por categoría** (≈25-35 total) es suficiente para tener señal sin
que curar el ground truth sea una tarea enorme. El dataset crece agregando los casos reales que
vayan fallando en producción (cada bug encontrado → una pregunta nueva de regresión).

## Flujo de ejecución y evaluación

### `run_eval.py` — Tier 1 + 2 (gate diario)

Por cada pregunta, **una sola corrida** del pipeline con `temperature = 0`:

```python
result = pipeline.run(
    {
        "text_embedder":     {"text": q},
        "keyword_retriever": {"query": build_keyword_query(q)},
        "prompt_builder":    {"question": q},
    },
    include_outputs_from={"embedding_retriever", "keyword_retriever", "document_joiner"},
)
answer        = result["llm"]["replies"][0]
retrieved_ids = [d.id for d in result["document_joiner"]["documents"]]
```

De ese único resultado:
- **Tier 1:** `retrieved_ids` vs `expected_doc_ids` → recall@k, hit@k, MRR + contribución por
  retriever.
- **Tier 2:** SAS entre `answer` y `reference_answer` (embeddings Ollama).

Costo: **1 generación Groq por pregunta** (~30 calls) — trivial para el rate limit.

Salida: `results/<timestamp>.json` con métricas agregadas, **por categoría** y **por pregunta**,
más el **delta contra `baseline.json`**. En consola, una tabla resumen y la lista de preguntas
que regresaron (para ver *cuál*, no solo el promedio).

### `run_eval_llm.py` — Tier 3 (manual)

Recorre el dataset y aplica los evaluators de Haystack con Groq como juez (faithfulness, answer
relevancy, context relevance). Salida análoga, en su propio archivo de resultados. Se corre a
mano antes de un merge grande.

### Baseline y delta

- `run_eval.py` compara siempre contra `baseline.json` y reporta ± por métrica y por categoría.
- `run_eval.py --set-baseline` promueve la última corrida a `baseline.json`.
- `dataset.yaml` y `baseline.json` **se commitean** (referencia compartida por el equipo);
  `results/` va al `.gitignore`.

## Estructura de archivos propuesta

```
src/pipeline/eval/
  dataset.yaml            # preguntas + ground truth (versionado)
  run_eval.py             # Tier 1 (retrieval) + Tier 2 (SAS) — gate diario
  run_eval_llm.py         # Tier 3 (juez LLM con Groq) — manual
  metrics.py              # recall@k, hit@k, mrr, coseno/SAS
  report.py               # tabla + delta vs baseline + desglose por pregunta/categoría
  baseline.json           # snapshot de referencia (versionado)
  results/                # corridas timestamped (gitignored)
docs/eval/eval_harness.md # este documento
```

## Comandos (target final)

```bash
# Gate diario (retrieval + SAS): correr tras cualquier cambio
docker compose exec pipelines python /app/pipelines/eval/run_eval.py

# Promover la corrida actual a baseline
docker compose exec pipelines python /app/pipelines/eval/run_eval.py --set-baseline

# Capa profunda (juez LLM): manual, antes de un merge grande
docker compose exec pipelines python /app/pipelines/eval/run_eval_llm.py
```

Loop de trabajo: cambio algo (retriever, chunking, prompt, LLM, valves) → `run_eval` → leo el
delta por categoría → decido si el cambio entra.

## ¿Cuándo se ejecuta? — decisión: manual, antes de mergear

El harness **no es un test unitario**: necesita el stack docker levantado (pgvector poblado,
Ollama, `GROQ_API_KEY`) y gasta llamadas a Groq. No corre en cada save.

**Decisión adoptada:** ejecución **manual, antes de mergear un cambio.** Se toca una pieza
(retriever / chunking / prompt / LLM / valves), se corre `run_eval.py` a mano, se lee el delta
por categoría y recién ahí se decide si el cambio entra y se commitea/mergea. Es el piso: cero
infra extra y da el grueso del valor ("cambio → script → delta").

Descartado por ahora (se puede reconsiderar más adelante):
- **Git hook `pre-push`:** el hook corre en el host pero el eval necesita el container arriba;
  frágil si el stack no está levantado. Quedaría como opcional no bloqueante para Tier 1+2.
- **CI en cada PR:** levantar y **poblar** pgvector + Ollama en el runner es pesado, y Groq en
  CI arriesga 429. Fuera de alcance por ahora; si algún día entra, solo Tier 1+2, nunca el juez.

**Orden crítico si el cambio implica reindexar** (chunking, converters, modelo de embeddings):
primero **reindexar el store**, después correr el eval. Si no, se estaría midiendo el retriever
nuevo contra chunks viejos. Para cambios que no tocan la indexación (prompt, LLM, `top_k`, RRF,
`build_keyword_query`) se corre directo.

Por tier: **Tier 1 + 2** se corren juntos en cada evaluación (barato, ~1 call Groq/pregunta).
**Tier 3** (juez LLM) se corre aparte y puntual, antes de un merge grande, por el riesgo de 429.

## Fases de implementación — estado

1. **Fase 0 — Habilitar headless. ✅** Verificado que el pipelines-server usa `os.listdir` (no
   escanea subdirs, así que `eval/` no se auto-carga). `get_document_store` y `build_rag_pipeline`
   extraídas a funciones módulo-level; la clase `Pipeline` delega → runtime idéntico.
2. **Fase 1 — Tier 1. ✅** `dataset.yaml` (27 preguntas) + `metrics.py` + `run_eval.py` con
   retrieval, desglose por categoría y contribución por retriever.
3. **Fase 2 — Tier 2 + reporte. ✅** SAS local (bge-m3 vía Ollama) + `report.py` con baseline y
   delta (global, por categoría y regresiones por pregunta). Loop "cambio → script → delta" cerrado.
4. **Fase 3 — Tier 3. ✅** `run_eval_llm.py` con `FaithfulnessEvaluator` y
   `ContextRelevanceEvaluator` de Haystack, usando Groq como juez en modo JSON. Manual, con `--limit`.

Umbral de regresión de SAS: `SAS_REGRESSION_THRESHOLD = 0.05` en `report.py`. Se observó que las
negativas tienen SAS algo ruidoso aun con `temp=0` (variación ~0.06 en la respuesta del LLM), así
que ese umbral puede necesitar subirse si genera falsos positivos.

## Decisiones abiertas

- **Curado del ground truth de `expected_doc_ids`:** requiere trabajo manual de mirar el store
  para las categorías `nombre_cve` / `multi_doc` (no basta el concepto, hay que saber qué ids
  están indexados). Definir si se cura una vez y se congela, o se re-cura al reindexar el corpus.
- **Ragas vs evaluators nativos de Haystack** para Tier 3: se recomienda Haystack nativo para no
  sumar dependencia; revisar si se necesita alguna métrica específica de Ragas (p. ej. context
  precision/recall con referencia).
- **Umbral de regresión:** decidir a partir de qué caída de métrica el harness marca "falla"
  (para eventualmente cablearlo a CI), y con cuánta varianza de temp 0 se convive.

## Relacionado

- [mejoras_harness.md](mejoras_harness.md) — plan priorizado de mejoras sobre este harness
  (dual-`k`, abstención, nDCG/tokens, recalibrar SAS, baseline de Tier 3). **Propuesto.**
- [optimizacion_keyword_retrieval.md](../optimizacion_keyword_retrieval.md) — el banco de queries de
  su sección "Verificación" es la semilla natural de las categorías `id_*` y `concepto_es`.
- [analisis_keyword_retriever.md](../analisis_keyword_retriever.md) — evidencia de logs que motiva
  medir la contribución por retriever (Tier 1).
- [data_splitting.md](../data_splitting.md) / [enriquecimiento_de_chunks.md](../enriquecimiento_de_chunks.md)
  — cambios de chunking que este harness permitiría evaluar objetivamente. `data_splitting.md`
  **extiende** este harness con **Tier 1b `source_recall@k`** (ground truth a nivel de archivo,
  estable entre estrategias de chunking) y la categoría de dataset `guia_incibe`, más un
  orquestador propio ([run_chunking_experiment.py](../../src/pipeline/eval/run_chunking_experiment.py))
  que compara estrategias en tablas pgvector separadas.
```
