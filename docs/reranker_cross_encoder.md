# Reranker cross-encoder (bge-reranker-v2-m3)

## Objetivo

Mejorar la **relevancia** de los documentos que llegan al prompt del LLM y, de paso,
**reducir tokens** en la generación. Se logra recuperando ancho en el retrieval y
filtrando fino con un cross-encoder que re-puntúa los candidatos por relevancia real a
la pregunta, dejando solo los mejores en el prompt.

## Contexto y problema

La búsqueda es híbrida: `PgvectorEmbeddingRetriever` (semántica) +
`PgvectorKeywordRetriever` (full-text) unidos por `DocumentJoiner`
(reciprocal rank fusion) en
[pipeline_ciberseguridad.py](../src/pipeline/pipeline_ciberseguridad.py).

Antes de este cambio:

- `retriever_top_k = 3`: cada retriever traía solo 3 docs. Un documento relevante que
  no entraba en el top-3 de **ninguno** de los dos retrievers se perdía para siempre —
  no había forma de recuperarlo aguas abajo.
- El `DocumentJoiner` **no tenía `top_k`**: al prompt llegaba la unión fusionada
  completa (hasta ~6 docs distintos), sin techo duro ni un segundo filtro de relevancia.

Subir el `top_k` a secas no alcanza: mete más ruido al prompt (más tokens, peor foco del
LLM) porque ni la similitud de embeddings ni el rank RRF son un juicio fino de "qué tan
bien responde este doc a *esta* pregunta".

## Concepto que lo justifica: retrieve-and-rerank (bi-encoder → cross-encoder)

Los retrievers son **bi-encoders**: embeben query y documento por separado y comparan
vectores. Es barato y escala a todo el store, pero pierde precisión porque nunca mira
query y doc juntos. Sirve para **recuperar ancho** (alto recall) barato.

Un **cross-encoder** (reranker) recibe el par `(query, documento)` *junto* y produce un
score de relevancia mirando la interacción palabra-a-palabra entre ambos. Es mucho más
preciso, pero también mucho más caro: hay que correr el modelo una vez por cada par, así
que no se puede aplicar a todo el store — solo a un puñado de candidatos.

El patrón estándar (**retrieve-and-rerank**) combina ambos:

1. Los bi-encoders recuperan **ancho** (top_k alto) → maximizan recall barato.
2. El cross-encoder **re-rankea** ese conjunto chico y se queda con los `k` mejores →
   maximiza precisión donde importa (lo que ve el LLM).

Elegimos `BAAI/bge-reranker-v2-m3`: es el **cross-encoder compañero** del `bge-m3` que ya
usamos para embeddings (misma familia/entrenamiento, multilingüe es↔en, clave porque los
CWE están en inglés y las preguntas en español), corre en **CPU** y es **gratis** (sin
API externa).

## Cambios

### Pipeline de query — [pipeline_ciberseguridad.py](../src/pipeline/pipeline_ciberseguridad.py)

- Nuevo componente `ranker` (`SentenceTransformersSimilarityRanker`, la variante vigente
  de Haystack; `TransformersSimilarityRanker` está deprecada) intercalado entre el joiner
  y el prompt: `embedding_retriever + keyword_retriever → document_joiner → ranker → prompt_builder → llm`.
- Nuevas valves:

  | Valve | Antes | Ahora | Rol |
  |---|---|---|---|
  | `retriever_top_k` | 3 | **15** | cada retriever recupera ancho |
  | `ranker_model` | — | `BAAI/bge-reranker-v2-m3` | modelo del cross-encoder |
  | `ranker_top_k` | — | **4** | techo duro de docs al prompt |

- `DocumentJoiner(top_k=retriever_top_k * 2)`: cap de candidatos que entran al reranker
  (con dos retrievers de 15 la unión fusionada puede llegar a ~30).
- El **techo duro real** de docs al prompt pasa a ser `ranker_top_k` (4), no el joiner.
- `pipe()` le pasa la pregunta al ranker (`"ranker": {"query": user_message}`).
- Logging: el bloque del joiner se rotula "candidatos fusionados" y se agrega un bloque
  `[RANKER]` con los docs finales al prompt y su **score de relevancia**.

El reranker descarga el modelo de HuggingFace en `warm_up()` (Haystack lo invoca solo al
primer `run`). Los embeddings (`bge-m3` en Ollama) **no cambian**.

### Imagen del container — [Dockerfile.pipelines](../infrastructure/Dockerfile.pipelines)

Se instala `sentence-transformers` y se **pre-descarga** `bge-reranker-v2-m3` al buildear
(≈2.3 GB), para que el primer arranque no espere la descarga y el container funcione
offline. Usa el mismo torch CPU ya fijado; corre en CPU sin cambios extra.

### Eval harness — [run_eval.py](../src/pipeline/eval/run_eval.py)

Las métricas Tier 1 (recall@k / hit_rate / mrr) se calculan ahora sobre la **salida del
reranker** (lo que realmente ve el LLM), no sobre los candidatos del joiner.

> **Ojo con el baseline:** `recall@k` cambió de semántica — ahora `k = ranker_top_k = 4`
> y sobre docs re-rankeados. El baseline anterior (top_k=3, sin reranker) **no es
> comparable directo**. Tras validar la mejora, re-generar con `--set-baseline`.

## Implicancias operativas

Sumar el reranker no es gratis en recursos, aunque no cueste dinero. Qué implica en la
práctica:

- **Modelo nuevo que descargar (~2.3 GB).** `bge-reranker-v2-m3` (568M parámetros) se baja
  de HuggingFace. Lo pre-descargamos en el build de la imagen (ver Dockerfile), así que la
  **imagen del container pesa ~2.3 GB más** y el **build tarda más** (y necesita red la
  primera vez). A cambio, el arranque del container **no** espera la descarga y funciona
  offline. Es un modelo distinto y aparte del `bge-m3` de embeddings (que vive en Ollama):
  este corre **dentro del proceso del pipeline**, vía `sentence-transformers`/torch.

- **Corre en CPU, en el propio container `pipelines`.** No usa Ollama ni GPU (aunque
  aprovecharía CUDA si estuviera). Consume **RAM** (el modelo + torch, ~1–2 GB residentes)
  mientras el pipeline está vivo, y **CPU** en cada query.

- **Carga diferida en el primer `run` (warm_up).** Haystack hace `warm_up()` la primera
  vez que corre el pipeline: ahí carga el modelo a memoria (unos segundos). La **primera
  consulta tras levantar** el container es más lenta; las siguientes ya tienen el modelo
  caliente.

- **Latencia por query.** Se agrega el forward pass del cross-encoder sobre los candidatos
  (hasta `retriever_top_k * 2` ≈ 30 pares `query,doc`). En CPU son típicamente **cientos de
  ms a ~1–2 s** según hardware y largo de los docs. Es el trade-off central: más latencia de
  retrieval a cambio de mejor relevancia y **menos tokens** enviados al LLM (el reranker
  recorta de ~6 a 4 docs), lo que puede **compensar** en latencia total de generación.

- **Sin costo de API ni de red en runtime.** Todo local; no toca Groq ni cuotas externas.

- **Reconstrucción del pipeline.** Al cambiar valves (`on_valves_updated`) el pipeline se
  rearma; el modelo ya está en disco (pre-bakeado) pero se vuelve a hacer `warm_up` en el
  primer `run` del pipeline nuevo.

## Resultados del eval

Corrida del harness (2026-07-10) — 27 preguntas (22 con ground truth, 14 con SAS),
`temp=0.0`, 965 docs en store. Comparado contra el baseline previo (top_k=3, sin reranker):

| Métrica | Baseline (top_k=3, sin reranker) | Con reranker (top_k 15→4) | Δ |
|---|---|---|---|
| recall@k | 0.841 | 0.841 | =0.000 |
| hit_rate | 0.864 | 0.864 | =0.000 |
| mrr | 0.693 | 0.708 | **▲0.015** |
| sas | ~0.755 | 0.765 | **▲0.010** |

**Lectura.** El reranker **no cambió recall/hit** (los retrievers deciden qué se recupera;
el reranker solo reordena y trunca) pero **mejoró el orden (mrr) y la calidad de respuesta
(sas)** recortando los docs al prompt de ≤6 a 4. Es el resultado teórico esperado de un
retrieve-and-rerank: misma cobertura, docs relevantes más arriba, contexto más limpio y
menos tokens.

**Intercambio por categoría (recall global plano = una recuperada + una perdida):**
- `concepto_es` ▲0.333 (0.333→0.667): subir `top_k` a 15 rescató una pregunta conceptual
  que con top_k=3 el embedding no alcanzaba a traer (beneficio de "recuperar ancho").
- `desarrollar` ▼0.500: regresión en `xss-explicar-prevenir` (recall 1.00→0.00). El doc
  relevante **sí lo trajo el retrieval** (`via=embedding+keyword`) pero el reranker lo dejó
  **fuera del top-4**. En preguntas amplias de "explicá y prevení", el cross-encoder
  subestimó el doc y el corte a 4 lo descartó.

**Caveats.** La regresión de `capital-francia` (SAS 0.637→0.535) es ruido: pregunta fuera
de dominio (negativa). Entre dos corridas el SAS varió (~0.772 vs 0.765) por no-determinismo
del LLM aún a `temp=0`; la mejora de SAS es real pero modesta/ruidosa — la de MRR es más sólida.

**Conclusión.** El reranker rinde como se esperaba (mejor orden + menos tokens, misma
cobertura), pero **`ranker_top_k=4` es algo agresivo para preguntas amplias multi-doc**.
Pendiente a evaluar: subir `ranker_top_k` a 5 o agregar `score_threshold` para recuperar
`xss-explicar-prevenir` sin perder el ahorro de tokens. El `recall@k` cambió de semántica
(ahora k=4 sobre docs re-rankeados), así que este es el **nuevo baseline** de referencia.

## Verificación (end-to-end)

1. **Build:** `docker compose -f infrastructure/docker-compose.yml build pipelines` —
   deben pasar las verificaciones nuevas del Dockerfile (sentence-transformers, descarga
   del modelo, import del ranker).
2. **Arranque + consulta real** (modo Groq por defecto, ver [modos_llm.md](modos_llm.md)):
   preguntar por XSS o un CVE concreto y revisar `src/pipeline/logCiberseguridad.txt`.
   El bloque `[RANKER]` debe mostrar ~4 docs finales con su score; el `[DOCUMENT JOINER]`,
   más candidatos.
3. **Eval:** `docker compose exec pipelines python /app/pipelines/eval/run_eval.py`
   (ver [eval_harness.md](eval/eval_harness.md)). Comparar contra el baseline; si mejora,
   re-correr con `--set-baseline`.

## Relacionado

- [optimizacion_keyword_retrieval.md](optimizacion_keyword_retrieval.md) — preprocesado
  de la query para el keyword retriever (aguas arriba del reranker).
- [eval_harness.md](eval/eval_harness.md) — cómo se miden recall/hit/mrr/sas.
