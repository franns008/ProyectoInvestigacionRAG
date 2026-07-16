# Plan de mejoras del harness de evaluación

> **Estado (2026-07-16): PROPUESTO — no implementado.** Documenta los refactors
> priorizados sobre el harness actual ([eval_harness.md](eval_harness.md) es el diseño
> vigente). Cada inciso arranca con **el problema** que resuelve, después el paso a paso.

Este documento parte de un análisis crítico de qué tan confiables son hoy las métricas
del harness. La conclusión de fondo: la arquitectura por tiers está bien, pero hay
problemas de **medición** — qué `k` se mide vs. cuál se reporta, un baseline obsoleto, un
`n` que no da para estadística, y que las dos cosas más importantes para *este* sistema
(no alucinar y el reranker recién agregado) son justo las peor medidas.

Los grupos están ordenados por retorno/costo. Ver el [orden de implementación](#orden-de-implementación) al final.

## Refactor compartido (habilitante)

**Problema:** la extracción de tokens (normalización Groq vs. Ollama) vive embebida dentro
de `pipe()` ([pipeline_ciberseguridad.py:658-670](../../src/pipeline/pipeline_ciberseguridad.py#L658-L670)),
así que el eval no puede reusarla y tendría que duplicarla para H9.

**Paso a paso:**
1. Extraer a una función módulo-level `extract_usage(reply_meta) -> {prompt, completion, total}`.
2. `pipe()` pasa a delegar en ella (runtime idéntico), y el eval la importa desde `rag`.

---

## H1 + H2 — Registrar los dos `k` + regenerar baseline

**Problema:** las métricas Tier 1 se calculan sobre la salida del **reranker**
(`ranker_top_k=4`, [run_eval.py:101](../../src/pipeline/eval/run_eval.py#L101)), pero el
snapshot guarda `top_k = retriever_top_k = 15`
([run_eval.py:243](../../src/pipeline/eval/run_eval.py#L243)). Es un `recall@4` disfrazado
de `recall@15`: por eso en `history.csv` subir `top_k` de 3 a 15 dejó el recall clavado en
0.8409. Además el `baseline.json` commiteado es **pre-reranker** (`config.top_k=3`, sin etapa
`ranker`), así que el delta compara peras con manzanas — sobre todo el MRR, que depende del
orden. Con esto, **todos los deltas actuales engañan**.

**Paso a paso (H1):**
1. En `run_question` ([run_eval.py:84](../../src/pipeline/eval/run_eval.py#L84)) leer también
   la salida del `document_joiner` (ya está en `include_outputs_from`, pero no se lee) y
   devolver `joined_ids` además de `ranked_ids`.
2. En `evaluate()` calcular recall/hit/rr en **dos puntos** y arreglar el nombre engañoso del
   parámetro (`joined_ids` hoy recibe el ranked):
   - `recall` / `hit` / `rr` → sobre `ranked_ids` (**efectivo**, lo que ve el LLM).
   - `recall_retriever` / `hit_retriever` → sobre `joined_ids` (**techo** del retriever).
   - El gap `recall_retriever − recall` = cuánto recall cuesta el reranker (hoy invisible).
3. Registrar ambos k en `config`: `retriever_top_k` y `ranker_top_k` por separado.
4. Extender `aggregate_retrieval` / `aggregate_by_category` en
   [metrics.py](../../src/pipeline/eval/metrics.py) para promediar también las variantes
   `_retriever`.
5. Mostrar ambos recalls en `_print_summary` y en `report.print_delta`.
6. Arreglar `history.csv`: cambiar la columna `top_k` por `retriever_top_k` + `ranker_top_k`.
   **Gotcha:** `DictWriter` solo escribe header si el archivo no existe; cambiar columnas a
   mitad de archivo rompe el append → rotar (mover el actual a `history_pre_reranker.csv`).

**Paso a paso (H2):**
7. Regenerar el baseline con el pipeline actual (tras aterrizar el bloque de métricas):
   `run_eval.py --set-baseline`. Commitear.
8. **Guarda contra la recurrencia (opcional, recomendado):** guardar en el snapshot un
   `pipeline_signature` (hash de nombres de componentes + `ranker_model` + los `top_k` +
   `split_length`). Si la firma actual ≠ la del baseline, `print_delta` avisa "baseline
   potencialmente obsoleto" → el bug se auto-denuncia.
9. Documentar la norma: re-promover baseline cuando cambia la **topología**, no solo los valves.

---

## H6 — Métrica determinística de abstención

**Problema:** las negativas (`fuera_dominio`, `id_inexistente`) no tienen `expected_doc_ids`,
así que Tier 1 las saltea, su SAS es ruido y faithfulness solo corre manual (Tier 3). No hay
**ningún número** que diga "abstuvo correctamente" — y no alucinar es un objetivo central del
sistema. Hoy se verifica a ojo.

**Paso a paso:**
1. Definir marcadores de rechazo en [metrics.py](../../src/pipeline/eval/metrics.py)
   (sacados de las respuestas reales del baseline: "no sé", "no dispongo", "no tengo
   información", "fuera del alcance"…) y una función `is_refusal(answer)`.
2. Detectar ids inventados reusando `rag._extract_vuln_ids` (desde `run_eval.py`, que ya
   importa `rag`). Clave anti-falso-positivo: **restar los ids que ya están en la pregunta**
   para no penalizar que la respuesta repita el CVE del enunciado:
   `fabricated = answer_ids − question_ids`.
3. Veredicto en `evaluate()` para items con `expected_doc_ids == []`:
   `abstention = { refused, fabricated, correct = refused and not fabricated }`.
   Esto atrapa el caso real `cve-log4j-nombre`, cuya respuesta soltó "CWE-779, CWE-464,
   CWE-244" (ruido inventado) → `correct=False`.
4. `aggregate_abstention(per_question)` → `{rate, n}` sobre las negativas; sumarlo al snapshot.
5. Mostrar en resumen, delta y regresiones (negativa que pasa de `correct=True` a `False`).
6. (Opcional) campos de dataset `expect_refusal` / `forbid_ids`, con default inferido de
   `expected_doc_ids` vacío, para dejar lugar a las futuras CVE que **sí** tendrán ids.

---

## H3 — Crecer el dataset a ~8/categoría (en las que importan)

**Problema:** con `n=22` y categorías de `n=1–3`, un cambio de estado en una sola pregunta
mueve una categoría de 1.0 a 0.5. Los deltas por categoría son anecdóticos, no estadísticos,
y no hay intervalos de confianza. Es la palanca de fondo. **Riesgo:** crecer mal reintroduce
la contaminación de ground truth (un LLM curando labels no es independiente del sistema que se
evalúa).

**Paso a paso:**
1. Priorizar: subir primero `concepto_es` (recall 0.33), `multi_doc` (0.25), `nombre_cwe`,
   `desarrollar` y las negativas (ahora que H6 las mide). Dejar chicas `id_cwe`/`id_variantes`
   (ya casi perfectas y objetivas).
2. **Fijar la fuente de verdad ANTES de escribir** (anti-circularidad): anclar cada
   `expected_doc_ids` a **MITRE**, no a lo que devuelve el RAG. Agregar campo `gt_source`.
3. Para multi-doc/concepto: introducir `acceptable_doc_ids` (superset que también cuenta como
   acierto) distinto de `expected_doc_ids` (el ideal), y un `soft_hit`/nDCG graded contra el
   aceptable para no generar falsas regresiones.
4. Curado en **dos pasadas independientes** para las de juicio: dos fuentes proponen ids, se
   difean, se resuelven desacuerdos; marcar las dudosas como baja confianza.
5. Versionar y (opcional) separar `split: dev|test` para reportar el número sobre held-out y
   no sobreajustar `build_keyword_query` al set.
6. Correr y re-promover baseline (después de que las métricas estén honestas).

---

## H10 + H9 — nDCG + tokens (para juzgar el reranker)

**Problema:** acaban de agregar un reranker cross-encoder cuyo único trabajo es **ordenar
mejor**, pero ninguna métrica actual lo evalúa bien: `recall@4` es insensible al orden dentro
del top-4 y `MRR` solo mira el primer acierto. Y su beneficio declarado — "menos docs → menos
tokens" — no se mide: el pipeline ya loguea tokens
([pipeline_ciberseguridad.py:675](../../src/pipeline/pipeline_ciberseguridad.py#L675)) pero el
harness no los persiste. Se está optimizando a ciegas el eje costo/eficiencia.

**Paso a paso (H10 — nDCG@k):**
1. Implementar `dcg_at_k` / `ndcg_at_k` en [metrics.py](../../src/pipeline/eval/metrics.py)
   con relevancia binaria desde `expected_doc_ids`.
2. Calcular sobre `ranked_ids` (lo que el reranker controla); guardar per-question y agregar.
   Nota: con doc único nDCG ≈ MRR descontado; su valor real aparece en multi-doc y con el
   graded de H3.

**Paso a paso (H9 — tokens y latencia):**
3. Reusar `extract_usage(...)` (refactor compartido) y capturar en `run_question`: `prompt`
   (de `prompt_builder`), `n_docs` (de `ranker`), tokens (de `llm.replies[0].meta`) y latencia
   (envolver `pipeline.run()` con `perf_counter`).
4. Guardar per-question y agregar medias en un bloque `cost` del snapshot.
5. Mostrar en `history.csv` y en el delta. Ahora el trade-off del reranker es legible:
   *recall_efectivo ▼0.02 pero prompt_tokens ▼40%*.

---

## H5 + H7 — SAS confiable + Tier 3 con baseline y answer relevancy

**Problema:** el gate de generación es débil por dos lados. **SAS** (Tier 2) es ruidoso — dos
corridas de config idéntica dieron 0.7722 vs 0.7654, y el umbral de regresión es 0.05
([report.py:15](../../src/pipeline/eval/report.py#L15)), apenas por encima del ruido —, tiene
sesgo de longitud (referencia corta vs. respuesta larga) y es circular (bge-m3 evalúa lo que
bge-m3 recuperó). **Tier 3** (faithfulness, lo que realmente atrapa alucinaciones) no tiene
baseline ni delta, no se trackea, y la `answer relevancy` prometida en el diseño no está
implementada.

**Paso a paso (H5 — recalibrar SAS):**
1. Medir el piso de ruido: correr la **misma** config N=5 veces (temp=0) y calcular la
   desviación de SAS por pregunta y global.
2. Fijar `SAS_REGRESSION_THRESHOLD` por encima del piso (si el ruido por-pregunta ~0.06, poner
   0.08–0.10). Documentar valor y fecha de medición.
3. Degradar el rol de SAS a "tripwire grueso" en el reporte; **excluir las negativas** del
   agregado (su SAS es ruido; ahora las mide H6).
4. (Opcional) atacar el sesgo de longitud (similitud máxima por oración, o registrar ratio).

**Paso a paso (H7 — Tier 3):**
5. Agregar answer relevancy a [run_eval_llm.py](../../src/pipeline/eval/run_eval_llm.py).
   **Verificar el nombre exacto del evaluator en la versión de Haystack**: ship
   `FaithfulnessEvaluator` y `ContextRelevanceEvaluator` nativos, pero answer-relevance puede
   no tener componente dedicado → usar el genérico `LLMEvaluator` con una rúbrica (reusando
   `make_judge`), o sumar Ragas si se quiere su versión calibrada.
6. Baseline + delta para Tier 3: replicar el mecanismo de [report.py](../../src/pipeline/eval/report.py)
   en un `baseline_tier3.json`; agregar `--set-baseline` a `run_eval_llm.py`; persistir
   `per_question` para listar regresiones.
7. Histórico `history_tier3.csv` análogo.
8. (Opcional, recomendado) calibrar el juez: etiquetar a mano ~15 `(pregunta, contexto,
   respuesta)` y medir el acuerdo con el juez.

---

## Orden de implementación

Agrupado para regenerar el baseline **una sola vez**:

1. **Refactor compartido** — extraer `extract_usage()`.
2. **Bloque de métricas (mismo PR):** H1 (dual-k + dual recall) + H10 (nDCG) + H9
   (tokens/latencia) + H6 (abstención). Todos tocan `metrics.py` + `run_eval.py` + `report.py`.
3. **H2** — regenerar y commitear `baseline.json` (con `pipeline_signature`).
4. **H5** — medir ruido y ajustar umbral (paralelo; toca `report.py` + unas corridas).
5. **H3** — crecer dataset con anclaje a MITRE y graded → dispara una **segunda** regeneración
   de baseline.
6. **H7** — Tier 3 (archivo independiente, paralelo al resto).

**Dependencias duras:** H9 depende del refactor compartido; H2 depende de que
H1/H10/H9/H6 estén (para que el baseline nazca honesto); H3 depende de H1/H2 (métricas
honestas) y de H6 (para puntuar negativas).

## Relacionado

- [eval_harness.md](eval_harness.md) — diseño vigente del harness (Tiers 1–3, dataset, flujo).
- [../reranker_cross_encoder.md](../reranker_cross_encoder.md) — el reranker que H10/H9 buscan
  poder evaluar objetivamente.
- [../optimizacion_keyword_retrieval.md](../optimizacion_keyword_retrieval.md) — el preprocesado
  de query cuyo sobreajuste al dataset previene H3 (split dev/test).
