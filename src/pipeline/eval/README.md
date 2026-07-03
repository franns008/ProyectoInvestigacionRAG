# Eval harness del RAG de ciberseguridad

Mide el impacto de cualquier cambio en el pipeline (retriever, chunking, embeddings,
prompt, LLM, valves) corriendo un dataset de preguntas con ground truth y comparando
las métricas contra un **baseline**.

Diseño completo y racional: [../../../docs/eval_harness.md](../../../docs/eval_harness.md).

## Requisitos

Se corre **dentro del container `pipelines`** (ahí resuelven `vdb`, `ollama` y
`GROQ_API_KEY`). El stack tiene que estar levantado:

```bash
cd infrastructure && docker compose up -d
```

El harness **lee el store ya poblado**; no reindexa. Si tu cambio toca la indexación
(chunking, converters, modelo de embeddings), **reindexá primero** y después evaluá.

## Uso

```bash
# Gate diario: corré esto tras cualquier cambio y mirá el delta por categoría
docker compose exec pipelines python /app/pipelines/eval/run_eval.py

# Fijar la corrida actual como baseline de referencia
docker compose exec pipelines python /app/pipelines/eval/run_eval.py --set-baseline

# Experimentar sin tocar código (override de parámetros)
docker compose exec pipelines python /app/pipelines/eval/run_eval.py --top-k 8
docker compose exec pipelines python /app/pipelines/eval/run_eval.py --limit 5

# Tier 3 (juez LLM): MANUAL, gasta muchas llamadas a Groq → usar --limit
docker compose exec pipelines python /app/pipelines/eval/run_eval_llm.py --limit 8
```

Ejecutar los comandos `docker compose` desde `infrastructure/`.

## Flujo de trabajo

1. Corré `run_eval.py` y `--set-baseline` una vez (referencia inicial).
2. Hacé un cambio en el pipeline.
3. Corré `run_eval.py` → leé el bloque **DELTA vs BASELINE** (global, por categoría,
   regresiones por pregunta).
4. Si el cambio es bueno, promovélo con `--set-baseline`.

## Los tres tiers de métricas

| Tier | Qué mide | Cuándo | Costo |
|---|---|---|---|
| **1 — Retrieval** | recall@k, hit@k, MRR contra `expected_doc_ids` | siempre (`run_eval.py`) | gratis, determinístico |
| **2 — SAS** | coseno entre respuesta y `reference_answer` (bge-m3 local) | siempre (`run_eval.py`) | gratis, local |
| **3 — Juez LLM** | faithfulness + context relevance (Groq como juez) | manual (`run_eval_llm.py`) | caro, rate-limited |

`run_eval.py` hace **1 generación Groq por pregunta**; Tier 3 hace varias por pregunta.

## Dataset (`dataset.yaml`)

Lista de preguntas. Campos:

```yaml
- id: cwe89-por-numero          # slug estable (aparece en el reporte)
  category: id_cwe              # categoría de la taxonomía (ver el diseño)
  question: "Explicame la vulnerabilidad CWE-89 y cómo prevenirla."
  expected_doc_ids: ["cwe-89"]  # ground truth de retrieval. [] = sin doc esperado
  reference_answer: |           # opcional; habilita Tier 2 (SAS) y Tier 3
    CWE-89 es SQL Injection: ...
  eval_focus: [retrieval, generation]
  tags: [injection, es]
```

### Cómo agregar preguntas

1. Elegí la pregunta y su categoría.
2. Averiguá los `expected_doc_ids` **reales** consultando el store. Los ids son
   determinísticos: CWE → `cwe-<n>`, CVE → `sha256(cve_id)`. Para verificar qué hay
   indexado:

   ```bash
   docker compose exec pipelines python -c "
   import sys; sys.path.insert(0,'/app/pipelines')
   import pipeline_ciberseguridad as p
   store=p.get_document_store()
   d={x.id:x for x in store.filter_documents()}
   print('cwe-89' in d, d.get('cwe-89').meta.get('name') if 'cwe-89' in d else None)"
   ```

3. Escribí una `reference_answer` breve si querés medir la respuesta (Tier 2/3).
4. Corré `run_eval.py` para ver que el ground truth es alcanzable.

> **CVEs:** hoy el corpus tiene 710 CWE y **0 CVE**. Las preguntas de CVE están con
> `expected_doc_ids: []` como tests de robustez (el RAG debe decir "no sé"). Cuando se
> indexen CVEs, cambiales el `expected_doc_ids` al `sha256(cve_id)` y la `category`.

## Persistencia de resultados

Cada corrida de `run_eval.py` deja tres archivos en `results/` (bind-mount al host →
**persisten entre reinicios del container**):

| Archivo | Contenido |
|---|---|
| `<timestamp>.json` | snapshot completo: config, métricas globales/por categoría y **por pregunta** (con la respuesta generada y los docs recuperados) |
| `<timestamp>.log` | el volcado de consola de esa corrida, en texto plano (legible) |
| `history.csv` | **una fila por corrida** (timestamp, top_k, temp, modelos, recall/hit/mrr/sas) — para ver la evolución en el tiempo |

`run_eval_llm.py` guarda su propio `tier3_<timestamp>.json`.

`results/` está gitignoreado (son artefactos regenerables); lo que **sí se versiona** es
`baseline.json`. Si querés compartir la evolución con el grupo, versioná `history.csv` a mano.

## Interpretar la salida

- `✓ / ✗` por pregunta con `recall`, `rr` y `via=` (qué retriever aportó el acierto:
  `embedding`, `keyword` o ambos). `–` = negativa (sin ground truth de retrieval).
- Resumen con SAS global y retrieval **por categoría** (ahí se ve qué tipo de pregunta
  mejoró o empeoró).
- DELTA: `▲`/`▼` por métrica y la lista de preguntas que regresaron.

## Archivos

| Archivo | Rol |
|---|---|
| `dataset.yaml` | preguntas + ground truth (versionado) |
| `run_eval.py` | runner Tier 1 + 2, baseline/delta |
| `run_eval_llm.py` | runner Tier 3 (juez LLM), manual |
| `metrics.py` | recall@k, hit@k, MRR, coseno/SAS |
| `report.py` | delta contra baseline + regresiones |
| `baseline.json` | referencia versionada (`--set-baseline` la actualiza) |
| `results/` | corridas timestamped (gitignored) |
