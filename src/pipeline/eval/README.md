# Eval harness del RAG de ciberseguridad

Mide el impacto de cualquier cambio en el pipeline (retriever, chunking, embeddings,
prompt, LLM, valves) corriendo un dataset de preguntas con ground truth y comparando
las métricas contra un **baseline**.

Diseño completo y racional: [../../../docs/eval/eval_harness.md](../../../docs/eval/eval_harness.md).
Persistencia en CSV: [../../../docs/eval/refactor_resultados_csv.md](../../../docs/eval/refactor_resultados_csv.md).

## Requisitos

Se corre **dentro del container `pipelines`** (ahí resuelven `vdb`, `ollama` y
`GROQ_API_KEY`). El stack tiene que estar levantado:

```bash
cd infrastructure && docker compose up -d
```

El harness **lee el store ya poblado**; no reindexa. Si tu cambio toca la indexación
(chunking, converters, modelo de embeddings), **reindexá primero** y después evaluá.

## Uso

Usá siempre los wrappers de `scripts/`, desde la raíz del repo. No son un atajo
cosmético: inyectan el commit, la rama y si había cambios sin pushear, y eso es lo que
después te deja saber **con qué código** se generó cada número.

```bash
# Gate diario: corré esto tras cualquier cambio
scripts/eval.sh

# Experimentar sin tocar código (override de parámetros)
scripts/eval.sh --top-k 8
scripts/eval.sh --limit 5 --label "prueba rápida"

# Fijar la corrida actual como baseline de referencia
scripts/eval.sh --set-baseline

# Tier 3 (juez LLM): MANUAL, gasta muchas llamadas al proveedor → usar --limit
scripts/eval_llm.sh --limit 8
```

Para mirar resultados ya guardados (no corre el eval, sólo lee las tablas):

```bash
docker compose exec pipelines python /app/pipelines/eval/report.py                    # última vs baseline
docker compose exec pipelines python /app/pipelines/eval/report.py runs               # historial + evolución
docker compose exec pipelines python /app/pipelines/eval/report.py runs --all         # incluye las corridas judge
docker compose exec pipelines python /app/pipelines/eval/report.py question cwe89-por-numero
```

## Flujo de trabajo

1. **Una vez:** corré `scripts/eval.sh --set-baseline` para fijar la referencia.
   Queda escrita en `eval_meta.yaml`, que se versiona → el baseline es del **equipo**,
   no de tu máquina.
2. Hacé un cambio en el pipeline.
3. Corré `scripts/eval.sh`. Al final imprime la comparación contra el baseline:
   advertencias, diff de config, delta global, delta por categoría y **qué preguntas
   regresaron**.
4. Si el cambio es bueno, promovélo con `--set-baseline`.

### Cuándo re-basear

Cambiar la **topología** del pipeline (agregar/sacar una etapa, cambiar el modelo de
embeddings, el chunking) hace que los deltas contra el baseline viejo no signifiquen
nada. En ese caso: declarar una `epoch` nueva en `eval_meta.yaml` y re-basear.

Mover un valor (`top_k`, `temperature`) **no** es cambio de topología: eso es
justamente lo que el delta tiene que medir.

`report.py` avisa solo cuando comparás corridas de epochs distintas.

## Los tres tiers de métricas

| Tier | Qué mide | Cuándo | Costo |
|---|---|---|---|
| **1 — Retrieval** | recall@k, hit@k, MRR contra `expected_doc_ids` | siempre (`run_eval.py`) | gratis, determinístico |
| **2 — SAS** | coseno entre respuesta y `reference_answer` (bge-m3 local) | siempre (`run_eval.py`) | gratis, local |
| **3 — Juez LLM** | faithfulness + context relevance | manual (`run_eval_llm.py`) | caro, rate-limited |

`run_eval.py` hace **1 generación por pregunta**; Tier 3 hace varias por pregunta.

**Las métricas de retrieval no dependen del LLM.** Verificado empíricamente
(2026-08-18): la misma corrida con Ollama y con Groq da delta **exactamente 0.000** en
`recall_eff`, `hit_eff`, `mrr_eff` y `source_recall`. Sólo el `sas_mean` se mueve. O
sea que si cambiás de proveedor, el retrieval sigue siendo comparable y el SAS no.

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
4. Corré `scripts/eval.sh --limit N` para ver que el ground truth es alcanzable.

> **CVEs:** hoy el corpus tiene 710 CWE y **0 CVE**. Las preguntas de CVE están con
> `expected_doc_ids: []` como tests de robustez (el RAG debe decir "no sé"). Cuando se
> indexen CVEs, cambiales el `expected_doc_ids` al `sha256(cve_id)` y la `category`.

## Persistencia de resultados

**Dos tablas acumulativas**, no un archivo por corrida. Todo lo que escribe y lee esas
tablas pasa por `csv_store.py`: nadie hace `csv.writer` ni `pd.read_csv` a mano.

| Archivo | Contenido | Git |
|---|---|---|
| `results/runs.csv` | **1 fila por corrida** — config + métricas globales | ✅ versionado |
| `results/questions.csv` | **1 fila por corrida × pregunta** — el detalle, con respuestas | ❌ gitignored (pesa) |
| `results/logs/<run_id>.log` | volcado de consola de esa corrida | ❌ gitignored |
| `eval_meta.yaml` | `baseline_run_id` + `epoch` vigente | ✅ versionado |

Las dos tablas se escriben **fila por fila**: si el proceso muere en la pregunta 20,
las 19 anteriores ya están guardadas.

`questions.csv` está gitignoreado, así que el detalle por pregunta de una corrida de
otra máquina no lo tenés. `report.py` lo detecta y degrada al delta global con un
aviso, en vez de romperse.

### Reglas de formato

Valen para las dos tablas:

| Regla | Por qué |
|---|---|
| Vacío = celda en blanco → NaN al leer | Distingue "no medido" de "midió 0" |
| Flags booleanos = `1` / `0` | Se leen como float → `.mean()` directo sobre la columna |
| Listas = separadas por `\|` | `expected_ids`, `retrieved_ids`, `retrieved_sources` |
| Saltos de línea = `\n` literal | **Cada fila del CSV es una línea física**, así `grep` y `wc -l` siguen sirviendo. Al mostrar texto, des-escapar con `csv_store.unescape_text()` |

### Arrancar a analizar con pandas

```python
import pandas as pd
runs = pd.read_csv("results/runs.csv")
q    = pd.read_csv("results/questions.csv")
df   = q.merge(runs[["run_id", "epoch", "label"]], on="run_id")
```

Para dtypes correctos (y que los vacíos queden NaN de verdad), mejor usar el módulo:

```python
import csv_store
runs = csv_store.load_runs("results/runs.csv")
q    = csv_store.load_questions("results/questions.csv")
```

## Diccionario de columnas

### El sufijo `_eff` — leé esto primero

`recall_eff`, `hit_eff`, `mrr_eff`, `rr_eff` son las métricas **efectivas**: se calculan
sobre la salida del **reranker**, que son los documentos que realmente le llegan al LLM.

Importa porque el pipeline es `retriever (15 docs) → reranker (4 docs) → LLM`. Un
documento que el retriever encontró pero el reranker descartó **no cuenta**: el LLM
nunca lo vio, así que no pudo usarlo para responder. Medir sobre los 15 daría un número
mejor y mentiroso.

### `runs.csv` — una fila por corrida

| Columna | Tipo | Significado |
|---|---|---|
| `run_id` | texto | Timestamp UTC compacto (`20260818T202351Z`). **Su orden alfabético es el cronológico**, de eso depende "la última corrida" |
| `timestamp_utc` | texto | Fecha ISO completa |
| `suite` | texto | `retrieval` (Tier 1+2) o `judge` (Tier 3). `report.py` sólo compara las `retrieval` |
| `epoch` | texto | Período de topología (`reranker-v1`). Comparar entre epochs distintas no significa nada |
| `label` | texto | Nota libre (`--label`) |
| `git_commit` / `git_branch` / `git_dirty` | texto / 1-0 | Con qué código se corrió. Los inyectan los wrappers de `scripts/` |
| `llm_provider` / `llm_model` | texto | Proveedor y modelo **efectivos**, no el default de los valves |
| `judge_model` | texto | Sólo en `suite=judge` |
| `temperature` | float | 0 = determinístico |
| `embedding_model` / `ranker_model` | texto | Modelos de embeddings y de reranking |
| `retriever_top_k` / `ranker_top_k` | int | Cuántos trae el retriever / cuántos sobreviven al reranker |
| `chunking` | texto | Estrategia de chunking (hoy `production`) |
| `n_docs_store` | int | Documentos en el store al momento de correr |
| `dataset_n` | int | Preguntas corridas (menor al total si usaste `--limit`) |
| `n_errors` | int | Preguntas que fallaron |
| `duration_s` | float | Duración |
| `n_gt` | int | Preguntas **con ground truth** — el denominador de las métricas Tier 1 |
| `recall_eff` | 0–1 | De los documentos esperados, cuántos llegaron al LLM |
| `hit_eff` | 0–1 | En qué proporción de preguntas llegó **al menos uno** |
| `mrr_eff` | 0–1 | Qué tan **arriba** quedó el primer acierto. Recall alto con MRR bajo = lo encuentra pero lo rankea mal |
| `n_source`, `source_recall`, `source_hit`, `source_mrr` | | Lo mismo pero a nivel **fuente** en vez de chunk: agnóstico al chunking (ver `metrics.py`) |
| `n_sas` | int | Preguntas con `reference_answer` |
| `sas_mean` / `sas_std` | 0–1 | Similitud semántica media contra la referencia, y su dispersión |
| `n_judge` | int | Preguntas juzgadas (sólo `suite=judge`) |
| `faithfulness` | 0–1 | Proporción de afirmaciones respaldadas por el contexto. **Anti-alucinación** |
| `context_relevance` | 0–1 | Qué tan relevantes eran los documentos recuperados |

### `questions.csv` — una fila por corrida × pregunta

| Columna | Tipo | Significado |
|---|---|---|
| `run_id` | texto | A qué corrida pertenece. Se une con `runs.csv` por acá |
| `question_id` | texto | El `id` del `dataset.yaml` |
| `category` | texto | Categoría de la taxonomía |
| `status` | texto | `ok` o `error` |
| `error` | texto | La excepción, si `status=error` |
| `n_expected` | int | Cuántos documentos esperaba |
| `expected_ids` | lista `\|` | Ground truth |
| `retrieved_ids` | lista `\|` | Los que sobrevivieron al **reranker** (lo que vio el LLM) |
| `joined_ids` | lista `\|` | Los del `document_joiner`, **antes** del reranker. Comparar con el anterior muestra qué descartó el reranker |
| `rank_first_hit` | int | Posición (1-indexada) del primer acierto. Vacío = no acertó |
| `recall_eff` / `hit_eff` / `rr_eff` | 0–1 | Las métricas de esta pregunta sola |
| `via_embedding` / `via_keyword` | 1-0 | Qué retriever aportó el acierto. Los dos en 1 = ambos |
| `expected_sources` / `retrieved_sources` | lista `\|` | Ídem pero a nivel fuente |
| `source_recall` / `source_hit` / `source_rr` | 0–1 | Métricas Tier 1b de esta pregunta |
| `correct_rejection` | 1-0 | Sólo en **negativas** (sin ground truth): si el RAG se abstuvo como corresponde. ⚠️ Se detecta por **coincidencia de frases** ("no puedo responder", "no tengo suficiente información"), así que una abstención redactada distinto cuenta como fallo. Es el H6 de [mejoras_harness.md](../../../docs/eval/mejoras_harness.md) |
| `sas` | 0–1 | Similitud contra la `reference_answer` |
| `faithfulness` / `context_relevance` | 0–1 | Sólo en filas de `suite=judge` |
| `question` / `answer` / `reference_answer` | texto | La pregunta, lo que respondió, y la referencia. Des-escapar con `unescape_text()` |

> **Filas de `suite=judge`:** llenan `faithfulness` y `context_relevance` y dejan
> **vacías** las columnas de Tier 1 y 2. No es un olvido: ese runner no calcula recall
> ni SAS. Es la misma tabla, y cada suite llena lo que sabe medir.

## Interpretar la salida

Durante la corrida:

- `✓ / ✗` por pregunta con `recall`, `rr` y `via=` (qué retriever aportó el acierto).
  `–` = negativa (sin ground truth de retrieval).
- Resumen con SAS global y retrieval **por categoría**.

Al final, el bloque de comparación (`report.py compare`):

- **Advertencias** `⚠` primero: epochs distintas, `dataset_n` distinto, errores. Van
  arriba porque son las que invalidan todo lo de abajo.
- **Config**: qué cambió respecto del baseline (`retriever_top_k: 15 → 8`).
- **Global** y **por categoría** con `▲`/`▼`/`=`.
- **Por pregunta**: cuáles regresaron y cuáles mejoraron.

El color es siempre redundante (las flechas y el texto dicen lo mismo) y se apaga solo
cuando la salida no es una terminal, así que el log queda limpio.

## Archivos

| Archivo | Rol |
|---|---|
| `dataset.yaml` | preguntas + ground truth (versionado) |
| `eval_meta.yaml` | `baseline_run_id` + `epoch` vigente (versionado) |
| `csv_store.py` | **único** lugar que lee/escribe las tablas; esquemas y migración |
| `run_eval.py` | runner Tier 1 + 2 |
| `run_eval_llm.py` | runner Tier 3 (juez LLM), manual |
| `metrics.py` | recall@k, hit@k, MRR, coseno/SAS |
| `report.py` | análisis: `compare`, `runs`, `question` (y `html`, en curso) |
| `results/runs.csv` | historial de corridas (versionado) |
| `results/questions.csv` + `logs/` | detalle por pregunta y logs (gitignored) |
