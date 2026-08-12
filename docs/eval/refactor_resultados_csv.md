# Refactor: resultados del eval en CSV (plan de implementación)

> **Estado (2026-08-12): PLAN APROBADO — pendiente de implementación.** Este documento es
> una especificación paso a paso, pensada para ejecutarse sin tomar decisiones nuevas:
> toda ambigüedad se resolvió acá. Contexto de diseño en [eval_harness.md](eval_harness.md)
> y [mejoras_harness.md](mejoras_harness.md).

## Resumen del cambio

Hoy cada corrida de `run_eval.py` deja `<stamp>.json` + `<stamp>.log` + una fila en
`history.csv`, y el baseline es un snapshot copiado (`baseline.json`). Se reemplaza por:

- **Dos tablas CSV acumulativas** pensadas para leerse con pandas (no a ojo):
  `runs.csv` (1 fila por corrida) y `questions.csv` (1 fila por corrida × pregunta).
- **El baseline es un `run_id`**, declarado en `eval_meta.yaml`. Rehacer el baseline =
  editar una línea. Los deltas **no se persisten**: los calcula `report.py` (pandas)
  sobre los crudos.
- **`epoch`**: nombre declarado del período de topología del pipeline; se estampa en cada
  corrida y evita comparar baselines de topologías distintas (bug H2 de
  [mejoras_harness.md](mejoras_harness.md)).
- Desaparecen: `<stamp>.json`, `baseline.json`, `history.csv` (se migra), `tier3_*.json`,
  y toda la maquinaria de snapshot/promoción (`--set-baseline`, `report.promote`,
  `report.load_snapshot`).

Estructura final:

```
src/pipeline/eval/
  dataset.yaml               # (sin cambios)
  eval_meta.yaml             # NUEVO: baseline_run_id + epoch (versionado)
  csv_store.py               # NUEVO: esquemas de columnas + escritura/lectura CSV
  run_eval.py                # escribe crudo (runs/questions/log); no compara
  run_eval_llm.py            # ídem, con suite=judge
  metrics.py                 # (sin cambios funcionales)
  report.py                  # REESCRITO: análisis pandas (deltas, regresiones, evolución)
  results/
    runs.csv                 # versionado (excepción en .gitignore)
    questions.csv            # gitignored
    logs/<run_id>.log        # gitignored
    report/*.html            # gitignored (reportes generados, ver Paso 5.3)
scripts/
  eval.sh                    # NUEVO: wrapper host (inyecta metadata git)
  eval_llm.sh                # NUEVO: ídem para Tier 3
```

Decisiones ya tomadas (no reabrir):

- Esquema **mínimo**: solo columnas con datos reales hoy. Las métricas futuras
  (nDCG, abstención, tokens — ver mejoras_harness) se agregan cuando existan, vía la
  migración automática de esquema de `csv_store.py`.
- Tier 3 se migra con alcance **mínimo** (persistencia, no H7).
- `questions.csv` **no** se versiona; el delta por pregunta es local-only y `report.py`
  degrada con aviso si faltan las filas del baseline.
- El texto (pregunta/respuesta/referencia) va **dentro** de `questions.csv`, al final.
- `pipeline_signature` (hash automático de topología) queda **diferido** a H2; la guarda
  por ahora es la `epoch` declarada a mano.

---

## Paso 0 — Verificación previa

Dentro del container (`cd infrastructure && docker compose exec pipelines ...`):

1. `python -c "import pandas; print(pandas.__version__)"` — debe existir (viene con
   Haystack). Si no existiera: agregar `RUN pip install --no-cache-dir pandas` al
   Dockerfile de pipelines, rebuildear, y dejarlo anotado en el commit.
   `matplotlib` en cambio NO está y se agrega en el Paso 5.6 (solo lo usa el modo
   `html` del reporte).
2. Verificar escritura: `python -c "from pathlib import Path; Path('/app/pipelines/eval/results/logs').mkdir(parents=True, exist_ok=True)"`.

## Paso 1 — `csv_store.py`

Módulo nuevo en `src/pipeline/eval/`. Único lugar del harness que sabe leer/escribir CSV.
Sin dependencia de pandas para **escribir** (usa `csv` stdlib); pandas solo en las
funciones de lectura, importado lazy dentro de la función.

### 1.1 Esquemas (constantes módulo-level)

```python
RUNS_COLUMNS = [
    # claves e identidad
    "run_id", "timestamp_utc", "suite", "epoch", "label",
    "git_commit", "git_branch", "git_dirty",
    # config
    "llm_provider", "llm_model", "judge_model", "temperature",
    "embedding_model", "ranker_model", "retriever_top_k", "ranker_top_k",
    "n_docs_store", "dataset_n", "n_errors", "duration_s",
    # Tier 1 (calculadas sobre la salida del ranker → sufijo _eff, "efectivo")
    "n_gt", "recall_eff", "hit_eff", "mrr_eff",
    # Tier 2
    "n_sas", "sas_mean", "sas_std",
    # Tier 3 (suite=judge)
    "n_judge", "faithfulness", "context_relevance",
]

QUESTIONS_COLUMNS = [
    # claves
    "run_id", "question_id", "category", "status", "error",
    # Tier 1
    "n_expected", "expected_ids", "retrieved_ids", "rank_first_hit",
    "recall_eff", "hit_eff", "rr_eff", "via_embedding", "via_keyword",
    # Tier 2 / Tier 3
    "sas", "faithfulness", "context_relevance",
    # texto (siempre al final)
    "question", "answer", "reference_answer",
]

# Renombres históricos de columnas: la migración de esquema los consulta antes de
# descartar datos. {nombre_viejo: nombre_nuevo}. Arranca vacío.
RENAMES: dict[str, str] = {}
```

Regla de orden: claves → config/dimensiones → métricas → texto. Columnas nuevas se
insertan en su grupo, nunca "al final de todo" (salvo texto).

### 1.2 Serialización (`_ser(value) -> str`)

| Valor Python | En el CSV |
|---|---|
| `None` | celda vacía `""` |
| `True` / `False` | `1` / `0` |
| `list` (de strings) | join con `\|` → `cwe-79\|cwe-80` |
| `float` | `str(round(v, 4))` |
| `int` | `str(v)` |
| `str` | reemplazar `\r\n`→`\n` y luego `\n` (newline real) → `\n` **literal** (backslash + n) |

Con esto **cada fila del CSV es una línea física** (`grep`/`wc -l` funcionan). El quoting
de comas/comillas lo maneja `csv.writer` (QUOTE_MINIMAL, default). Encoding UTF-8,
newline `\n` (pasar `newline=""` al abrir, como ya hace el código actual).

La inversa vive acá también: `unescape_text(s: str) -> str` (reemplaza el `\n` literal
por newline real). Es lo que usa `report.py` cada vez que muestra texto completo
(§5.2 `question`, §5.3 tablas HTML); nadie des-escapa a mano.

Notas asumidas y documentadas en el docstring del módulo:
- Si un texto contiene la secuencia literal `\n` (backslash-n tipeado), es
  indistinguible de un newline escapado. Aceptado.
- **Sin locking**: se asume un solo runner a la vez (uso manual del harness;
  `run_eval.py` y `run_eval_llm.py` no se corren en simultáneo). No implementar locks.

### 1.3 `append_row(path: Path, row: dict, columns: list[str]) -> None`

1. Serializar `row` con `_ser` (claves ausentes en `row` → vacía; claves de `row` que no
   están en `columns` → `ValueError`, para atrapar typos).
2. Si `path` no existe: crear directorio padre, escribir header + fila. Fin.
3. Leer la primera línea del archivo (header existente, parseado con `csv.reader`).
4. Si header existente == `columns`: abrir en modo append y escribir la fila. Fin.
5. Si difieren → **migración con reescritura atómica**:
   a. Leer todo el archivo con `csv.DictReader`.
   b. Para cada columna vieja que no está en `columns`: si está en `RENAMES`, mapear sus
      valores al nombre nuevo; si no, descartarla e imprimir
      `⚠ csv_store: columna '<x>' eliminada del esquema; datos descartados en <path>`.
   c. Escribir `<path>.tmp` con el header nuevo y todas las filas viejas (celdas sin dato
      → vacías), más la fila nueva.
   d. `os.replace(tmp, path)` (rename atómico: un crash a mitad de escritura nunca
      corrompe el archivo original).

### 1.4 Lectura (para `report.py`)

```python
def load_runs(path) -> "pd.DataFrame": ...
def load_questions(path) -> "pd.DataFrame": ...
```

Ambas: `pd.read_csv(path, dtype=DTYPES_X, keep_default_na=True)` con dicts de dtype
explícitos definidos en este módulo — **nadie más hace `pd.read_csv` a mano**:

- Columnas de texto/ids/config-string → `"string"`.
- Métricas, contadores y flags 1/0 → `"float64"` (los vacíos quedan `NaN`; los flags se
  leen como float, es intencional para poder hacer `.mean()`).
- No parsear `timestamp_utc` como fecha en la lectura base (quien lo necesite hace
  `pd.to_datetime` aparte).

Si el archivo no existe, devolver un DataFrame vacío con las columnas del esquema (evita
`FileNotFoundError` en report).

## Paso 2 — `eval_meta.yaml`

Archivo nuevo en `src/pipeline/eval/`, **versionado**:

```yaml
schema_version: 1

# Baseline = run_id de una fila de results/runs.csv. Para re-basear, editar esta línea
# (o correr scripts/eval.sh --set-baseline) y commitear.
baseline_run_id: null   # se fija en el Paso 10 con la corrida de validación

# Epoch vigente: nombre del período de topología actual. Se estampa en cada corrida.
# Cambiaste la topología del pipeline (reranker, chunking, modelo de embeddings,
# componentes)? → declarar acá un nombre nuevo y re-basear.
epoch: reranker-v1

# Historial documental (el código no lo lee; es memoria del proyecto).
epochs:
  - name: pre-reranker
    nota: "híbrido RRF sin cross-encoder"
  - name: reranker-v1
    nota: "bge-reranker-v2-m3; retriever_top_k=15, ranker_top_k=4"
```

Helper (puede vivir en `csv_store.py`): `load_meta(path) -> dict` con
`yaml.safe_load`; si falta el archivo o una clave → error claro, no default silencioso.

## Paso 3 — `run_eval.py`

Principio: **escribe crudo, no compara**. Cambios sobre el archivo actual:

1. **`run_id`**: `datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")`. Antes de usarlo,
   si ya existe en la columna `run_id` de `runs.csv`, sufijarlo con `-2` (`-3`, …).
2. **Log**: `results/logs/<run_id>.log` (subdirectorio nuevo). El mecanismo `_Tee` se
   mantiene igual.
3. **Metadata**: leer `eval_meta.yaml` (para `epoch`) y las env vars `GIT_COMMIT`,
   `GIT_BRANCH`, `GIT_DIRTY` (vacías si no están — **nunca** fallar ni intentar correr
   `git` adentro del container: no hay `.git` montado).
4. **Flags**: agregar `--label "texto"` (default vacío). **Eliminar** `--baseline`.
   Conservar `--dataset`, `--out`, `--top-k`, `--temperature`, `--limit`.
   `--set-baseline` **se conserva pero cambia de semántica**: ya no promueve un
   snapshot JSON; al final de una corrida exitosa reescribe la línea
   `baseline_run_id: <run_id>` en `eval_meta.yaml`. Mecanismo obligatorio:
   **reemplazo de esa línea por regex** (`^baseline_run_id:.*$` → valor nuevo, sobre el
   texto crudo del archivo) — **no** usar `yaml.safe_dump` para reescribir el archivo,
   porque PyYAML descarta los comentarios (y este yaml es casi todo comentarios y el
   historial de epochs). Si la regex no matchea ninguna línea, error claro y no tocar
   el archivo.
5. **Por pregunta**: tras evaluar cada una, construir la fila y
   `csv_store.append_row(questions_csv, fila, QUESTIONS_COLUMNS)` **inmediatamente**
   (si la corrida muere en la pregunta 20, las 19 anteriores quedan persistidas).
   Mapeo desde el `rec` actual de `evaluate()`:
   - `question_id` ← `rec["id"]`; `expected_ids` ← `rec["expected_doc_ids"]`;
     `retrieved_ids` ← `rec["retrieved_ids"]` (los del ranker, como hoy).
   - `recall_eff`/`hit_eff`/`rr_eff` ← `recall`/`hit`/`rr` (renombre honesto: se calculan
     sobre la salida del reranker; ver H1 en mejoras_harness).
   - `rank_first_hit` ← posición 1-indexada del primer esperado en `retrieved_ids`
     (vacío si no hubo hit o no hay ground truth). Equivale a `round(1/rr)` pero
     calcularlo directo, no dividir.
   - `via_embedding`/`via_keyword` ← `rec["hit_source"]["embedding"|"keyword"]`
     (vacías si `hit_source is None`).
   - `faithfulness`/`context_relevance` ← vacías (las llena Tier 3).
   - `status`/`error` como hoy (`ok`/`error`).
6. **Al final**: agregados con `metrics.aggregate_retrieval` y `aggregate_sas` (sin
   cambios) → una fila a `runs.csv` con `suite="retrieval"`. Campos:
   - `duration_s`: `time.perf_counter()` alrededor del loop completo, redondeado a 1
     decimal. `sas_std`: `statistics.stdev` de los SAS individuales si `n_sas >= 2`,
     si no vacío. `n_errors`: cantidad de `status == "error"`.
   - `llm_model` **efectivo**: replicar la resolución de `run_eval_llm.py` actual
     (líneas 103-108): con provider `ollama` es `os.getenv("LLM_MODEL") or
     rag.DEFAULT_OLLAMA_LLM`; con `groq` es `os.getenv("LLM_MODEL") or valves.llm_model`.
     Extraer eso a una función compartida `effective_llm_model(valves)` (puede vivir en
     `csv_store.py` o en un helper común del eval) y usarla en ambos runners.
   - `judge_model`, `n_judge`, `faithfulness`, `context_relevance` → vacías.
7. **`by_category` deja de persistirse** (se recalcula con groupby en report).
8. **Cortesía**: tras persistir, invocar el reporte (`META_PATH = EVAL_DIR /
   "eval_meta.yaml"`, constante módulo-level junto a las demás rutas):
   ```python
   META_PATH = EVAL_DIR / "eval_meta.yaml"
   try:
       report.compare(run_id=run_id, results_dir=args.out, meta_path=META_PATH)
   except Exception as e:
       print(f"⚠ reporte falló ({e!r}); los datos quedaron persistidos igual")
   ```
   Los datos se escriben **antes** de cualquier análisis, siempre.
9. La impresión por pregunta y el resumen de la corrida actual (`_print_row`,
   `_print_summary`) se conservan tal cual.

## Paso 4 — Migrar `history.csv` y limpiar `results/` (one-off, manual)

No es código del repo: son acciones que el implementador ejecuta una vez.

1. Para cada una de las 3 filas de `results/history.csv`, appendear (con
   `csv_store.append_row`, p. ej. desde un script descartable en el scratchpad o un
   heredoc de python) una fila a `runs.csv` con este mapeo:
   `timestamp→timestamp_utc` (y derivar `run_id` reformateando:
   `2026-07-03T21:27:11.078335+00:00` → `20260703T212711Z`), `top_k→retriever_top_k`,
   `n_retrieval→n_gt`, `recall_at_k→recall_eff`, `hit_rate→hit_eff`, `mrr→mrr_eff`,
   `sas→sas_mean`, `n_sas→n_sas`, `llm_model`/`temperature`/`embedding_model` directo,
   `suite=retrieval`, `label=migrada-de-history`, resto vacío.
2. `epoch` de cada fila migrada: la de 2026-07-03 es `pre-reranker`. Para las dos de
   2026-07-10: `grep -l ranker results/20260710T*.log` — si el log menciona la etapa
   `ranker`, son `reranker-v1` (y además fijar `ranker_top_k=4`); si no, `pre-reranker`.
   Si el grep no es concluyente, poner `epoch=legacy` y seguir (son 3 filas; no hacer
   arqueología); en ese caso, declarar también `legacy` en la lista `epochs` de
   `eval_meta.yaml` (`nota: "corridas pre-migración de topología incierta"`).
   Las corridas migradas **no tienen filas en `questions.csv`** (el detalle por
   pregunta de esas corridas se pierde; aceptado): `report.py` las muestra en `runs` y
   en la evolución, pero `question <qid>` simplemente no listará esos run_ids y
   compararlas con `compare` degrada a delta global (§5.2.9) — comportamiento
   esperado, no bug.
3. Borrar `history.csv`. Mover los `*.log` sueltos a `results/logs/`. Borrar los
   `<stamp>.json` sueltos (su detalle por pregunta se pierde; aceptado — el resumen ya
   está migrado).
4. `git rm src/pipeline/eval/baseline.json`.

## Paso 5 — `report.py` (reescritura completa)

Script de análisis con pandas + reporte HTML con gráficos. Sin estado propio: lee
`runs.csv` + `questions.csv` + `eval_meta.yaml` vía `csv_store.load_*`/`load_meta`.
Nunca escribe en las tablas; sus únicos outputs son stdout y archivos bajo
`results/report/`. Corre dentro del container siempre; en el host solo si hay
pandas+pyyaml (no es requisito).

**Se elimina** del report viejo: `load_snapshot`, `promote`, `print_delta` sobre dicts
de snapshot — el módulo se reescribe entero; no queda código que lea JSON.

**Orden interno obligatorio**: primero los modos terminal (§5.1–5.2) hasta dejarlos
validados — son el gate diario y no dependen de matplotlib —; el modo `html`
(§5.3–5.4) se implementa después y su eventual retraso no bloquea el resto del plan.

### 5.0 Investigación previa (resumen de lo verificado)

- **pandas** está en el container (dependencia de `haystack-ai`); es lo único que
  requieren los modos de terminal.
- **matplotlib NO está** en la imagen → solo lo necesita el modo `html`; se agrega una
  línea al Dockerfile (§5.6) y el import es lazy (los modos terminal jamás lo tocan).
- Los gráficos siguen el método de dataviz por roles: forma según el trabajo del dato,
  color por rol, paleta **validada por script** (no a ojo). La paleta usada acá pasó el
  validador (2026-08-12, modo light, superficie `#fcfcfb`):
  `#2a78d6,#eb6834,#1baf7a,#eda100` → ALL CHECKS PASS, con un WARN de contraste en
  `#1baf7a` (2.74:1) y `#eda100` (2.11:1) que **obliga** a labels directos en las
  líneas y a la tabla de datos junto a cada gráfico. Esas mitigaciones son parte del
  spec, no opcionales.
- El HTML es un **documento** autocontenido: un solo archivo, cero requests externas,
  sin JS; SVGs de matplotlib embebidos inline + tablas. Comprometido deliberadamente a
  **modo claro único** con fondos explícitos.

Principios: (1) solo lee crudos; (2) degradación en cascada, nunca crash — sin baseline
→ resumen solo; sin filas del baseline en `questions.csv` → delta global con aviso; sin
matplotlib → terminal funciona y `html` falla con mensaje accionable; (3) el color
nunca es el único canal (`▲▼=` y texto siempre acompañan); (4) exit 0 siempre, salvo
`--strict`.

### 5.1 CLI

`argparse` con subcomandos. Sin subcomando ⇒ `compare` con defaults.

```bash
python report.py                          # compare: última corrida vs baseline
python report.py compare --run A --baseline B
python report.py runs [--all]             # tabla de corridas + sparklines
python report.py question <question_id>   # evolución de una pregunta
python report.py html [--run A --baseline B] [-o PATH]
```

Flags globales (parser padre): `--results` (default `EVAL_DIR/"results"`), `--meta`
(default `EVAL_DIR/"eval_meta.yaml"`), `--no-color`, `--strict`.

Resoluciones compartidas:

- **"última corrida"** = máximo `run_id` (orden lexicográfico = cronológico) entre
  filas con `suite == "retrieval"`.
- **baseline default** = `baseline_run_id` del meta; `null`/ausente → aviso
  `sin baseline configurado (eval_meta.yaml)` + resumen de la corrida sola, exit 0.
- `--run`/`--baseline` con `run_id` inexistente → `run_id '<x>' no existe; corridas
  disponibles: <últimas 5>`, exit 2.
- Función importable `compare(run_id=None, baseline_run_id=None, results_dir=…,
  meta_path=…) -> int` (retorna el nº de regresiones; es lo que invoca `run_eval.py`).
  `main()` solo parsea y delega.
- **`--strict`**: exit 1 si hubo regresiones (cualquier caída de `recall_eff` por
  pregunta, o caída de `sas` > `SAS_REGRESSION_THRESHOLD`); 0 si no. Para cablear a un
  hook/CI futuro. Sin `--strict`, siempre 0 (informa, no bloquea).

### 5.2 Capa terminal

Reglas de formato:

- Números: 3 decimales (`0.841`), enteros sin decimales, faltantes `–` (`na_rep="–"`,
  `float_format=lambda v: f"{v:.3f}"`).
- Tablas: `DataFrame.to_string(index=False)` sobre un DF ya ordenado y renombrado para
  display. Nunca `to_markdown` (dependería de `tabulate`).
- Delta: `0.841 ▲0.012` / `▼` / `=` (reusar la semántica de `_arrow`/`_fmt` del report
  viejo).
- ANSI: helper único `paint(s, role)` con roles `good` (`\033[32m`), `bad` (`\033[31m`),
  `warn` (`\033[33m`), `dim` (`\033[2m`), `bold`. Devuelve `s` sin códigos si:
  `--no-color`, o env `NO_COLOR`, o `not sys.stdout.isatty()` — esto último deja limpio
  el log tee-ado de `run_eval.py` sin código extra. El color refuerza lo que `▲▼` ya
  marca, nunca informa solo.
- Encabezados de bloque: `── TÍTULO ─────…` a 64 columnas (estilo ya usado en el eval).

**`compare`** — bloques en orden:

1. **Identificación**: `run <id> ('<label>', epoch <e>) vs baseline <id> (…)`.
2. **Advertencias de honestidad** (`⚠` + `paint(..., "warn")`):
   - epochs distintas → `topologías diferentes; el delta NO es comparable. Re-baseá.`
   - `dataset_n` distinto → `distinto nº de preguntas (X vs Y); promedios no comparables.`
   - `n_errors > 0` en cualquiera de las dos → mencionarlo.
3. **Config diff**: columnas del grupo config de `runs.csv` con valores distintos,
   formato `retriever_top_k: 15 → 8`; si no hay: `config idéntica`.
4. **Global**: `recall_eff`, `hit_eff`, `mrr_eff`, `sas_mean` con delta pintado.
5. **Por categoría**: DF `category, n, recall_base, recall_cur, Δrecall, sas_base,
   sas_cur, Δsas`, ordenado por `Δrecall` ascendente (lo peor arriba). Fuente:
   `questions.csv` filtrado a ambos run_ids + `groupby("category")`.
6. **Regresiones por pregunta** (vía `pivot_table(index="question_id",
   columns="run_id")`):
   - retrieval: `recall_eff` actual < baseline (cualquier caída) → `qid  0.50 → 0.00`.
   - generación: `sas` cayó > `SAS_REGRESSION_THRESHOLD = 0.05` (constante acá; se
     recalibra en H5).
   - preguntas nuevas/desaparecidas vs baseline → nota en `dim`.
   - si no hay: `Sin regresiones por pregunta. 🎉`
7. **Mejoras por pregunta** (simétrico, compacto, `dim`): las subas también validan.
8. Pie (`dim`): `Detalle HTML: python report.py html`.
9. **Degradación**: sin filas del baseline en `questions.csv` local → `⚠ sin detalle
   por pregunta del baseline en esta máquina; solo delta global (runs.csv)` y saltear
   5–7.

**`runs`** — historial: tabla de `runs.csv` (default últimas 15 con `suite=retrieval`;
`--all` = todas + judge), columnas `run_id, epoch, label, dataset_n, recall_eff,
mrr_eff, sas_mean, n_errors`; la fila baseline marcada con `*`. Debajo, **sparklines**
unicode (orden cronológico, solo corridas completas de la epoch vigente):

```
  recall_eff  ▄▄▅▆▆█  0.841
  sas_mean    ▃▄▄▅▆▆  0.772
```

`spark(vals)`: índice `round(v*7)` sobre `"▁▂▃▄▅▆▇█"` (escala fija 0–1, no min-max
relativo: comparable entre métricas), NaN → `·`; al final el último valor numérico.

**`question <qid>`** — drill-down: (1) ficha: categoría, `expected_ids`, pregunta
completa des-escapada con `csv_store.unescape_text`; (2) tabla cronológica sobre todas las corridas: `run_id, epoch,
label, recall_eff, rank_first_hit, via (emb/kw/emb+kw/-), sas, faithfulness`;
(3) última respuesta y la del baseline si difieren, completas, encabezado `dim`;
(4) qid inexistente → sugerencias con `difflib.get_close_matches` y exit 2.

### 5.3 Reporte HTML (`html`)

Salida: `results/report/report_<run>_vs_<baseline>.html` (cubierto por el gitignore de
`results/*`), `-o` para elegir path. Al final imprimir el path del lado host
(`src/pipeline/eval/results/report/...`).

Tokens visuales (CSS literal en `<style>`; modo claro único, fondos explícitos;
tipografía `system-ui`; números de tabla con `font-variant-numeric: tabular-nums`):

```css
:root {
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
  --border:rgba(11,11,11,.10);
  --good:#006300; --bad:#d03b3b;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;  /* series */
  --seq-100:#cde2fb; --seq-250:#86b6ef; --seq-400:#3987e5; --seq-550:#1c5cab; --seq-700:#0d366b;
  --div-neutral:#f0efec;
}
body { background:var(--page); color:var(--ink);
       font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
       max-width:1080px; margin:0 auto; padding:32px 24px; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:8px;
        padding:20px 24px; margin:16px 0; }
```

Estructura del documento:

1. **Header**: título `Eval RAG — <run_id>`, subtítulo con label, epoch, fecha,
   `git_commit`, config resumida y baseline comparado.
2. **Banda de advertencias** (si aplica): las del compare, como card con borde `--bad`.
3. **KPI row** — 4 stat tiles (no un bar chart de una barra): `recall_eff`, `mrr_eff`,
   `hit_eff`, `sas_mean`; valor grande (32px), delta debajo (`▲` `--good` / `▼` `--bad`
   / `=` `--muted`), nombre de métrica en `--ink-2`.
4. **Gráfico 1 — evolución** + tabla de runs plegada en `<details>`.
5. **Gráfico 2 — categorías baseline → actual** + la tabla por categoría del compare.
6. **Gráfico 3 — matriz pregunta × corrida**.
7. **Tablas finales**: regresiones y mejoras; config diff completa; `<details>` con la
   tabla por pregunta de la corrida actual (`question_id, category, recall_eff,
   rank_first_hit, via, sas, answer` truncada a 160 chars).
8. **Footer** (`--muted`): `generado por report.py · <timestamp> · datos: runs.csv
   (N corridas), questions.csv (M filas)`.

Tablas HTML: `df.to_html(index=False, classes="tbl", na_rep="–", float_format=…)` +
CSS propio para `.tbl` (bordes hairline `--grid`, header `--ink-2`, sin zebra, números
a la derecha). En regresiones, la celda caída lleva `▼` y `--bad` (post-proceso simple
del HTML o `Styler.map`, lo que menos código requiera).

### 5.4 Gráficos — especificación exacta

Generación: matplotlib backend `Agg`, export SVG a `io.StringIO`, embebido inline
(remover `<?xml …?>` y DOCTYPE; el `<svg>` va directo, `width:100%; height:auto`).
Cada gráfico es una función `fig_*(...) -> str` (SVG). rcParams comunes (literal):

```python
matplotlib.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "font.family": "sans-serif", "font.size": 11,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.75,
    "axes.axisbelow": True,
    "xtick.color": "#898781", "ytick.color": "#898781",
    "axes.labelcolor": "#52514e", "text.color": "#0b0b0b",
    "svg.fonttype": "none",   # texto como <text>: legible y liviano
})
```

**Gráfico 1 — Evolución por corrida** (*trend con series distintas → multi-línea
categórica*):

- Datos: filas `suite=retrieval` completas (`dataset_n` == moda) en orden cronológico;
  x ordinal (ticks `MMDD·HHMM` rotados 45°), y fijo 0–1.
- 4 series: `recall_eff` `--s1`, `mrr_eff` `--s2`, `sas_mean` `--s3`, `hit_eff` `--s4`
  (paleta validada, §5.0). Línea `linewidth=2`, marcador `o` `markersize=6`, sin área.
- **Obligatorio** (relief del WARN de contraste): label directo al final de cada línea
  (nombre de la métrica en su color, `fontsize=10`, desplazado para no solaparse)
  **además** de la leyenda (`loc="lower left"`, `frameon=False`) — con 4 series ambas
  son mandatorias.
- Cambios de epoch: vertical punteada `--axis` entre corridas donde cambia `epoch`,
  con el nombre nuevo en `--muted` `fontsize=9`.
- Baseline marcado con anillo (marcador extra `markerfacecolor="none",
  markeredgecolor="#0b0b0b", markersize=11`).
- Con < 2 corridas completas: omitir el gráfico, nota `aún no hay historia suficiente`.

**Gráfico 2 — Categorías baseline → actual** (*before→after por ítem → dumbbell
horizontal*):

- Una fila por categoría, ordenadas por `recall_cur` ascendente (lo peor arriba).
  Punto baseline `--seq-250`, punto actual `--seq-550` (un tono, dos pasos), conector
  `--grid` 1.5pt, `markersize=8`. x = recall 0–1.
- Label selectivo: solo el valor actual, a la derecha del punto (`fontsize=10`,
  `--ink-2`). Leyenda de dos entradas. Categoría presente en un solo lado → punto solo,
  sin conector.
- Nota bajo el título: `n por categoría: 1–3 — un cambio de una pregunta mueve la barra
  entera` (honestidad estadística; cita H3 de mejoras_harness).

**Gráfico 3 — Matriz pregunta × corrida** (*magnitud en grilla → heatmap secuencial*):

- `pivot_table(index="question_id", columns="run_id", values="recall_eff")` sobre las
  últimas ≤ 12 corridas completas; filas ordenadas por categoría y luego id; línea
  separadora `--axis` 1pt entre categorías, label de categoría a la izquierda en
  `--muted`.
- Color: un solo tono, más = más oscuro. `BoundaryNorm` con bins `[0,.25,.5,.75,1.0]`
  sobre `ListedColormap(["#cde2fb","#86b6ef","#3987e5","#1c5cab"])`; exactamente 1.0 →
  `#0d366b`. Celda NaN (negativa / no corrida): `--div-neutral` con `–` en `--muted`.
- Valor anotado en cada celda `fontsize=8`; tinta blanca sobre los dos pasos oscuros,
  `--ink` sobre los claros (umbral por paso, hardcodeado).
- Gap entre celdas: `linewidth=2, edgecolor="#fcfcfb"` (spacer del método).
- Leyenda: barra discreta horizontal abajo (5 pasos con sus rangos), no colorbar
  continua.

Anti-patrones vigilados: **nunca** doble eje y; no pie charts; no rainbow; no anotar
todos los puntos (solo el label final de línea); una métrica no cambia de color entre
gráficos (recall es siempre `--s1` donde aparezca).

### 5.5 Estructura interna y pandas

```
report.py
  # datos
  load_context(results_dir, meta_path) -> Ctx    # runs, questions, meta
  resolve_pair(ctx, run, baseline) -> (row_cur, row_base | None)
  # análisis (puros: DataFrame in → DataFrame out, sin prints — importables/testeables)
  category_table(q, run_a, run_b) -> DataFrame
  question_pivot(q, metric, run_ids) -> DataFrame
  regressions(q, run_a, run_b) -> dict[str, DataFrame]   # {"retrieval": …, "sas": …}
  config_diff(row_a, row_b) -> list[tuple]
  # terminal
  paint(s, role), spark(vals)
  compare(run_id=None, baseline_run_id=None, results_dir=…, meta_path=…) -> int
      # ÚNICA implementación del modo compare: imprime los bloques de §5.2 y retorna
      # el nº de regresiones. La llaman tanto main() (subcomando compare) como
      # run_eval.py (cortesía). No existe un cmd_compare separado.
  cmd_runs(args), cmd_question(args)      # imprimen; usan las funciones de análisis
  # html (matplotlib import lazy acá adentro)
  fig_evolution(...), fig_categories(...), fig_matrix(...)   # -> str (SVG)
  render_html(ctx, ...) -> str;  cmd_html(args)
  # entrada
  main()   # argparse; despacha a compare/cmd_*
SAS_REGRESSION_THRESHOLD = 0.05
```

Idioms pandas fijados (no reinventar): filtros con `q[q.run_id.isin([a,b])]`; por
categoría `groupby("category").agg(n=("question_id","size"),
recall=("recall_eff","mean"), sas=("sas","mean"))` por lado + `join` con sufijos
`_base`/`_cur`; evolución/regresiones vía `pivot_table` y comparación de columnas (no
loops); `via` con `np.select` sobre `via_embedding`/`via_keyword` → `emb`/`kw`/
`emb+kw`/`-`; los flags leídos como float se comparan con `== 1.0`; orden cronológico =
`sort_values("run_id")`.

### 5.6 Dependencia nueva: matplotlib

En `infrastructure/Dockerfile.pipelines`, antes del bloque "Verificaciones":

```dockerfile
# Gráficos del reporte de evaluación (report.py html). Sin backend GUI (Agg).
RUN pip install --no-cache-dir matplotlib
```

y una verificación más: `RUN python -c "import matplotlib; print('matplotlib OK')"`.
Requiere `docker compose build pipelines` (layer apendeada; no invalida las capas
pesadas de torch/marker). En `report.py` el import vive dentro de `fig_*`/`cmd_html`;
si falla: `el modo html requiere matplotlib en la imagen — rebuildeá con docker compose
build pipelines` y exit 2. Los demás modos no lo tocan.

### 5.7 Validación del reporte (se suma al Paso 10)

1. `report.py runs` con las corridas migradas + 2 nuevas → tabla y sparklines sin
   excepciones; baseline con `*`.
2. `report.py` con y sin TTY (`| cat`) → con pipe, ningún código ANSI en el output.
3. `--strict` tras una corrida con `--top-k 1` (regresión garantizada) → exit 1; sin
   regresiones → exit 0.
4. `report.py question <qid>` y un id con typo → sugerencias y exit 2.
5. `report.py html` → **abrir el archivo y mirarlo** (el validador chequea color, no
   layout): sin colisiones de labels en la evolución, dumbbell legible, matriz con
   separadores y celdas anotadas legibles, KPI row con deltas correctos. Verificar que
   no hace ninguna request externa (abrirlo sin red) y que pesa < 1 MB.
6. Bordes: una sola corrida (compare degrada a resumen), baseline `null`,
   `questions.csv` sin filas del baseline (delta global + aviso), corrida `--limit 3`
   presente (excluida de evolución/matriz por `dataset_n` distinto, con nota).

Fuera de alcance del reporte: dark mode del HTML (documento comprometido a claro),
tooltips/JS (documento estático; cada gráfico lleva su tabla), export PNG, comparación
entre corridas `suite=judge` (llega con H7). El diseño los admite después sin
reestructurar (análisis y `fig_*` ya separados).

## Paso 6 — `run_eval_llm.py` (Tier 3, migración mínima)

1. `run_id`, log en `results/logs/`, `--label`, metadata git/epoch: igual que el Paso 3
   (puntos 1-3). Nota: hoy este script no tiene `_Tee`; agregarlo igual que en
   `run_eval.py` para que también deje log.
2. Por pregunta → fila en `questions.csv`: `run_id`, `question_id`, `category` (tomarla
   del item del dataset; hoy el script no la guarda), `status="ok"`, `faithfulness`,
   `context_relevance`, `question`, `answer`. Resto de columnas vacías (no hay ids
   de retrieval en este runner: usa `document_joiner` para contexts, no ids — no
   inventar métricas Tier 1 acá).
3. Al final → fila en `runs.csv` con `suite="judge"`: config conocida (`llm_provider`,
   `judge_model`, `temperature=0`, `embedding_model`, `retriever_top_k`, `ranker_top_k`,
   `ranker_model`, `dataset_n`, `duration_s`, `epoch`, git), `n_judge` = cantidad de
   preguntas con score no-None, `faithfulness` = `fres.get("score")`,
   `context_relevance` = `cres.get("score")`. Columnas Tier 1/2 vacías.
4. Eliminar la escritura de `tier3_<stamp>.json`.
5. `report.py` ignora las filas `suite=judge` al elegir "última corrida"; la comparación
   entre corridas judge queda para H7 (fuera de alcance).

## Paso 7 — Wrappers `scripts/eval.sh` y `scripts/eval_llm.sh`

Crear el directorio `scripts/` en la raíz del repo. `eval.sh` (ejecutable, `chmod +x`):

```bash
#!/usr/bin/env bash
# Corre el eval harness dentro del container `pipelines`, inyectando metadata de git
# (el container no ve .git). Uso: scripts/eval.sh [args de run_eval.py]
set -euo pipefail
cd "$(dirname "$0")/.."
GIT_COMMIT="$(git rev-parse --short HEAD)"
GIT_BRANCH="$(git branch --show-current)"
GIT_DIRTY="$([ -n "$(git status --porcelain)" ] && echo 1 || echo 0)"
cd infrastructure
docker compose exec \
  -e GIT_COMMIT="$GIT_COMMIT" -e GIT_BRANCH="$GIT_BRANCH" -e GIT_DIRTY="$GIT_DIRTY" \
  pipelines python /app/pipelines/eval/run_eval.py "$@"
```

(El `cd infrastructure` antes del `docker compose` es deliberado: es la forma ya
documentada en el README del eval y no depende del comportamiento de
`--project-directory`. Las vars de git se capturan **antes** de cambiar de directorio.)

`eval_llm.sh`: idéntico apuntando a `run_eval_llm.py`.

## Paso 8 — Git: ignore y merge

En `.gitignore`, reemplazar la línea `src/pipeline/eval/results/` por:

```gitignore
src/pipeline/eval/results/*
!src/pipeline/eval/results/runs.csv
```

(La negación **requiere** ignorar el contenido con `*`; ignorar el directorio a secas
haría la excepción inefectiva.)

Crear `.gitattributes` en la raíz (o agregar si existiera):

```gitattributes
# runs.csv es append-only: dos branches que agregan filas mergean por unión.
src/pipeline/eval/results/runs.csv merge=union
```

Limitación conocida de `merge=union`: si el header cambió en una sola branch puede
duplicarse una línea; el síntoma es ruidoso (pandas falla) y se arregla a mano. Aceptado.

## Paso 9 — Documentación

1. `src/pipeline/eval/README.md`: reescribir las secciones **Uso**, **Flujo de
   trabajo**, **Persistencia de resultados** y **Archivos** al esquema nuevo. Agregar una
   sección **Diccionario de columnas** con una tabla por CSV (columna, tipo, significado
   — en particular explicar el sufijo `_eff`: métricas sobre la salida del reranker, lo
   que ve el LLM). Documentar las reglas de formato (vacío = NaN, `1`/`0`, listas con
   `|`, `\n` literal) y los snippets pandas de arranque:
   ```python
   import pandas as pd
   runs = pd.read_csv("results/runs.csv")
   q    = pd.read_csv("results/questions.csv")
   df   = q.merge(runs[["run_id", "epoch", "label"]], on="run_id")
   ```
2. `docs/eval/eval_harness.md`: actualizar "Baseline y delta", "Estructura de archivos
   propuesta" y "Comandos" (ahora `scripts/eval.sh`); anotar en el encabezado de estado
   que la persistencia migró a CSV con link a este doc.
3. `docs/eval/mejoras_harness.md`: anotar en H1 (punto 6) y H2 que la parte de
   persistencia/rotación de history y la promoción de baseline quedaron resueltas por
   este refactor (la parte de métricas dual-k sigue pendiente).

## Paso 10 — Validación y baseline inicial

Requiere el stack levantado (`cd infrastructure && docker compose up -d`) y el store
poblado.

1. `scripts/eval.sh --label "primera corrida CSV"` → verificar: fila nueva en `runs.csv`,
   27 filas nuevas en `questions.csv`, log en `results/logs/`, y que
   `csv_store.load_runs()/load_questions()` cargan sin warnings de dtype.
2. Segunda corrida idéntica → `report.py --run <run2> --baseline <run1>` debe dar deltas
   ≈ 0 (retrieval exactamente 0; SAS puede moverse ~±0.01 — anotar el valor observado,
   sirve para H5).
3. `scripts/eval.sh --top-k 5 --label "prueba top_k=5"` → el reporte debe mostrar el
   config diff (`retriever_top_k: 15 → 5`) y deltas reales.
4. `scripts/eval.sh --limit 3` → verificar el warning de `dataset_n` distinto.
5. Probar la migración de esquema: agregar una columna dummy a `QUESTIONS_COLUMNS`,
   correr con `--limit 1`, verificar que el archivo se reescribió con la columna nueva y
   las filas viejas intactas (vacías en la dummy); revertir la columna y verificar el
   warning de columna descartada.
6. `scripts/eval_llm.sh --limit 5` → fila `suite=judge` en `runs.csv` + scores por
   pregunta en `questions.csv`.
7. **Fijar el baseline**: elegir la corrida completa del punto 1 (o repetir una limpia si
   hubo cambios), correr `scripts/eval.sh --set-baseline` o editar
   `baseline_run_id` en `eval_meta.yaml` a mano. Commitear `eval_meta.yaml` + `runs.csv`.

## Criterios de terminado

- [ ] `run_eval.py` y `run_eval_llm.py` no escriben ningún JSON.
- [ ] `results/` contiene solo `runs.csv`, `questions.csv` y `logs/`.
- [ ] `baseline.json` y `history.csv` eliminados; las 3 corridas históricas viven en `runs.csv`.
- [ ] `report.py` funciona standalone (sin correr el eval) y comparando cualquier par de corridas.
- [ ] `pd.read_csv` de ambas tablas sin warnings; tipos estables por columna.
- [ ] `eval_meta.yaml` con `baseline_run_id` real, commiteado junto con `runs.csv`.
- [ ] README y docs actualizados (incluido el diccionario de columnas).

## Fuera de alcance (queda en mejoras_harness.md)

Dual-k (`recall_ret` post-joiner), nDCG, abstención determinística, tokens/latencia por
pregunta, `pipeline_signature`, recalibración de SAS, baseline/answer-relevancy de Tier 3
(H7). El esquema mínimo + la migración automática de `csv_store.py` están pensados para
absorber esas columnas cuando lleguen.
