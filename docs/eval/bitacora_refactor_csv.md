# Bitácora de implementación: refactor de resultados a CSV

> **Estado (2026-08-18): EN CURSO.** Pasos 0–4, 7 y 8 terminados y verificados.
> Del Paso 5 está hecha y validada **toda la capa de terminal** (§5.1–5.2:
> `compare`, `runs` y `question`), que es el gate obligatorio del plan. Falta el
> modo `html` (§5.3–5.6). Pendientes: resto del 5, 6, 9 y 10.
> `run_eval.py` ya escribe `runs.csv` + `questions.csv`, y el histórico está migrado.
> Rama: `eval-csv-turco` (sacada de `eval-luca` @ `d7c0418`). Implementa: Valentino.
>
> Este documento **no reemplaza al plan**: la especificación sigue siendo
> [refactor_resultados_csv.md](refactor_resultados_csv.md) y se sigue **al pie de la
> letra**. Acá se registra qué está hecho, qué falta, y —lo más importante— los
> **hallazgos** que aparecieron al implementar y que el plan no podía anticipar.

## Por qué existe esta bitácora

El plan de Luca está escrito para ejecutarse sin tomar decisiones nuevas. Pero al
bajarlo a una máquina real aparecieron cosas que no estaban en el análisis: el
entorno estaba roto de tres formas distintas, y la primera corrida del eval
**confirmó empíricamente** dos de los bugs que [mejoras_harness.md](mejoras_harness.md)
sólo había deducido leyendo el código. Eso merece quedar escrito.

---

## Avance por paso

| Paso | Qué es | Estado |
|---|---|---|
| 0 | Verificación previa (pandas, permisos) | ✅ hecho |
| 1 | `csv_store.py` | ✅ hecho y testeado |
| 2 | `eval_meta.yaml` | ✅ hecho y testeado |
| 3 | `run_eval.py` escribe crudo | ✅ hecho y testeado |
| 4 | Migrar `history.csv` y limpiar `results/` | ✅ hecho (con desvío, ver abajo) |
| 5 | `report.py` (reescritura completa) | 🟨 parcial: capa de terminal hecha; falta `html` |
| 6 | `run_eval_llm.py` (Tier 3, mínimo) | ⬜ pendiente |
| 7 | Wrappers `scripts/eval.sh` | ✅ hecho (con fix para Windows) |
| 8 | Git: `.gitignore` + `.gitattributes` | ✅ hecho (adelantado al Paso 4) |
| 9 | Documentación | ⬜ pendiente |
| 10 | Validación y baseline inicial | ⬜ pendiente |

### Paso 0 — verificación previa ✅

- `pandas 2.3.1` presente (viene con Haystack, como preveía el plan). ✔
- `pyyaml 6.0.2` presente. ✔
- Escritura en `results/logs/` OK. ✔
- **`matplotlib` PRESENTE**, contra lo que asume el plan (§5.0). Ver
  [Hallazgo 2](#2-matplotlib-está-pero-por-un-motivo-que-no-generaliza): la
  suposición del plan es correcta para la imagen default; esta máquina tiene la
  imagen "full". **El Paso 5.6 sigue siendo necesario.**

### Paso 1 — `csv_store.py` ✅

Implementado según §1.1–1.4, sin desvíos:

- `RUNS_COLUMNS` / `QUESTIONS_COLUMNS` copiadas **textuales** del plan (esquema
  mínimo; no se agregó ninguna columna "por las dudas").
- `_ser()` con la tabla de serialización del plan; `unescape_text()` como inversa.
- `append_row()` con los 5 pasos, incluida la migración de esquema con
  `os.replace` (rename atómico).
- `load_runs()` / `load_questions()` con dtypes explícitos; devuelven un DataFrame
  vacío **con el esquema puesto** si el archivo no existe.
- `load_meta()` (el plan lo permite acá): sin defaults silenciosos.
- `RENAMES` arranca vacío, como indica el plan.

**Verificado en el container** (`docker compose exec pipelines`):

| Caso | Resultado |
|---|---|
| `None`→vacío, `bool`→`1`/`0`, listas→`a\|b`, floats a 4 decimales | ✔ |
| Roundtrip `unescape_text(_ser(t)) == t` con saltos de línea | ✔ |
| Una fila = **una línea física** (`wc -l` confiable) | ✔ |
| Clave fuera del esquema → `ValueError` (atrapa typos) | ✔ |
| Migración: agregar una columna y sacar otra, filas viejas intactas + aviso de descarte | ✔ |

### Paso 2 — `eval_meta.yaml` ✅

Copiado literal del plan (§Paso 2), comentarios incluidos. `baseline_run_id: null`
—se fija en el Paso 10—, `epoch: reranker-v1`, y la lista documental `epochs`.

Verificado: `load_meta()` lo lee bien, y falla con mensaje claro si falta el archivo
o una clave obligatoria.

### Paso 3 — `run_eval.py` ✅

Implementado según §Paso 3, puntos 1–9. El script pasó a **escribir crudo, no
comparar**: ya no emite `<stamp>.json`, `history.csv` ni lee `baseline.json`.

- `make_run_id()` — timestamp UTC, con sufijo `-2`/`-3` si colisiona dentro del
  mismo segundo (consulta la columna `run_id` de `runs.csv`).
- Log movido a `results/logs/<run_id>.log`; el `_Tee` se mantiene igual.
- `git_metadata()` — lee `GIT_COMMIT`/`GIT_BRANCH`/`GIT_DIRTY` del entorno. **Nunca**
  corre `git` adentro del container (no hay `.git` montado); si faltan, quedan vacías.
- Flags: `--label` nuevo, `--baseline` eliminado, `--set-baseline` con la semántica
  nueva (reescribe una línea de `eval_meta.yaml`).
- `set_baseline()` — reemplazo por regex sobre el texto crudo, no `yaml.safe_dump`.
- `question_row()` / `run_row()` — el mapeo de §3.5 y §3.6, con `rank_first_hit`
  calculado directo (no `round(1/rr)`) y `effective_llm_model()` para el modelo.
- `by_category` **deja de persistirse** (report.py lo recalcula con groupby).
- Cortesía final: `report.compare(...)` en `try/except` — los datos se escriben
  **antes** de cualquier análisis.

**Verificado con una corrida real** (`--limit 3 --label "..."`):

| Qué | Resultado |
|---|---|
| `runs.csv` (1 fila × 35 columnas) y `questions.csv` (3 × 27) creados | ✔ |
| Log en `results/logs/<run_id>.log` | ✔ |
| `--set-baseline` reescribe la línea **conservando los 13 comentarios** y el historial de `epochs` | ✔ |
| Si no encuentra la línea → error claro y archivo **intacto** | ✔ |
| Migración de esquema sobre el `questions.csv` real: columna nueva → filas viejas intactas; al revertir, aviso de descarte | ✔ |
| `report.compare` no existe todavía → warning y los datos igual persistidos | ✔ (esperado hasta el Paso 5) |

**Lo que la fila nueva registra y antes era invisible:**

```
llm_provider: ollama     llm_model: llama3.1      ← hallazgo 8, arreglado
ranker_top_k: 4                                   ← habilita H1
n_docs_store: 965                                 ← habría detectado el hallazgo 7
sas_std: 0.0698                                   ← insumo directo para H5
```

Y en `questions.csv`, `joined_ids` guardó **16 candidatos** del joiner contra los 4
de `retrieved_ids`: el dato que H1 necesita para medir el techo del retriever, que
hasta ahora se descartaba en cada corrida.

---

## Hallazgos

### A. Del entorno (bloqueantes, le pasan a cualquiera)

Nada de esto es culpa del refactor, pero **impide correr el eval** en una máquina
limpia. Vale avisarle al equipo.

#### 1. La imagen de `pipelines` quedó desactualizada respecto de `requirements.txt`

La imagen tenía **`haystack-ai 3.0.0`** instalado. Haystack 3.0 elimina
`SentenceTransformersSimilarityRanker`, que importa `pipeline_ciberseguridad.py`
→ el módulo no carga.

Es exactamente el bug que Migue documentó y arregló en `requirements.txt`
(`a9c8971`, 2026-08-07, `haystack-ai>=2.30,<3`). **El pin está bien; lo que faltó
fue rebuildear la imagen.** Un pin sólo actúa al construir.

Por el mismo motivo faltaban tres dependencias que sí están declaradas:
`mdit-plain`, `python-docx` y `datasets`.

**Arreglo temporal aplicado** (vive sólo en el container corriendo; se pierde al
recrearlo):

```bash
docker compose exec -T pipelines pip install --no-cache-dir -r /dev/stdin < requirements.txt
```

**Arreglo durable** (pendiente, alguien del equipo debería commitearlo como tarea):

```bash
cd infrastructure
docker compose build --build-arg INSTALL_MARKER=true pipelines
```

#### 2. `matplotlib` está, pero por un motivo que no generaliza

El plan (§5.0) dice que matplotlib no está en la imagen. En esta máquina **sí está**,
pero como dependencia transitiva: la imagen fue construida con `INSTALL_MARKER=true`
→ `marker-pdf` → `seaborn` → `matplotlib`.

El default del Dockerfile es `INSTALL_MARKER=false`, así que **en la imagen default
la suposición del plan es correcta** y el Paso 5.6 hace falta igual. Conclusión: no
tocar el plan; sí tener en cuenta que un rebuild sin `INSTALL_MARKER=true` **pierde**
marker-pdf y matplotlib.

#### 3. Síntoma engañoso: el pipeline "desaparece" a `src/pipeline/failed/`

Si `Pipeline()` tira una excepción al instanciarse, el server de OpenWebUI Pipelines
**mueve el archivo** a `failed/` y loguea `WARNING:root:No Pipeline class found in
pipeline_ciberseguridad`. El mensaje miente: la clase existe, lo que falla es el
constructor. El error real no aparece en el log.

Para diagnosticarlo hay que importar el módulo a mano:

```bash
docker compose exec -T pipelines python -c \
  "import sys; sys.path.insert(0,'/app/pipelines'); import pipeline_ciberseguridad"
```

**Efecto secundario que confunde todavía más:** con el `.py` movido, `import
pipeline_ciberseguridad` resuelve al **directorio** `pipeline_ciberseguridad/` (el de
`valves.json`) como namespace package. El síntoma no es `ImportError` sino
`AttributeError: module 'pipeline_ciberseguridad' has no attribute 'get_document_store'`,
que manda a investigar al lado equivocado.

#### 4. `LLM_PROVIDER` faltante en `.env` rompe el arranque en máquinas sin Groq

Si `.env` no declara `LLM_PROVIDER`, toma el default `groq` y `build_generator()`
pide `GROQ_API_KEY` → excepción → el pipeline va a `failed/` (hallazgo 3).

`.env.example` ya trae ambas variables; los `.env` viejos (previos a
[modos_llm.md](../modos_llm.md)) no. Además, con `LLM_PROVIDER=ollama` hay que
setear `LLM_MODEL` explícito si no se tiene bajado el default
`qwen2.5:3b-instruct` (`DEFAULT_OLLAMA_LLM`).

#### 5. El corpus local puede estar contaminado

La base tenía **332 documentos y ninguno de ciberseguridad**: eran chunks de
`Informe OpenClaw_ Qué, Cómo y Dónde.docx`, indexados desde una ubicación de corpus
vieja (`infrastructure/appdata/rawdata/`). El corpus real es **`data/raw/`**, que es
lo que el volumen `rawdata` bind-montea.

Tras `TRUNCATE` + reindexado: **965 documentos** = 710 CWE atómicos + 255 chunks de
las 3 guías INCIBE. Los 710 coinciden con lo documentado en
[eval_harness.md](eval_harness.md).

### B. De la primera corrida (validan el diagnóstico de `mejoras_harness.md`)

Corrida completa del eval con el corpus ya sano (39 preguntas, `retriever_top_k=15`,
`ranker_top_k=4`, generación con `llama3.1` en Ollama local):

```
Tier 1 RETRIEVAL (n=22):  recall@k=0.841  hit_rate=0.864  mrr=0.708
Tier 1b FUENTE  (n=12):   source_recall=1.000  source_hit=1.000
Tier 2 SAS      (n=26):   sas=0.771
```

#### 6. H1 confirmado en vivo: el recall está clavado

`recall@k = 0.841`, **idéntico** al `0.8409` del `baseline.json`. Y no es que no
cambió nada: el baseline es `top_k=3` **sin reranker**; esta corrida es `top_k=15`
**con** reranker. Dos sistemas distintos, mismo número exacto.

Es literalmente lo que predice H1 en [mejoras_harness.md](mejoras_harness.md): la
métrica se calcula sobre la salida del reranker (siempre 4 docs), así que **es ciega
a lo que haga el retriever**. Ya no es una deducción del código: está reproducido.

Refuerza la prioridad de H1 y de persistir `joined_ids` (que el Paso 3 ya deja
guardado, habilitando el `recall_ret` sin migración futura).

#### 7. Las "regresiones" reportadas no son regresiones

El reporte marcó `xss-explicar-prevenir 1.00 → 0.00`. Investigado:

```
Pregunta: ¿Qué es el Cross-Site Scripting y cómo se previene?
Esperaba: cwe-79
Trajo:    4 chunks, todos de INCIBE-CERT_ESTUDIO_VULNERABILIDADES_PERSISTENTES_XSS_v1.0.pdf
```

El estudio del INCIBE sobre XSS le ganó el puesto a `cwe-79`. El RAG **no empeoró**:
el ground truth sólo acepta `cwe-79`, y el baseline se hizo el 2026-07-03 sobre un
corpus de 710 docs **sin las guías INCIBE**. Son corpus distintos (710 vs 965).

Argumenta a favor de dos cosas ya previstas: la columna **`n_docs_store`** en
`runs.csv` (hoy el snapshot no registra el tamaño del corpus, así que este confound
es invisible) y el **`acceptable_doc_ids`** de H3 en `mejoras_harness.md`.

#### 8. El snapshot registra el LLM equivocado (confirma el helper del Paso 3.6)

La corrida se hizo con `LLM_PROVIDER=ollama` y `LLM_MODEL=llama3.1`. El snapshot
guardó:

```json
"llm_model": "meta-llama/llama-4-scout-17b-16e-instruct"
```

Porque `run_eval.py:305` guarda `valves.llm_model`, que es el default de Groq y no
cambia con el proveedor. **Toda corrida local quedó registrada como si fuera Groq.**

Por eso se adelantó `effective_llm_model(valves)` (§Paso 3.6) antes que el resto del
Paso 3: no es una prolijidad, es un dato corrupto que ya está entrando al histórico.

### Paso 4 — migración y limpieza ✅ (con desvío)

**El plan no aplicaba como estaba escrito.** Asume 3 corridas históricas en
`history.csv` (una del 2026-07-03 y dos del 2026-07-10). En esta máquina no existen:
`results/` está gitignoreado y **nunca se commiteó nada de esa carpeta**, así que
esas corridas viven sólo en la máquina de quien las corrió. Son irrecuperables desde
acá.

Lo que sí había en `history.csv` eran **2 filas nuestras** del 2026-08-14 (la prueba
de 3 preguntas y la corrida completa). El chequeo de integridad de §4.1 dio limpio:
las 3 líneas con 14 columnas, consistentes con el header — el archivo lo generó el
código actual desde cero, así que la desalineación que advertía el plan no ocurrió.

**Desvío: se migró `baseline.json` en vez de descartarlo.** El plan dice
`git rm src/pipeline/eval/baseline.json` a secas (§4.5). Pero ese archivo **sí** está
versionado y contiene la corrida del equipo del 2026-07-03 completa (config, recall
0.8409, hit 0.8636, mrr 0.6932, SAS 0.7549 — exactamente una fila de `runs.csv`).
Descartarlo habría perdido el único punto de comparación histórico que sobrevive en
el repo. Se migró como fila con `epoch: pre-reranker` (top_k=3 y sin etapa `ranker` en
su config) y recién después se borró.

**Desvío: se corrigió el `llm_model` de las 2 filas migradas.** El plan mapea columnas
tal cual. Acá se registró `llm_provider=ollama` / `llm_model=llama3.1` en vez del
`meta-llama/llama-4-scout` que decía el archivo, porque es un valor **conocido como
falso** (hallazgo 8) y se verificó contra el `.env` y los logs de Ollama. Migrar el
dato mentiroso habría metido dos filas incomparables en la tabla nueva. Por el mismo
criterio se completaron `ranker_top_k=4` y `n_docs_store=965`, que constan en la
cabecera de los logs de esas corridas.

La fila de `baseline.json` **no** lleva esa corrección: esa corrida sí fue por Groq.

**Desvío de orden: el Paso 8 se adelantó.** Se aplicaron el `.gitignore` y el
`.gitattributes` de §Paso 8 **antes** de borrar `baseline.json`. Con `results/`
enteramente ignorado, borrarlo primero habría dejado una ventana con cero datos
históricos versionados. Verificado: `runs.csv` es versionable y `questions.csv` +
`logs/` siguen ignorados.

`results/` quedó como pide el criterio de terminado: sólo `runs.csv`,
`questions.csv` y `logs/`.

**`runs.csv` resultante (4 filas):**

```
run_id            epoch         provider  modelo      top_k  rank_k  docs  n_gt  recall_eff
20260703T210122Z  pre-reranker  groq      llama-4-...     3       –     –    22      0.8409
20260814T030325Z  reranker-v1   ollama    llama3.1       15       4   965     3      1.0
20260814T032439Z  reranker-v1   ollama    llama3.1       15       4   965    22      0.8409
20260814T035911Z  reranker-v1   ollama    llama3.1       15       4   965     3      1.0
```

Con la tabla armada, **H1 se ve de un vistazo**: la primera y la tercera fila tienen
el mismo `recall_eff` (0.8409) sobre el mismo `n_gt` (22), con topologías y `top_k`
distintos. Es el mismo hallazgo 6, pero ahora legible sin abrir un solo archivo — que
es exactamente lo que el refactor buscaba.

### Paso 7 — wrappers ✅ (con fix para Windows)

`scripts/eval.sh` y `scripts/eval_llm.sh` según §Paso 7, con una línea agregada.

**Hallazgo 9 — el wrapper del plan no corre en Windows.** Tal cual está escrito falla:

```
python: can't open file '/app/C:/Program Files/Git/app/pipelines/eval/run_eval.py'
```

Git Bash (MSYS) reescribe los argumentos que parecen rutas POSIX, así que
`/app/pipelines/eval/run_eval.py` llega al container como una ruta de Windows. Se
resuelve con `export MSYS_NO_PATHCONV=1` al principio del script; **no tiene efecto en
Linux/macOS**, así que la solución sirve para todo el equipo sin bifurcar el script.

Se adelantó este paso (antes que el 5) porque hasta ahora **todas** las corridas
quedaron con `git_commit`/`git_branch` vacíos: ese dato lo inyecta el wrapper y no
existía. Verificado tras el fix:

```
run_id            label            git_commit  git_branch      git_dirty
20260814T162753Z  prueba wrapper   217f7ce     eval-csv-turco  1
```

---

### Paso 5 — `report.py` 🟨 (capa de terminal hecha; `html` pendiente)

Se respetó el **orden interno obligatorio** del plan: primero la capa de terminal, que
es el gate diario y no depende de matplotlib. Dentro de esa capa se hizo primero
`compare` solo, porque es el modo que llama `run_eval.py` y el que hacía falta para
que desapareciera el `⚠ reporte falló` que arrastraban todas las corridas desde el
Paso 3.

El módulo viejo se reescribió entero: no queda nada que lea JSON (`load_snapshot`,
`promote` y `print_delta` sobre dicts ya no existen). Lo único que sobrevive es la
**semántica** de `_arrow`/`_fmt`, como pedía §5.2, ahora en `fmt_delta`.

Estructura según §5.5: `load_context` / `resolve_pair` para datos, un bloque de
funciones de análisis puras (`config_diff`, `category_table`, `question_pivot`,
`regressions`, `improvements`, `question_membership`) que devuelven DataFrames sin
imprimir nada, y encima la capa de presentación. Esa separación es la que va a
permitir que el modo `html` reuse los mismos números sin recalcularlos.

**Validación hecha** (adelanta parte del §5.7, con dos corridas reales nuevas —
`20260818T135812Z` y `20260818T140451Z` — más un `results/` sintético para los casos
que los datos reales no producen):

| Caso | Resultado |
|---|---|
| corrida vs baseline, todo normal | ✅ exit 0 |
| epochs distintas (`pre-reranker` vs `reranker-v1`) | ✅ aviso de topología + config diff de 7 columnas |
| baseline sin filas en `questions.csv` | ✅ degrada a delta global con aviso, exit 0 |
| `dataset_n` distinto (3 vs 2) | ✅ aviso + nota de pregunta ausente |
| `baseline_run_id: null` | ✅ resumen solo, exit 0 |
| `results/` vacío | ✅ mensaje accionable, exit 2 |
| `run_id` inexistente | ✅ lista las últimas 5, exit 2 |
| regresiones de retrieval y de SAS + mejoras + pregunta nueva | ✅ las 3 listas, exit 1 con `--strict` |
| SAS que baja **menos** que el umbral | ✅ correctamente NO marcado como regresión |
| output con pipe | ✅ cero códigos ANSI |

Después se completaron los otros dos modos de terminal, con lo que la capa que el
plan marca como **gate obligatorio** (§5.1–5.2) queda cerrada:

- **`runs`** — historial (últimas 15 con `suite=retrieval`, `--all` suma las judge),
  baseline marcado con `*` pegado al `run_id` para que no se pierda de vista cuando
  la tabla es ancha, y sparklines de `recall_eff`, `mrr_eff` y `sas_mean`.
  La escala de la sparkline es **fija 0–1 y no min-max relativo**, como pide el plan:
  con min-max, una métrica que se movió de 0.80 a 0.82 se vería igual de dramática
  que una que fue de 0.10 a 0.95, y las series no se podrían comparar entre sí.
- **`question <qid>`** — ficha (categoría, `expected_ids`, pregunta des-escapada con
  `csv_store.unescape_text`), tabla cronológica con `via` (emb / kw / emb+kw / -)
  resuelto con `np.select` sobre los flags leídos como float, y la respuesta de la
  última corrida más la del baseline **sólo si difieren** (repetir el mismo texto dos
  veces empuja lo importante fuera de la pantalla). Con un id que no existe, sugiere
  con `difflib.get_close_matches` y sale con exit 2 — verificado: `cwe89-por-numer`
  sugiere `cwe89-por-numero, cwe79-por-numero, cwe798-por-numero`.

Validación de estos dos modos: tabla y sparklines sin excepciones sobre los datos
reales y sobre el sintético; `*` del baseline correcto; `runs` sobre un `results/`
vacío avisa y sale con 0; sparkline con faltantes (`·`) y con valores fuera de 0–1
(clampeados) probados aparte.

**Hallazgo 10 — `--top-k 1` no fuerza una regresión.** El §5.7.3 propone correr con
`--top-k 1` para provocar una regresión garantizada y así probar `--strict`. No
funciona: las preguntas `cwe*-por-numero` se recuperan por keyword y mantienen
`recall=1.0` aun con `top_k=1`. La corrida `20260818T140451Z` lo confirma. Para
validar ese camino hubo que armar un `results/` sintético. **Conviene corregir el
§5.7.3 del plan**, o el Paso 10 va a "validar" `--strict` contra un caso que nunca
dispara.

**Hallazgo 11 — `argparse.set_defaults()` muta los actions compartidos por `parents=`.**
Un flag global escrito **antes** del subcomando (`report.py --results X compare`) se
perdía y el reporte leía el `results/` de siempre. La causa no es el bug clásico de
argparse: `set_defaults()` escribe `action.default` sobre los objetos action, y
`parents=` los **comparte por referencia** entre el parser principal y el subparser.
El `default=SUPPRESS` que justamente evita el problema quedaba pisado. Se resolvió
usando `set_defaults` sólo para `cmd` y resolviendo los defaults reales después de
parsear (helper `_opt`). Verificado en las tres posiciones posibles del flag.

**Hallazgo 13 — definir "corrida completa" por la MODA deja afuera las corridas que
importan.** §5.4 (gráfico 1) define las corridas completas como las que tienen
`dataset_n == moda`. Al implementar la evolución del modo `runs`, el estado real de
la epoch vigente era:

```
dataset_n = 3   → 5 corridas   (pruebas con --limit de este mismo día)
dataset_n = 39  → 1 corrida    (la única sobre el dataset entero)
dataset_n = 2   → 1 corrida
```

La moda es **3**, así que la tendencia mostraba la evolución del smoke test y
excluía la única corrida real. Y el problema se agrava solo: cuanto más se usa
`--limit` para probar, más se afianza la moda equivocada. Es un caso donde el plan
razonó sobre un `results/` maduro y la realidad es un `results/` de desarrollo.

Decidido con Valentino (2026-08-18): **"completa" = `dataset_n` máximo de la epoch**.
Se queda dentro de los CSV (no mira `dataset.yaml`, respetando el principio de que
`report.py` sólo lee `runs.csv` / `questions.csv` / `eval_meta.yaml`) y es un cambio
de una línea. Contra conocida y aceptada: si algún día se achica el dataset a
propósito, el máximo apunta al tamaño viejo hasta re-basear — pero achicar el
dataset es deliberado y raro, y correr con `--limit` es diario. **Conviene llevarle
esto a Luca para que lo baje al plan.**

Efecto inmediato: con el criterio nuevo hay 1 sola corrida completa, así que la
evolución no se dibuja. §5.4 pide omitirla con menos de 2, pero omitirla en
silencio se lee como "no pasó nada", así que se agregó el aviso `aún no hay
historia suficiente…` diciendo cuál es el criterio.

**Hallazgo 12 — los faltantes llegan de tres formas.** Según el dtype de la columna,
un valor vacío es `None`, `float('nan')` o el `pd.NA` de las columnas string. El
tercero se imprimía literal como `<NA>` en el diff de config. Se normalizan los tres
en `_celda`, en un solo lugar.

---

## Desvíos respecto del plan

Ninguno funcional hasta ahora. El único punto de implementación libre:

- Los dicts `DTYPES_RUNS` / `DTYPES_QUESTIONS` se derivan de dos conjuntos de nombres
  de columnas de texto en vez de escribirse a mano columna por columna. El plan pide
  "dicts de dtype explícitos definidos en este módulo" y eso se cumple (siguen siendo
  dicts del módulo); derivarlos evita que se desincronicen del esquema al agregar una
  columna.

Del Paso 5 (todos chicos, ninguno cambia el comportamiento especificado):

- **El diff de config no cubre todo el grupo "config" de `runs.csv`.** §5.2.3 dice
  "columnas del grupo config". `CONFIG_COLUMNS` deja afuera tres: `dataset_n` y
  `n_errors` porque ya se avisan en el bloque de advertencias (mostrarlos dos veces
  resta atención al aviso, que es lo importante), y `duration_s` porque es un
  resultado, no una decisión: aparecería en el diff de **todas** las corridas y lo
  volvería ruido.
- **La columna `n` de la tabla por categoría es la de la corrida actual**, no el
  máximo de los dos lados. Con el máximo, comparando una corrida de 2 preguntas
  contra un baseline de 3 la tabla decía `n=3` al lado de un promedio calculado
  sobre 2. La categoría que sólo existe en el baseline cae de vuelta a su `n`.
- **`--run` y `--baseline` también cuelgan del parser principal.** §5.1 los muestra
  sólo bajo `compare`, pero la misma sección dice "sin subcomando ⇒ `compare` con
  defaults": si `report.py` sin más es un compare, `report.py --run X` tiene que
  andar. Escribir `compare` sigue siendo válido.
- **"Corrida completa" = `dataset_n` máximo de la epoch, no la moda** (§5.4). Es el
  único desvío del Paso 5 que cambia comportamiento especificado; está decidido con
  Valentino y explicado en el hallazgo 13. **Falta bajárselo a Luca.**
- **La evolución omitida avisa.** §5.4 dice omitir el gráfico con menos de 2 corridas
  completas; se agregó además la nota `aún no hay historia suficiente…` porque un
  bloque que desaparece sin decir nada se lee como "no pasó nada".
- **`runs` muestra `mrr_eff` además de `recall_eff` y `sas_mean`.** El ejemplo de
  §5.2 dibuja dos sparklines; se agregó la tercera porque `mrr` es la métrica que
  distingue "encontró el documento" de "lo puso primero", que es justamente lo que
  el reranker vino a mejorar — sin ella no se ve si el reranker está haciendo algo.

Si algún desvío real aparece más adelante, se registra acá **antes** de codearlo.

---

## Pendientes y riesgos conocidos

1. **Coordinación:** el plan lo escribió Luca (2026-08-12) y lo está implementando
   Valentino. Confirmar que Luca no lo arranque en paralelo — `report.py` se reescribe
   entero y un merge de dos versiones sería doloroso.
2. **Rebuild de la imagen** (hallazgo 1): mientras no se haga, cualquiera que levante
   el proyecto se choca con lo mismo, y los arreglos de esta máquina se pierden al
   recrear el container.
3. **El baseline del Paso 10 va a morir pronto.** Dos cambios en vuelo lo invalidan:
   el embedder nuevo de Fran (`qwen3-embedding:4b`, 2560 dims, en `extract_indexing`)
   y la ingesta de CVEs (hoy 0 en el store). Ambos son **cambio de topología** → epoch
   nuevo → re-basear. No es motivo para frenar el refactor (la plomería vale igual),
   pero no conviene invertir en que ese primer baseline sea perfecto.
4. **Tier 3 sin correr:** `run_eval_llm.py` usa Groq y esta máquina no tiene
   `GROQ_API_KEY`. El Paso 6 se puede implementar igual, pero la validación del
   Paso 10.6 va a necesitar una key o correrla otro integrante.

## Relacionado

- [refactor_resultados_csv.md](refactor_resultados_csv.md) — **la especificación**; esta
  bitácora sólo la acompaña.
- [mejoras_harness.md](mejoras_harness.md) — H1/H3/H6/H9/H10, fuera del alcance de este
  refactor pero validados por los hallazgos 6 y 7.
- [eval_harness.md](eval_harness.md) — diseño vigente del harness (Tiers 1–3).
- [../modos_llm.md](../modos_llm.md) — `LLM_PROVIDER`/`LLM_MODEL` (hallazgos 4 y 8).
