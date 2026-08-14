# Bitácora de implementación: refactor de resultados a CSV

> **Estado (2026-08-14): EN CURSO.** Pasos 0–3 terminados y verificados.
> Pasos 4–10 pendientes. `run_eval.py` ya escribe `runs.csv` + `questions.csv`.
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
| 4 | Migrar `history.csv` y limpiar `results/` | ⬜ pendiente |
| 5 | `report.py` (reescritura completa) | ⬜ pendiente |
| 6 | `run_eval_llm.py` (Tier 3, mínimo) | ⬜ pendiente |
| 7 | Wrappers `scripts/eval.sh` | ⬜ pendiente |
| 8 | Git: `.gitignore` + `.gitattributes` | ⬜ pendiente |
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

---

## Desvíos respecto del plan

Ninguno funcional hasta ahora. El único punto de implementación libre:

- Los dicts `DTYPES_RUNS` / `DTYPES_QUESTIONS` se derivan de dos conjuntos de nombres
  de columnas de texto en vez de escribirse a mano columna por columna. El plan pide
  "dicts de dtype explícitos definidos en este módulo" y eso se cumple (siguen siendo
  dicts del módulo); derivarlos evita que se desincronicen del esquema al agregar una
  columna.

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
