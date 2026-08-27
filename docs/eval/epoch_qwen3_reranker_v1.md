# Epoch `qwen3-reranker-v1`: apertura, limpieza de `runs.csv` y baseline pendiente

> **Estado (2026-08-27): COMPLETO.** Epoch abierta, historial limpio y baseline nuevo
> fijado en `20260827T160439Z` — la primera corrida completa de `qwen3-reranker-v1`
> sobre `main` (`17e9960`), 39 preguntas, **0 errores**. Los números están en
> [§5](#5-la-punta-20260827t160439z).

Este documento cubre tres cosas hechas el 2026-08-27 sobre `main`, todas
consecuencia del merge del 2026-08-26 (`17e9960`).

## 1. Por qué hacía falta una epoch nueva

El merge de `extract_indexing` trajo el embedder de Fran: **`bge-m3` (1024 dims) →
`qwen3-embedding:4b` (2560 dims)**.

[`eval_harness.md`](eval_harness.md) es explícito: cambiar el modelo de embeddings es
**cambio de topología**, no un movimiento de valve. Cambia el espacio vectorial y
obliga a reindexar, así que un delta contra el baseline viejo estaría restando
números medidos en dos espacios distintos. La regla existe para eso.

Se declaró entonces `qwen3-reranker-v1` en
[`eval_meta.yaml`](../../src/pipeline/eval/eval_meta.yaml). La **forma** del pipeline
no cambió respecto de `reranker-v1` (retriever híbrido → reranker `bge-reranker-v2-m3`,
`retriever_top_k=15`, `ranker_top_k=4`); lo único que cambió es el embedder, y con eso
alcanza.

### Corrección: la corrida del 26/08 estaba mal etiquetada

`20260826T213340Z` se corrió **después** del merge, o sea sobre la topología nueva
(su `embedding_model` dice `qwen3-embedding:4b`), pero quedó estampada con
`epoch=reranker-v1` porque `eval_meta.yaml` todavía no se había actualizado.

Se le reescribió la celda `epoch` a `qwen3-reranker-v1`. **No es falsear un dato:** la
corrida efectivamente usó 2560 dims; la etiqueta era la que estaba mal. Dejarla como
estaba anulaba en la práctica el mecanismo de epochs, porque `report.py` la habría
comparado contra las corridas de agosto **sin emitir la advertencia** que justamente
existe para atrapar ese caso.

Invariante que ahora se cumple en toda la tabla: `epoch` y `embedding_model` son
consistentes entre sí.

## 2. Limpieza de `runs.csv`

`runs.csv` es la **única** cosa de `results/` que se versiona
([`.gitignore:41-42`](../../.gitignore)); `questions.csv` y `logs/` son locales de cada
máquina. Venía con **19 filas arrastradas del refactor CSV**, la mayoría corridas de
`--limit 1/2/3` hechas mientras se validaba cada paso en `eval-csv-turco`. Como es el
historial que el equipo lee, el ruido tenía costo real: para encontrar una corrida de
verdad había que filtrar a mano por `dataset_n`.

**Criterio adoptado:** sobrevive una fila si la corrida fue **completa**
(`dataset_n` = 39, o vacío en las dos migradas históricas) **y** terminó **sin errores**
(`n_errors` = 0). Quedaron **7 de 20**.

| Se conservó | Por qué |
|---|---|
| `20260703T210122Z` | baseline histórico del equipo, migrado de `baseline.json` |
| `20260814T032439Z` | migrada de `history.csv` |
| `20260818T202351Z` | baseline vigente de la epoch `reranker-v1` |
| `20260818T212046Z` | corrida idéntica al baseline (evidencia de los hallazgos 18–19) |
| `20260818T212732Z` | prueba `top_k=5`, corrida completa |
| `20260818T234116Z` | primera corrida con H6 (abstención medida de verdad) |
| `20260826T213340Z` | post-merge: primera de la epoch `qwen3-reranker-v1` |

Se borraron 13: 12 smoke tests y validaciones de paso (`--limit 1/2/3`) y una corrida
fallida del 2026-08-27 con 39 errores (ver §4).

**Lo que se perdió y conviene saber:** entre las 13 estaban `20260818T204144Z` y
`20260818T205726Z`, las **únicas dos filas `suite=judge` que existieron**. Eran
validaciones de n=3 del Paso 6, así que caen bajo el criterio, pero con eso el repo
queda **sin ningún registro de Tier 3**. La próxima corrida de `run_eval_llm.py` va a
ser, a efectos del historial, la primera.

La limpieza se propagó a `questions.csv` (195 filas = 5 corridas × 39) y a
`results/logs/` (12 logs huérfanos borrados). Se filtró **por líneas físicas**, no con
un round-trip de pandas, para no alterar el formato: el harness garantiza que cada fila
del CSV es una línea física, y eso se verificó después (`líneas == filas + 1` en ambas
tablas).

## 3. Los logs de eval nunca se subieron a git

Vale dejarlo escrito porque es un malentendido fácil.
[`.gitignore:41`](../../.gitignore) ignora **todo** `src/pipeline/eval/results/*` y hace
una sola excepción, `runs.csv`. Los `.log` de `results/logs/` y el `questions.csv`
(~400 KB) son **locales de cada máquina** y no llegan al repo por ninguna rama.

Consecuencia práctica ya conocida: el detalle por pregunta de una corrida hecha en otra
máquina no lo tenés, y `report.py` lo detecta y degrada al delta global con un aviso en
vez de romperse.

## 4. Incidente: la `GROQ_API_KEY` se perdió

Vale documentarlo porque la causa raíz sigue viva para cualquiera del equipo.

**Qué pasó.** `infrastructure/.env` (gitignoreado, local) **desapareció** en algún
momento entre la corrida del 2026-08-26 21:33 y el 2026-08-27. Ese archivo era el único
lugar donde vivían `LLM_MODEL` y `GROQ_API_KEY`.

**Primer síntoma.** Sin `.env`,
[`docker-compose.yml:67`](../../infrastructure/docker-compose.yml) resuelve
`LLM_MODEL: ${LLM_MODEL:-}` a vacío y el pipeline cae a su default hardcodeado
`meta-llama/llama-4-scout-17b-16e-instruct` — el modelo que Groq ya no sirve. Resultado:
las 39 preguntas fallaron con `404 model_not_found` (corrida `20260827T150445Z`, borrada
en la limpieza).

**Cómo se perdió la key.** El container `pipelines` seguía **vivo desde antes** de que
el `.env` desapareciera, así que su environment era la última copia de la key. Al
recrearlo para inyectarle el `LLM_MODEL` correcto, se fue con él. Verificado después:
no está en el entorno de Windows (User, Machine ni proceso) ni persistida en
`valves.json` (que sólo guarda `llm_model`, `embedding_model`, los `top_k`,
`max_tokens` y `temperature`).

> **Lección operativa:** antes de recrear un container cuyo `.env` no existe, volcar su
> environment. Un container corriendo puede ser la última copia de un secreto.

**Dos fragilidades reales que esto expone**, ambas anteriores a este incidente:

1. **`LLM_MODEL` vacío falla tarde y mal.** El default hardcodeado apunta a un modelo
   que ya no existe, así que un `.env` ausente no da un error de config al arrancar:
   da 39 fallos idénticos de 404 en medio del eval. Un chequeo temprano de
   `LLM_PROVIDER`/`LLM_MODEL` convertiría eso en un mensaje claro.
2. **`docker compose up -d pipelines` arrastra `ollama`** (está en `depends_on`), y si
   se invoca con el compose base, **`ollama` pierde la GPU**. Como indexar con
   `qwen3-embedding:4b` en CPU falla en silencio (`timed out` por lote, con exit 0),
   esto es una forma fácil de corromper el store sin enterarse. Para restaurarla:

   ```bash
   docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up -d ollama
   ```

   **Sólo a `ollama`.** El overlay nvidia también le pone `TORCH_DEVICE: cuda` y un
   `TORCH_INDEX_URL` de cu126 a `pipelines`, cuya imagen está construida con torch CPU:
   aplicárselo sin rebuildear rompe el reranker en runtime.

**Resuelto el mismo día.** `infrastructure/.env` se recreó con `LLM_PROVIDER=groq` y
`LLM_MODEL=openai/gpt-oss-120b`, y Valentino repuso una key nueva desde
console.groq.com. Las credenciales de la DB no se pusieron a propósito: la base ya está
creada con los defaults del compose (`avdbuser`/`avdbpass`) y declararlas distintas
rompería la conexión.

> **Cómo se reconoce una key de Groq mal pegada.** En el camino se probó una key con
> prefijo `sk-`, que es el formato de **OpenAI**; las de Groq empiezan con **`gsk_`**.
> Como `build_generator` siempre apunta a `GROQ_BASE_URL`
> ([`pipeline_ciberseguridad.py:214-216`](../../src/pipeline/pipeline_ciberseguridad.py)),
> una key de otro proveedor da `401 Invalid API Key` — distinto del `404
> model_not_found` que da el modelo inexistente. Los dos errores se parecen en que
> revientan las 39 preguntas por igual, pero el código dice cuál es cuál.

## 5. La punta: `20260827T160439Z`

Corrida completa sobre `main` (`17e9960`), 39 preguntas, **0 errores**, 965 docs en el
store, Groq `openai/gpt-oss-120b`, `temperature=0`. Es el nuevo `baseline_run_id`.

| Métrica | Valor | n |
|---|---|---|
| `recall_eff` | **0.841** | 22 con ground truth |
| `hit_eff` | 0.864 | 22 |
| `mrr_eff` | 0.689 | 22 |
| `source_recall` | **1.000** | 12 |
| `sas_mean` | 0.769 | 26 con `reference_answer` |
| `abstention_rate` | 0.600 | 5 con `expect_refusal` |

**El retrieval reproduce exacto.** `recall_eff` y `mrr_eff` dieron idénticos hasta el
cuarto decimal a `20260826T213340Z` (0.8409 / 0.6894), que es evidencia directa de que
el número es estable y no ruido de corrida. `sas_mean` sí se movió (0.750 → 0.769),
consistente con lo ya sabido: el retrieval es determinístico, la generación no lo es ni
siquiera con `temp=0` en Groq.

### Dónde está flojo

- **`multi_doc` recall 0.250** y **`desarrollar` 0.500** — las dos peores, igual que en
  la epoch anterior. El embedder nuevo no las movió.
- **`regresion` recall 1.000 pero mrr 0.292** — encuentra el documento y lo rankea
  ~3°/4°. Es un problema del reranker, no del retriever.
- **`concepto_es` 0.667** — la brecha es→en de siempre.
- Todas las categorías son n=2–3: **una sola pregunta mueve el número entero.** No leer
  estos deltas por categoría como si fueran señal fina.

### Dos fallos de generación que ninguna métrica global muestra

1. **`prevenir-deserializacion` se abstiene con `recall=1.00` y `rr=1.00`.** Retrieval
   perfecto — el documento correcto le llegó primero al LLM — y aun así responde que no
   sabe. Es un fallo de **generación**, ya visto el 2026-08-18 y de nuevo el 26/08:
   tres corridas, dos embedders y sigue igual.
2. **`capital-francia` y `receta-tarta` NO se abstienen** siendo `fuera_dominio`. La
   `abstention_rate` de 0.600 son exactamente estas dos de cinco. El RAG contesta
   preguntas que debería rechazar.

Las 6 "regresiones de SAS" que reportó `report.py` contra el baseline viejo son casi
todas de categorías negativas (`fuera_dominio` −0.169, `id_inexistente` −0.066), donde
los hallazgos 18–19 ya demostraron que el SAS premia el fraseo y no la corrección. No
son señal.

### Cómo reproducirla

```bash
./scripts/eval.sh --label "..."
```

El store tiene que estar poblado con **965 docs** y `ollama` **con GPU**:

```bash
docker compose exec -T ollama nvidia-smi --query-gpu=name --format=csv,noheader
docker compose exec -T pipelines python -c "import sys; sys.path.insert(0,'/app/pipelines'); \
  import pipeline_ciberseguridad as p; print(p.get_document_store().count_documents())"
```

Y si hay que recrear `pipelines` (por ejemplo tras tocar el `.env`), usar **`--no-deps`**
para no arrastrar `ollama` y tumbarle la GPU:

```bash
docker compose up -d --no-deps pipelines
```

## Pendiente de fondo, sin resolver

**El equipo sigue sin decidir el modelo de Groq.** `meta-llama/llama-4-scout-17b-16e-instruct`
está hardcodeado en cuatro lugares (`pipeline_ciberseguridad.py`,
`infrastructure/.env.example`, [`../modos_llm.md`](../modos_llm.md),
[`../arquitectura_groq.md`](../arquitectura_groq.md)) y ya no existe en Groq. Mientras
no se decida y se cambien esos cuatro lugares, cualquiera que levante el proyecto de
cero se choca con el mismo 404.

**El embedder nuevo no compró nada** (medido en `20260826T213340Z`): contra el baseline
de agosto, `recall_eff`, `hit_eff` y `source_recall` dieron delta **0.000** y el MRR
empeoró levemente (0.708 → 0.689). Es el número a llevar al equipo antes de que alguien
construya sobre el supuesto de que el embedder ayudó.

## Relacionado

- [eval_harness.md](eval_harness.md) — diseño del harness, tiers y la regla de epochs.
- [bitacora_refactor_csv.md](bitacora_refactor_csv.md) — hallazgos 18–19 sobre por qué
  el SAS miente en las preguntas negativas.
- [mejoras_harness.md](mejoras_harness.md) — H5/H6, abstención y umbral de regresión.
- [../modos_llm.md](../modos_llm.md) — `LLM_PROVIDER` / `LLM_MODEL`.
- [../../src/pipeline/eval/README.md](../../src/pipeline/eval/README.md) — uso diario y
  diccionario de columnas.
