# Informe de simplificación de la codebase

**Estado:** aplicado (ver la tabla de abajo) · **Fecha del análisis:** 2026-10-22 ·
**Aplicado:** 2026-10-22 a 2026-10-24 · **Rama:** `chore/refactor-to-thesis`

Resultado de un análisis de sólo lectura de toda la codebase (Python, extensión TS, scripts e
infraestructura), hecho después de eliminar el frontend de OpenWebUI. Cada hallazgo indica
archivo de origen, severidad, descripción y solución sugerida. Las líneas son aproximadas y
se refieren al estado de la rama en la fecha del informe; el código ya no está ahí.

## Estado de aplicación

| Ítem | Estado | Commits / nota |
|---|---|---|
| B1 | aplicado | `0646ec6` |
| B2 | aplicado | `916b878`: lo decide sólo la extensión (`cibersec.explainTopN`, default 3, máx. 10) |
| B3 / D3 | aplicado | `4a33d68`: `build_query()` en `pipe()` y en los tres runners; epoch `inputs-produccion-v1` |
| B4 | aplicado | `3826beb`: `load_splittable` / `load_atomic` compartidos con la indexación |
| B5 | aplicado | `2614a31` |
| M1 | aplicado | `f699c2b` y docs `63dc147`..`102606d` |
| M2 | aplicado | `31aeab6`, `8542c1c`, `9442dc7` |
| M3, M4, C8 | aplicado | `70e42ea`..`d070ad9` |
| M5 | aplicado | `e8e5273`..`955ee63` |
| M6 | aplicado | `e03cc1f` |
| M7 | aplicado | `f5d4bd0`, `c38f4de` |
| M8 | aplicado | `02578f8`, `719e591` |
| M9 | aplicado | `59f8691`, `8970bce`, `2f33288` |
| M10 | parcial | `docs/image.png` borrado; `data/processed/` se queda (decisión del equipo) |
| D1, D2, D4, D6–D10 | aplicado | `30c98d0`..`d455372` (`common/`, `eval/runner.py`, helpers de la extensión) |
| D5 | aplicado | `c1338e6`: `wireFinding` en la extensión y `finding_identifier` en `common/text.py` |
| C1 | aplicado | por D1 y B3: `pipe()` arma una `Query` y corre |
| C2 | aplicado | `a83244d` |
| C3 | aplicado | `a558d67`: CSS y JS de los webviews en `extension/media/` |
| C4 | aplicado | `e230a38` |
| C5 | aplicado | `3728c54`: torch 2.7 ya trae lo que emulaban los parches |
| C6 | descartado | funciona y no molesta; `RENAMES` ya se había sacado (`abe7bf4`) |
| C7 | aplicado | `340ab1a`, `847d05e`: `scripts/check_extension.py` reemplaza al `.sh` |
| X1 | aplicado | `4dc1d9a`: `requirements-host.txt`; imagen rebuildeada |
| X2 | aplicado | `ccd8cd5` |
| §5 docstrings | aplicado | `961ba94`..`70bfa96`; los `__init__.py` quedaron como estaban (opcional) |

Encontrado al aplicar, fuera del análisis original:

- `bdd9c85`: `resolve_vuln_ids` heredaba IDs del `system` de los hallazgos adjuntos (con
  `8d2698f`, que la movió a `chat/vuln_ids.py` para testearla).
- `1816d33`: el eval atribuye aciertos también al canal `id_lookup` (`via_id_lookup`).

**Severidad**

- **Alta**: rompe o distorsiona funcionalidad, o es peso muerto grande y sin riesgo de sacar.
- **Media**: duplicación o complejidad que encarece cambios futuros, o riesgo de divergencia.
- **Baja**: limpieza cosmética o de legibilidad.

**Verificado**: ✔ = confirmado a mano además del análisis automático; sin marca = reportado
por el análisis, a confirmar al implementarlo.

**Restricción transversal.** El servidor de Pipelines trata cada `.py` de la raíz de
`src/pipeline/` como un pipeline y lo mueve a `failed/` si no define `Pipeline`. Todo código
compartido nuevo tiene que vivir en un subpaquete (p. ej. `src/pipeline/common/`), nunca como
archivo suelto en la raíz. Tampoco se pueden renombrar ni eliminar `Pipeline`, `Valves`,
`pipe`, `on_startup`, `on_shutdown` ni `on_valves_updated`: el server los invoca por
convención.

---

## 1. Bugs de funcionalidad

### B1 — Modelo por defecto de Groq en el pipeline de dependencias ✔

- **Archivo:** `src/pipeline/pipeline_dependencias.py` (~73 y ~129)
- **Severidad:** Alta

El valve `llm_model` tiene como default `meta-llama/llama-4-scout-17b-16e-instruct`, un
residuo de la etapa en la que la generación iba a Groq. Hoy la generación corre siempre en el
Ollama local, así que si `LLM_MODEL` no está seteada las explicaciones apuntan a un modelo que
no existe y fallan.

El mensaje de error del mismo archivo ("revisá LLM_PROVIDER y la API key") también es de esa
etapa: ni `LLM_PROVIDER` ni la API key existen ya, así que el mensaje desorienta.

**Solución:** usar como default el mismo modelo de Ollama que el pipeline del chat
(`DEFAULT_OLLAMA_LLM`) y reescribir el mensaje de error para que indique revisar `LLM_MODEL` y
que el modelo esté descargado en Ollama.

### B2 — `top_n` de explicaciones incoherente entre extensión y servidor ✔

- **Archivos:** `extension/src/scan/explainedProvider.ts` (~68), `extension/package.json`
  (`cibersec.explainTopN`), `src/pipeline/explain/payload.py` (`MAX_FINDINGS`),
  `src/pipeline/pipeline_dependencias.py` (~76)
- **Severidad:** Media

La cantidad de hallazgos a explicar se decide en tres lugares con valores distintos: la
extensión tiene un ajuste con default 10 (y lo limita a 10), el servidor acepta como máximo 10
y el valve `top_n` del pipeline vale 3. El efecto real es 3, no los 10 que sugieren el ajuste y
el README de la extensión.

**Solución:** dejar una sola fuente de verdad, probablemente el valve del servidor, y que la
extensión mande todos los hallazgos que tenga sin recortarlos. Si se quiere conservar un tope
en la extensión, documentar que el valve manda.

### B3 — El eval mide una variante del pipeline distinta de la de producción

- **Archivos:** `src/pipeline/eval/run_eval.py` (~145), `src/pipeline/eval/run_eval_llm.py`
  (~154), `src/pipeline/eval/run_chunking_experiment.py` (~187), comparado con `pipe` en
  `src/pipeline/pipeline_ciberseguridad.py` (~479)
- **Severidad:** Alta (para la validez de las métricas)

Los scripts de eval arman el diccionario de inputs del pipeline a mano y no usan las mismas
funciones que producción: no pasan los IDs al canal `id_lookup`, que entonces aporta cero
documentos, y no pasan por `build_embedding_query` ni `build_ranker_query`. Las métricas que
reporta el eval corresponden a un pipeline que el usuario nunca ejecuta.

Además usan la función privada `_extract_vuln_ids` del pipeline, y el componente `VulnIdLookup`
tiene un comentario que justifica que `ids` sea opcional justamente porque el eval no lo pasa.

**Solución:** exponer en el pipeline una única función que construya los inputs a partir del
mensaje (y opcionalmente hallazgos e historial), usarla en `pipe` y en los tres scripts de
eval. Como cambia lo que se mide, hay que abrir una nueva epoch en el eval y re-baselinear; es
una decisión del dueño del eval.

### B4 — El experimento de chunking no ve los mismos documentos que producción

- **Archivo:** `src/pipeline/eval/run_chunking_experiment.py` (~64-116), comparado con
  `src/pipeline/indexing/run_indexing.py` (`Indexer.load_documents`)
- **Severidad:** Media

`load_splittable_corpus` reimplementa la carga de PDF/DOCX/MD/TXT de la indexación pero omite
dos pasos que sí hace producción: la limpieza de `<br>` en los Markdown convertidos
(`_md_source`) y el descarte de documentos en blanco (`_is_blank`). Las estrategias de
chunking se comparan sobre un corpus que no es el que se indexa de verdad.

**Solución:** exponer en `indexing/` una función de carga reutilizable y que el experimento la
use, en vez de mantener una copia.

### B5 — `on_startup` y `run_eval` cargan la tabla entera para contar documentos

- **Archivos:** `src/pipeline/pipeline_ciberseguridad.py` (~430),
  `src/pipeline/eval/run_eval.py` (~441)
- **Severidad:** Media

Ambos cuentan los documentos con `len(store.filter_documents())`, que trae todas las filas
con su embedding de 2560 floats a memoria sólo para obtener un número. Con el corpus completo
es lento y consume RAM innecesariamente; `run_indexing.py` ya documenta este problema.

**Solución:** usar `store.count_documents()`, que hace el conteo en la base.

---

## 2. Código muerto

### M1 — `src/ingestion/index_nvd.py` huérfano y roto ✔

- **Archivo:** `src/ingestion/index_nvd.py` (archivo entero, ~240 líneas)
- **Severidad:** Alta

Ningún script, Dockerfile ni código lo invoca; sólo lo mencionan algunos docs. Además está
desactualizado: usa el modelo `bge-m3` con dimensión 1024, mientras el store real usa
`qwen3-embedding:4b` con 2560, así que contra la base actual falla al escribir.

También reimplementa la conversión de CVEs que ya vive en `indexing/converters.py`, con el
riesgo de divergencia que eso implica.

**Solución:** borrarlo y actualizar las menciones en `docs/escaneo_dependencias.md`,
`docs/explicacion_hallazgos.md` y `docs/unificacion_ingesta_nvd.md`. Como este último es una
propuesta abierta que ya planteaba borrarlo, conviene confirmarlo antes.

### M2 — `infrastructure/appdata/rawdata/` versionado y sin uso ✔

- **Archivo:** `infrastructure/appdata/rawdata/` (~25 MB: XML de CWE, PDF, DOCX)
- **Severidad:** Alta

El compose monta `../data/raw/` como `rawdata`; esta carpeta es una copia vieja que nada
monta, y los propios docs lo dicen. Ocupa espacio en cada clon sin aportar nada.

**Solución:** sacarla del repo. Revisar de paso `infrastructure/appdata/pipelines/.gitignore`,
que parece huérfano, confirmando antes que no sea el que protege algún bind mount.

### M3 — `RagProvider`, un proveedor de escaneo que siempre falla

- **Archivos:** `extension/src/scan/ragProvider.ts`, `extension/src/extension.ts` (~19,
  ~146-151), `extension/package.json` (`cibersec.provider`, `cibersec.ragModel`),
  `extension/README.md`
- **Severidad:** Media

Es un stub que siempre lanza `ScanError`. Dependía de una fase en la que el servidor iba a
escanear dependencias, pero ese no es el camino actual: el pipeline de dependencias sólo
explica hallazgos que la extensión ya resolvió. Arrastra una rama en `extension.ts`, dos
ajustes de configuración y partes del README de la extensión.

**Solución:** borrar el stub, la rama que lo elige y los ajustes `provider` y `ragModel`, y
actualizar el README de la extensión, `docs/escaneo_dependencias_setup.md` y
`docs/explicacion_hallazgos.md`. Con sólo dos implementaciones, la interfaz `ScanProvider` se
puede simplificar después.

### M4 — `token` y `label` de `ScanProvider` sin uso

- **Archivos:** `extension/src/scan/types.ts` (~84-86), `localProvider.ts`,
  `explainedProvider.ts` (~51-70)
- **Severidad:** Baja

La interfaz declara un token de cancelación opcional, pero `extension.ts` nunca lo pasa, así
que los chequeos de cancelación en `explainedProvider.ts` no se ejecutan nunca. La propiedad
`label` la implementan todos los proveedores y nadie la lee.

**Solución:** quitar ambos de la interfaz y de las clases. Si más adelante hace falta
cancelar, agregar el token cuando haya un caller que lo use.

### M5 — Restos sueltos en Python ✔ (parcial)

- **Archivos y severidad:** Baja en todos los casos
  - `src/pipeline/pipeline_ciberseguridad.py` (~570): variable `sep` asignada y nunca usada ✔.
  - `src/pipeline/indexing/converters.py` (~26-27): `logger` y `CWE_RE` redefinidos, idénticos
    a los de unas líneas antes ✔.
  - `src/pipeline/deps/resolver.py` (~83, ~193, ~213): el campo `Requirement.marker` se
    calcula y nadie lo lee; los parámetros `include_malicious` e `include_withdrawn` nunca se
    pasan con otro valor fuera de un test.
  - `src/pipeline/eval/csv_store.py` (~84): mecanismo de renombres `RENAMES` que está vacío
    desde que se creó.
  - `src/pipeline/pipeline_ciberseguridad.py` (~195): parámetros `history_turns` y `max_ids`
    de `resolve_vuln_ids` que nadie overridea.
  - `src/pipeline/pipeline_ciberseguridad.py` (~355): `getattr` con default para
    `id_lookup_top_k`, cuando todos los callers usan los valves que ya lo definen.

Son restos de iteraciones anteriores que no afectan el comportamiento pero obligan a quien lee
a preguntarse para qué están.

**Solución:** borrar las variables y redefiniciones duplicadas; quitar los parámetros y campos
sin uso (actualizando el test de `include_withdrawn`); dejar `history_turns` y `max_ids` como
constantes de módulo; acceder a `id_lookup_top_k` directamente.

### M6 — Rama de tokens en formato Groq en `_log_token_usage`

- **Archivo:** `src/pipeline/pipeline_ciberseguridad.py` (~577-588)
- **Severidad:** Baja

La función tiene una rama para el formato de uso de OpenAI/Groq (`prompt_tokens`,
`completion_tokens`), que no aplica a `OllamaGenerator`: Ollama devuelve `prompt_eval_count` y
`eval_count`.

**Solución:** confirmar con una corrida real qué metadata devuelve Ollama y reducir la función
a esa rama (de ~25 a ~8 líneas).

### M7 — Formato viejo de hallazgos en el contexto del chat

- **Archivos:** `src/pipeline/chat/context.py` (~60-82, ~93, ~118),
  `tests/test_chat_context.py` (`test_acepta_el_hallazgo_suelto_de_antes`)
- **Severidad:** Baja

El parser acepta un hallazgo suelto en el mensaje `system` (formato del chat por hallazgo, ya
eliminado) y lee los campos `cve` y `aliases`, que la extensión no manda: `contextFor` en
`chatView.ts` sólo envía `vulnerability`.

**Solución:** quitar la rama del formato viejo, la lectura de `cve`/`aliases` y el test que la
cubre. Si se encara la unificación de payloads (D5), hacerlo ahí.

### M8 — Variables de entorno que nadie lee

- **Archivos:** `infrastructure/docker-compose.yml` (`PG_CONN_STR`),
  `infrastructure/env/pipelines.env` (`PG_CONN_STR`, `OLLAMA_BASE_URL`)
- **Severidad:** Baja

Ningún `.py` lee estas variables: las conexiones a la base y a Ollama están como constantes en
el código. Probablemente son residuo del frontend de OpenWebUI.

**Solución:** quitarlas del compose y del env, o, si se hace D4, que el módulo de configuración
común las lea y dejen de estar hardcodeadas. `TORCH_DEVICE` no se toca: podría leerla
marker-pdf.

### M9 — Exports, estilos y chequeos sin uso en la extensión

- **Archivos:** `extension/src/chatView.ts` (`MAX_ATTACHMENTS`), `markdown.ts`
  (`renderInline`), `references.ts` (`referenceUrl`), `styles.ts` (~207, `.button.secondary`),
  `panel.ts` (~437, ~449)
- **Severidad:** Baja

Tres símbolos se exportan pero sólo se usan en su propio archivo; una regla CSS no la usa
ningún elemento; en `panel.ts` hay un chequeo de `explanation` que el caller ya hizo y un
comentario huérfano pegado a una función con la que no tiene relación.

**Solución:** quitar los `export` innecesarios, la regla CSS, el chequeo redundante y el
comentario huérfano. Los campos de `Finding` que TS no lee (`cvss_vector`, `references`) se
quedan: son parte del contrato JSON de la CLI.

### M10 — Archivos de datos y de docs sin referencias

- **Archivos:** `data/processed/*.md` (3 archivos), `docs/image.png`
- **Severidad:** Baja

Ningún código ni doc los referencia; parecen salida vieja de conversión de PDFs y una imagen
suelta.

**Solución:** confirmar que no son material de la tesis y, si no lo son, sacarlos del repo.

---

## 3. Código duplicado

### D1 — `pipe` y `_stream` repiten la corrida y el logging

- **Archivo:** `src/pipeline/pipeline_ciberseguridad.py` (~493-515 y ~534-552)
- **Severidad:** Media

Los caminos con y sin streaming repiten el conjunto de seis componentes de
`include_outputs_from` y todo el bloque posterior a la corrida: log de documentos, log de
tokens, respuesta y tiempo final. Cualquier cambio en el logging hay que hacerlo dos veces.

**Solución:** extraer el conjunto de componentes a una constante y la corrida más su logging a
un único método que reciba opcionalmente el callback de streaming. `_stream` queda sólo con el
hilo y la cola.

### D2 — Helpers de texto e IDs repetidos entre módulos Python

- **Archivos:** `src/pipeline/pipeline_ciberseguridad.py` (`_message_text`, `_CWE_ID_RE`,
  `_CVE_ID_RE`, `_extract_vuln_ids`), `src/pipeline/chat/context.py` (`_message_text`, `_trim`,
  `_VULN_ID_RE`), `src/pipeline/explain/prompt.py` (`_trim`), `src/pipeline/explain/lookup.py`
  (`_CWE_PREFIX`), `src/pipeline/indexing/converters.py` (`CWE_RE`)
- **Severidad:** Media

`_message_text` está idéntica en dos módulos; `_trim` (mismo corte con `[…]` y mismo largo)
también; el armado del texto de un advisory (resumen más detalle) aparece en tres lugares,
uno en la extensión; y hay cuatro expresiones regulares distintas para IDs CVE/CWE.

Además `_extract_vuln_ids` es privada pero la usan los scripts de eval, y `resolve_vuln_ids`
es lógica pura que hoy no se puede testear sin importar Haystack.

**Solución:** crear un subpaquete `src/pipeline/common/` con un módulo de texto (mensaje,
recorte, texto de advisory) y otro de IDs de vulnerabilidad (extracción, normalización de CWE,
regex única), y que el pipeline, `chat/`, `explain/` e `indexing/` los importen. Mover
`resolve_vuln_ids` ahí o a `chat/` para testearla sin stack.

### D3 — Inputs del pipeline armados en cuatro lugares

- **Archivos:** ver B3.
- **Severidad:** Media

Es la causa estructural de B3: cuatro copias del mismo diccionario que ya divergieron.

**Solución:** la misma de B3; se lista aparte porque, aun sin re-baselinear, la duplicación
seguiría siendo un problema.

### D4 — Constantes de conexión y armado del store duplicados

- **Archivos:** `src/pipeline/pipeline_ciberseguridad.py` (~72-75),
  `src/pipeline/indexing/run_indexing.py` (~43-46, ~145-150)
- **Severidad:** Media

La conexión a la base, la URL de Ollama, el nombre de la tabla y la dimensión de embeddings
están copiados en el pipeline y en la indexación; `run_indexing.py` además construye el
`PgvectorDocumentStore` a mano en vez de usar `get_document_store`. La copia que tenía
`index_nvd.py` ya quedó desfasada (M1), lo que muestra el riesgo.

**Solución:** un módulo de configuración en `src/pipeline/common/` con esas constantes y
`get_document_store()`, usado por ambos. Al hacerlo, decidir si `run_indexing.py` pasa a
correrse como módulo desde `/app/pipelines`, lo que permitiría quitar su hack de imports.

### D5 — Dos formatos de hallazgo entre extensión y pipelines

- **Archivos:** `extension/src/chatView.ts` (`contextFor`),
  `extension/src/scan/explainedProvider.ts` (payload), `src/pipeline/explain/summarize.py`
  (`finding_identifier`), `src/pipeline/chat/context.py` (`_finding_key`)
- **Severidad:** Media

El chat manda los hallazgos con un campo `vulnerability` y las explicaciones con `cve` y
`osv_ids`. Cada pipeline tiene su propia función para identificar un hallazgo. Son dos
contratos para la misma entidad.

**Solución:** una sola función en la extensión que convierta un hallazgo al formato de envío y
un solo identificador en Python. Cambia el contrato del `:9099`, así que conviene hacerlo en
un commit propio y probar chat y explicaciones juntos.

### D6 — Utilidades repetidas en la extensión

- **Archivos:** `extension/src/chatView.ts`, `extension/src/scan/explainedProvider.ts`,
  `extension/src/panel.ts`, `extension/src/scan/types.ts`, `extension/src/extension.ts`
- **Severidad:** Media

Hay cuatro variantes de "identificador de un hallazgo" con fallbacks distintos, dos copias de
la normalización de nombre de paquete, de `trimSlash` y de `describe(error)`. Los defaults
`http://localhost:9099` y `0p3n-w3bui` se repiten cinco veces en `extension.ts`. El handler que
abre links externos validando `http(s)`, el JS que lo dispara y el HTML base del webview (CSP,
nonce, doctype) están duplicados entre el panel y el chat. En `chatView.ts` el HTML de una
respuesta se arma dos veces con una sola diferencia.

**Solución:** mover a `scan/types.ts` el identificador (con fallback como parámetro) y la
normalización de paquete; a un módulo común `trimSlash` y `describe`; leer URL y key del
servidor una sola vez; extraer un helper para abrir links externos y otro para el HTML base del
webview; y una función para el HTML de la respuesta.

### D7 — Fetchers de ingestion casi idénticos

- **Archivos:** `src/ingestion/fetch_osv.py`, `fetch_epss.py`, `fetch_kev.py`, `fetch_cwe.py`,
  `fetch_nvd.py` (~106-160)
- **Severidad:** Baja

Los fetchers de OSV, EPSS y KEV son el mismo archivo con tres constantes distintas, y el
`main()` con `--force` está copiado cuatro veces. En `fetch_nvd.py`, la descarga completa y la
de muestra repiten el recorrido por ventanas de fecha.

**Solución:** una tabla de fuentes (URL, destino, tamaño mínimo) con un único `main`
compartido, y una función común para el recorrido por ventanas en NVD. Ahorra unas 40 líneas;
es más estilo que peso muerto.

### D8 — Boilerplate repetido entre runners de eval

- **Archivos:** `src/pipeline/eval/run_eval.py` (~46-96), `run_eval_llm.py` (~47-93),
  `run_chunking_experiment.py` (~245), `report_chunking.py` (~67-79)
- **Severidad:** Baja

`run_eval_llm.py` copia la carga del dataset, el `_Tee`, la generación del id de corrida y la
metadata de git, con un comentario que justifica la copia. El experimento de chunking vuelve a
leer el yaml a mano, y `report_chunking.py` reimplementa una métrica que ya existe en
`metrics.py`.

**Solución:** un módulo `eval/common.py` con esas piezas (y `effective_llm_model`, hoy en
`csv_store.py`), y reusar la métrica de `metrics.py`.

### D9 — Configuración de logging repetida

- **Archivos:** `src/pipeline/pipeline_ciberseguridad.py` (~54),
  `src/pipeline/pipeline_dependencias.py` (~65), `src/pipeline/indexing/run_indexing.py` (~81)
- **Severidad:** Baja

El mismo formatter y handler se configuran tres veces.

**Solución:** una función común que devuelva el logger configurado, con archivo opcional.

### D10 — Duplicados menores

- **Severidad:** Baja

`pipeline_dependencias.py` importa `explain` dentro de `pipe` aunque el módulo es liviano y ya
importa `explain.payload` arriba; `_get_document_store()` es un wrapper de una línea;
`explain/summarize.py` hace una copia defensiva de una tupla inmutable y `as_dict` podría ser
`dataclasses.asdict`; `_STOPWORDS` tiene elementos repetidos; `deps/cli.py` busca rutas
alternativas que ningún fetcher produce.

**Solución:** mover el import al tope, inlinear el wrapper, simplificar `summarize.py`,
deduplicar las stopwords y usar en la CLI las rutas que efectivamente generan los fetchers.

---

## 4. Sobrecomplejidad

### C1 — `pipe()` hace demasiado

- **Archivo:** `src/pipeline/pipeline_ciberseguridad.py` (~439-515)
- **Severidad:** Media

Unas 75 líneas que mezclan extracción de contexto, resolución de IDs, armado de inputs,
logging y ejecución.

**Solución:** se resuelve en gran parte con D1 y B3/D3; quedaría en unas 25 líneas que se
leen de corrido.

### C2 — `_log_retrieved_docs` con cinco bloques casi iguales

- **Archivo:** `src/pipeline/pipeline_ciberseguridad.py` (~596-630)
- **Severidad:** Baja

**Solución:** recorrer una lista de (componente, título, si muestra score, ancho) en un solo
loop.

### C3 — CSS y JS embebidos en template strings

- **Archivos:** `extension/src/chatView.ts` (~370 de 823 líneas), `extension/src/panel.ts`
  (~200 líneas)
- **Severidad:** Media

Más de la mitad de `chatView.ts` es CSS y JS del webview dentro de strings, lo que dificulta
leer la lógica de la vista y no permite resaltado ni linting de ese código.

**Solución:** mover estilos y script a módulos propios (o a archivos `.css`/`.js` que se lean
al activar). Ojo: el JS embebido interpola constantes TS, que habría que pasar de otra forma.

### C4 — `timeoutMs` opcional con un default inalcanzable

- **Archivo:** `extension/src/scan/explainedProvider.ts` (~29, ~38, ~110, ~204)
- **Severidad:** Baja

`extension.ts` siempre pasa el timeout, así que la constante por defecto nunca se usa.

**Solución:** hacer el campo obligatorio y borrar la constante.

### C5 — Parches de torch para versiones viejas

- **Archivo:** `src/pipeline/pipeline_ciberseguridad.py` (~8-25)
- **Severidad:** Media (riesgo medio)

Los parches cubren APIs de torch 2.4 y 2.5, pero la imagen fija `torch==2.7.0`. Además fuerzan
`import torch` al importar el módulo, que es justo lo que `pipeline_dependencias.py` evita con
imports locales.

**Solución:** buildear la imagen y arrancar sin los parches; si el reranker carga bien,
eliminarlos.

### C6 — `csv_store.py` con más maquinaria de la necesaria

- **Archivo:** `src/pipeline/eval/csv_store.py`
- **Severidad:** Baja

Migración de esquema, renombres y escritura atómica para un CSV que escribe un solo runner de
uso manual. Funciona y no molesta.

**Solución:** no tocarlo salvo que haya otro motivo; si se toca, quitar al menos `RENAMES`
(M5).

### C7 — `scripts/check_explicaciones.sh` obsoleto

- **Archivo:** `scripts/check_explicaciones.sh` (~217 líneas)
- **Severidad:** Baja

Es un diagnóstico de un problema de arranque ya resuelto, con restos de Groq (`LLM_PROVIDER`,
`GROQ_API_KEY`) y de una rama vieja, y depende de un string del código del pipeline.

**Solución:** decidir si se borra o se reduce a los dos pasos que siguen siendo útiles
(verificar que el modelo esté registrado y hacer una llamada real).

### C8 — Descripciones de ajustes desactualizadas en la extensión

- **Archivo:** `extension/package.json` (`cibersec.ragUrl`, `cibersec.ragApiKey`)
- **Severidad:** Baja

Dicen "sólo se usa con el proveedor rag", pero se usan en el chat y en las explicaciones.

**Solución:** corregirlas al eliminar `RagProvider` (M3).

---

## 5. Docstrings y comentarios excesivamente largos

- **Severidad:** Baja en todos los casos.

Comentarios que cuentan historia de decisiones, mediciones o motivación de diseño que ya está
(o debería estar) en `docs/`. Entorpecen la lectura del código sin ayudar a entenderlo.
**Solución general:** dejar cada uno en 1-3 líneas que digan qué hace el código, y mover el
resto al doc indicado si todavía no está ahí.

| Ubicación | Largo aprox. | Destino del contenido |
|---|---|---|
| `src/pipeline/pipeline_dependencias.py` (encabezado) | 29 líneas | `docs/explicacion_hallazgos.md` |
| `infrastructure/docker-compose.nvidia-3gb.yml` (encabezado) | 52 líneas | doc nuevo de GPU con poca VRAM |
| `src/pipeline/deps/resolver.py` (docstring de módulo, desactualizado) | 29 líneas | `docs/escaneo_dependencias.md` |
| `src/pipeline/explain/prompt.py` (módulo) | 23 líneas | `docs/explicacion_hallazgos.md` |
| `src/pipeline/explain/lookup.py` (módulo y `cwe_key`) | 18 + 8 líneas | `docs/explicacion_hallazgos.md` |
| `src/pipeline/explain/summarize.py` (módulo) | 15 líneas | `docs/explicacion_hallazgos.md` |
| `src/pipeline/chat/context.py` (módulo) | 20 líneas | `docs/arquitectura.md` §4.4 |
| `src/pipeline/deps/prioritize.py` (módulo) | 21 líneas | `docs/escaneo_dependencias.md` |
| `src/pipeline/deps/cli.py`, `src/ingestion/fetch_all.py`, `fetch_cwe.py`, `fetch_nvd.py` | 14-24 líneas c/u | `docs/ingestion_fetchers.md` |
| `src/pipeline/pipeline_ciberseguridad.py` (`VulnIdLookup` y dos funciones de IDs) | 15 + 8 + 9 líneas | `docs/analisis_keyword_retriever.md` |
| `src/pipeline/pipeline_ciberseguridad.py` y `pipeline_dependencias.py` (comentarios de `sys.path`) | 8-10 líneas c/u | `docs/arquitectura.md` |
| `src/pipeline/indexing/run_indexing.py` (`_is_blank`, `_md_source`) | 14 + 12 líneas | `docs/data_splitting.md` |
| `src/pipeline/eval/report.py` (módulo y `corridas_completas`) | 20 + 18 líneas | `docs/eval/bitacora_refactor_csv.md` |
| `src/pipeline/eval/report_chunking.py`, `run_chunking_experiment.py`, `chunking_strategies.py` | 21-25 líneas c/u | `docs/data_splitting.md` |
| `src/pipeline/eval/run_eval.py`, `run_eval_llm.py` (comentarios internos) | 7-11 líneas c/u | `docs/eval/bitacora_refactor_csv.md` |
| `src/pipeline/eval/metrics.py` | 12 + 7 líneas | `docs/eval/bitacora_refactor_csv.md` |
| `src/pipeline/eval/csv_store.py` | 22 + 8 líneas | `docs/eval/eval_harness.md` |
| `extension/src/styles.ts` | 24 + 15 líneas | se elimina (historia de estilos) |
| `extension/src/panel.ts` (seis bloques internos y encabezados) | 6-16 líneas c/u | se elimina o va al README de la extensión |
| `extension/src/scan/ragProvider.ts` | 20 líneas | se va con M3 |
| `infrastructure/Dockerfile.pipelines` (tres bloques) | 7-12 líneas c/u | `docs/imagen_pipelines.md` |
| `infrastructure/docker-compose.yml` (bloque `marker_cache` comentado) | ~15 líneas | se elimina: es configuración desactivada |
| `requirements_cves.txt` (encabezado) | 26 líneas | `docs/escaneo_dependencias.md` |

Los `__init__.py` de `chat/`, `explain/` y `deps/` re-exportan más nombres de los que usan los
callers (por ejemplo, `explain/__init__.py` exporta 11 y se usan 2). Se pueden recortar, pero
es opcional.

---

## 6. Dependencias y documentación

### X1 — Dependencias de `requirements.txt` sin uso en el container

- **Archivo:** `requirements.txt`
- **Severidad:** Media (riesgo medio)

`datasets` y `pypdf` no se importan en ningún módulo (los PDF pasan por marker). `requests` y
`python-dotenv` sólo los usan los scripts de `src/ingestion/`, que corren en el host, no en el
container. Tocar este archivo invalida las capas pesadas de la imagen (torch y el modelo del
reranker).

**Solución:** confirmar con `pip check` en la imagen que ningún componente de Haystack los exige
y moverlos a un requirements del host o eliminarlos, agrupando el cambio con otro que ya
obligue a rebuildear.

### X2 — `README.md` raíz con restos

- **Archivo:** `README.md`
- **Severidad:** Baja

Tiene una viñeta vacía en "Incluye", una sección que menciona una carpeta `seven_wonders` que
no existe y emojis de más en la sección de generación local.

**Solución:** quitar la viñeta vacía y la sección obsoleta, y emparejar el estilo con el resto
de los docs.

---

## 7. Orden sugerido de implementación

Un cambio lógico por commit, de menor a mayor riesgo:

1. Borrar `index_nvd.py` y ajustar sus menciones (M1).
2. Sacar `infrastructure/appdata/rawdata/` (M2) y, si se confirma, `data/processed/` y
   `docs/image.png` (M10).
3. Restos triviales: `sep`, duplicados de `converters.py`, comentario huérfano, regla CSS
   (M5, M9).
4. `count_documents()` en el arranque y el eval (B5).
5. Default de Ollama y mensaje de error en el pipeline de dependencias (B1); import al tope
   (D10).
6. Eliminar `RagProvider`, `token` y `label` en la extensión (M3, M4, C8).
7. Helpers compartidos de la extensión (D6, C4).
8. CSS y JS del webview a módulos propios (C3).
9. `src/pipeline/common/` con texto e IDs (D2); en otro commit, configuración y store (D4) y
   logging (D9).
10. Unificar `pipe`/`_stream` y simplificar los logs (D1, C1, C2, M6).
11. Inputs unificados y `eval/common.py` (B3, D3, D8, B4). Obliga a re-baselinear.
12. Recorte de docstrings, un commit por directorio (sección 5).
13. Opcionales a validar con build: parches de torch (C5), `requirements.txt` (X1), variables
    de entorno (M8), payload único (D5, M7), `top_n` (B2).

Los tests de `tests/` no cubren `pipeline_ciberseguridad.py` ni la extensión. Los pasos 9 y 11
requieren correr `pytest`, y el 11 además una corrida completa de eval en el stack.

## 8. Decisiones pendientes

- ¿`data/processed/*.md` y `docs/image.png` son material de tesis o se borran?
- ¿Se borra `index_nvd.py` aunque lo mencione la propuesta abierta de unificación de NVD?
- ¿El paso 11 (eval con inputs de producción) entra en esta chore, con nueva epoch?
- ¿`scripts/check_explicaciones.sh` se borra o se reduce?
