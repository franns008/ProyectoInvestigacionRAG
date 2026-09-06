# Explicación de hallazgos: del escaneo al LLM sin pasar por el retrieval

> **Estado: IMPLEMENTADO** (falta la prueba con el stack levantado). Rama `feature/idQuery`.
>
> Código: [`src/pipeline/explain/`](../src/pipeline/explain/),
> [`pipeline_dependencias.py`](../src/pipeline/pipeline_dependencias.py),
> [`explainedProvider.ts`](../extension/src/scan/explainedProvider.ts).
> 44 tests nuevos en [`tests/test_explain.py`](../tests/test_explain.py), que corren sin stack.
>
> Cierra la brecha que dejó la Fase 2: hoy la extensión muestra una tabla priorizada
> —correcta pero muda— y el RAG y el LLM están completamente desacoplados de ella.
> Acá se define cómo se explican los **primeros N hallazgos** (N=3) trayendo el contenido
> del corpus **por clave exacta** y mandándoselo al LLM como contexto.
>
> **Restricción de diseño (la misma de siempre):** el pipeline de chat
> [`pipeline_ciberseguridad.py`](../src/pipeline/pipeline_ciberseguridad.py) **no se toca**,
> [`deps/`](../src/pipeline/deps/) **sigue siendo lógica pura** —sin Haystack, sin base— y
> **no se reindexa nada**. Todo lo nuevo es aditivo.

## Qué se quiere

Que cada una de las primeras 3 tarjetas del panel, además de los campos duros que ya tiene
(versión de arreglo, CVSS, EPSS, KEV, identificadores), traiga **dos o tres frases** que
digan qué es esa vulnerabilidad y por qué importa — con la cita de dónde salió cada cosa.

El contrato de la extensión ya lo previó: `Finding.explanation` y `Finding.citations` son
**opcionales desde el día uno** en [`types.ts`](../extension/src/scan/types.ts). Esta
integración los puebla; no cambia el contrato.

## La decisión central: lookup por clave, no retrieval

Cuando el resolver termina, ya se tienen los strings exactos: `CVE-2023-4863`, `CWE-787`.
Buscarlos con embeddings + full-text + RRF + cross-encoder es usar una herramienta
probabilística para un problema determinístico, y el eval mide `recall_eff` **0.841** — o
sea que fallaría ~1 de cada 6 veces en encontrar un documento cuyo id ya se conoce.

Por eso este camino **no pasa por el pipeline RAG**: es un `SELECT` por metadata. Exacto, o
un "no está en el corpus" honesto.

```
requirements.txt
   │
   ├─ deps.resolver + deps.prioritize        (YA EXISTE, determinístico, sin stack)
   │      └─→ hallazgos priorizados  ──→  top-3
   │                                        │
   │                    ids: {cve, cwe_ids}  ▼
   ├─ lookup por clave en pgvector      ──→ 2 queries por metadata (NO retrieval)
   │      └─→ docs MITRE (cwe_weakness) + docs NVD (nvd_cve, hoy vacío — ver medición)
   │                                        │
   └─ prompt de resumen ──→ LLM ──────────→ explanation (2-3 frases) + citations
                                            (los campos duros NO los escribe el modelo)
```

## Medición previa: qué hay realmente en los datos (2026-09-06)

Se corrió el escaneo real sobre [`requirements_demo.txt`](../requirements_demo.txt) y se
cruzaron los ids que devuelve contra los datos del repo:

| | Presentes en los datos locales |
|---|---|
| **CWEs del escaneo** | **6 de 7** (falta `CWE-1321`) |
| **CVEs del escaneo** | **0 de 10** |

Causa del 0/10: el dump de [`data/raw/nvd/`](../data/raw/nvd/) tiene 8.000 CVEs y **todos
son de 1999–2004** — el fetcher paginó desde el inicio del catálogo y se cortó. Los CVEs que
devuelve el escaneo (`CVE-2023-4863`, `CVE-2019-11358`, `CVE-2020-7471`, …) no caen en esa
ventana. **No es un problema del lookup: el dato no está.**

Dos agravantes del mismo lado:

- [`run_indexing.py:63`](../src/pipeline/indexing/run_indexing.py#L63) tiene
  `include_cve=False` por defecto: aun con el dump correcto, indexar los CVE es un paso
  explícito (`--include-cve`, ~20 min de embedding en CPU).
- Los XML de MITRE en `data/raw/` cubren **710 weaknesses únicas** de las ~940 del catálogo.
  Suficiente para la demo; el faltante se tapa dejando caer el `cwec_latest.xml` completo.

**Consecuencia de diseño:** el lookup se escribe **igual para CVE y para CWE** —es el mismo
filtro por metadata—, pero el contexto del CVE sale **del advisory de OSV**, que ya viaja
dentro de cada hallazgo. Cuando el dump de NVD se actualice, la query de CVE empieza a
devolver filas y **no hay que cambiar código**.

## Decisiones tomadas

| # | Decisión | Elegido |
|---|---|---|
| 1 | Dónde corre la explicación | **Pipeline nuevo en `:9099`** (`pipeline_dependencias.py`) |
| 2 | Contexto del CVE | **Advisory de OSV** por ahora; el lookup por `cve_id` se escribe igual |
| 3 | Mitigaciones de MITRE en los CWE | **Postergado** (no se toca la indexación, no se reindexa) |

Sobre la 3: `XMLCWEConverter` guarda hoy sólo `Name` + `Description`; `Potential_Mitigations`
y `Common_Consequences` se descartan (`grep` sobre `src/`: cero coincidencias). Con eso, la
explicación de un CWE va a ser una definición correcta pero **genérica** — justo la prosa que
un LLM sin RAG también escribe. Es una limitación asumida a conciencia para esta iteración, y
es el primer punto a mejorar antes de defender el trabajo.

## Dónde vive cada cosa

```
src/pipeline/deps/                       SIN CAMBIOS — lógica pura, sin base, sin LLM
src/pipeline/explain/                    la parte impura (store y LLM inyectados)
    lookup.py      ids → documentos del store (filtro exacto por metadata)
    prompt.py      contexto en cascada + plantilla del resumen
    summarize.py   orquestación: hallazgos → explicaciones
    payload.py     validación de la entrada del pipeline (pura, testeable sin stack)
src/pipeline/pipeline_dependencias.py    Pipeline de OpenWebUI (2º modelo en :9099)
extension/src/scan/explainedProvider.ts  decora al escáner local con explicaciones
```

**Por qué un paquete nuevo y no `deps/`.** `deps/` no importa Haystack ni abre conexiones a
propósito: es lo que permite testearlo sin stack y lo que hace que la CLI corra en 3 s en
cualquier máquina. `explain/` es el lugar de lo impuro, y **recibe el store y el generador
inyectados** (no los construye), así sus propios tests también corren sin levantar nada.

## Cómo se conecta la extensión: decorador, no reemplazo

El escaneo lo sigue haciendo `LocalScannerProvider` (3 s, offline, sin stack). Un
`ExplainedProvider` lo envuelve, manda **sólo los 3 hallazgos de arriba** al `:9099`, y
completa `explanation`/`citations`.

```
ExplainedProvider.scan()
   ├─ LocalScannerProvider.scan()          → ScanResult completo (como hoy)
   └─ POST :9099/v1/chat/completions       → sólo top-3 {cve, cwe_ids, summary, details}
        model: "pipeline_dependencias"        ↓
                                            {"explanations": [{...}]} → se mergea al ScanResult
```

Por qué así y no mandando el `requirements.txt` entero al servidor:

- **Si el stack está caído, el escaneo sigue funcionando** y sólo faltan los párrafos. En una
  demo en vivo eso es la diferencia entre una limitación y un papelón.
- El servidor **no necesita el dump de OSV**, ni parsear requirements, ni la lógica de rangos.
  Sólo recibe ids y texto, hace el lookup y genera. Es la mitad del trabajo.
- No cambia `types.ts` (los campos ya son opcionales) ni el panel.

`cibersec.provider`, `cibersec.ragUrl` y `cibersec.ragModel` ya existen en
[`package.json`](../extension/package.json). Falta agregar `cibersec.ragApiKey` (el servidor
de Pipelines exige `Authorization: Bearer <PIPELINES_API_KEY>`).

> El stub de [`ragProvider.ts`](../extension/src/scan/ragProvider.ts) describe el otro
> reparto (mandar el manifiesto entero). Se deja como está, con una nota: el decorador es el
> camino elegido; el pipeline puede sumar el modo "escaneo completo" más adelante sin romper
> nada.

## Contrato del lookup

```python
@dataclass(frozen=True)
class ContextDoc:
    doc_id: str          # id en el store
    kind: str            # "nvd_cve" | "cwe_weakness"
    identifier: str      # "CVE-2023-4863" | "CWE-787"
    title: str
    content: str

def fetch_context(store, cve_ids: list[str], cwe_ids: list[str]) -> ContextIndex: ...
```

**Dos queries en total, no una por hallazgo.** Se juntan los ids de los 3 hallazgos, se piden
todos los `nvd_cve` de una y todos los `cwe_weakness` de otra, y el join hallazgo ↔ documento
se hace en memoria. Evita el N+1 y deja el costo en ~2 round-trips.

Filtro de Haystack (genera exactamente el SQL que se quiere):

```python
{"operator": "AND", "conditions": [
    {"field": "meta.source_type", "operator": "==", "value": "nvd_cve"},
    {"field": "meta.cve_id",      "operator": "in", "value": cve_ids},
]}
```

El equivalente crudo sería `... where meta->>'cve_id' = any(%s)`. Se prefiere
`filter_documents` porque es el mismo SQL pero sobrevive a un cambio de esquema de la
integración de pgvector y no obliga a manejar una conexión aparte. Que sea "SQL" no es una
elección de API: es que **no interviene ningún vector**.

### La trampa: `cwe_id` es un entero en el store

[`converters.py:84`](../src/pipeline/indexing/converters.py#L84) guarda `"cwe_id": int(cwe_id)`
→ **`787`**, mientras que OSV entrega **`"CWE-787"`**. Un filtro `in ["CWE-787"]` contra una
columna que tiene `787` no falla: devuelve **cero filas en silencio**, y el síntoma es "el
corpus no tiene el CWE" cuando sí lo tiene.

Es el bug más probable de toda esta integración. Normalización en un solo lugar
(`_cwe_key("CWE-787") -> 787`) y un test dedicado.

> Segunda asimetría, para no tropezarse después:
> [`index_nvd.py`](../src/ingestion/index_nvd.py) escribe `cwe_ids` como lista de strings
> `"CWE-787"`, y `converters.py` escribe el `cwe_id` pelado como int. Son dos indexadores
> distintos sobre la misma tabla. Como el filtro se hace por `cwe_id` (el del converter que
> corre en la indexación real), alcanza con normalizar a int; queda anotado que hay que
> unificarlo cuando se toque la indexación.

## Degradación en cascada del contexto

Ninguna fuente está garantizada. El contexto se arma por niveles y **se registra cuál se usó**:

| Nivel | Fuente | Requiere | Estado hoy |
|---|---|---|---|
| 1 | Doc MITRE del CWE (`cwe_weakness`) | corpus indexado | **6 de 7** de los CWE del escaneo |
| 2 | Doc NVD del CVE (`nvd_cve`) | corpus indexado con `--include-cve` | **0 de 10** (dump 1999-2004) |
| 3 | `summary` + `details` del advisory OSV | **nada** — ya viene en el `Finding` | siempre disponible |

El nivel 3 es el que sostiene la demo: viaja dentro del hallazgo, no toca la base, y es texto
autoritativo del advisory. Con la base caída o el corpus vacío, **igual hay resumen**.

Si los tres niveles vienen vacíos, no hay `explanation`: la tarjeta se renderiza como hoy.
**Nunca se deja que el modelo escriba de memoria** — es exactamente el fallo que este diseño
existe para evitar.

## Cuántas llamadas al LLM: una por hallazgo

Tres llamadas cortas, no una grande con los tres adentro:

- El mapeo hallazgo → explicación es **determinístico**; no hay que parsear qué párrafo
  corresponde a qué CVE.
- No hay contaminación cruzada: con tres CVEs en un mismo prompt, el modelo mezcla productos
  y versiones. Es el error más caro posible acá.
- Falla parcial = una tarjeta sin explicación, no las tres.

Con Groq son ~1 s cada una. Si el costo molesta, la alternativa (una sola llamada con salida
JSON por id) queda como plan B, no como default.

## Contrato de salida (no cambia)

**Python arma toda la tarjeta; el LLM escribe únicamente el párrafo.** El CVSS, el EPSS, el
flag de KEV, la versión de arreglo y los identificadores se insertan por concatenación desde
los datos estructurados. Un modelo no puede equivocarse en un score que nunca escribe.

Las `citations` **también las arma Python**, a partir de los `ContextDoc` que realmente se
mandaron — no se parsean de la salida del modelo.

## Prompt

Distinto al del chat: no responde una pregunta, resume un hallazgo concreto.

- **Rol:** explicar en 2-3 frases, en español, qué es esta vulnerabilidad y por qué importa.
- **Prohibido:** inventar identificadores, scores o severidades; repetir los campos duros (ya
  están en la tarjeta); recomendar acciones que no estén en el contexto.
- **Obligatorio:** si el contexto no alcanza, decirlo. Vale devolver vacío.
- **Delimitación:** el contenido de los advisories y de NVD es **texto de terceros**. Va entre
  delimitadores y con la instrucción explícita de que es dato, no instrucción. Es la
  superficie de *prompt injection* que el [plan de trabajo](plan_de_trabajo.md) declara
  (OWASP LLM01). Además se recorta cada documento (~1200 caracteres).

## El pipeline nuevo

`src/pipeline/pipeline_dependencias.py`. El directorio `src/pipeline/` ya está bindeado a
`/app/pipelines`, así que el servidor lo levanta solo y aparece como segundo modelo en el
`:9099`, al lado del chat.

- **Entrada:** `pipe()` recibe en `user_message` un JSON `{"findings": [...]}` con los top-3
  (ids, `summary`, `details`). Si no parsea, devuelve un error legible en vez de tratarlo
  como pregunta de chat.
- **Salida:** un JSON `{"explanations": [{"cve": ..., "explanation": ..., "citations": [...]}]}`
  como string. La extensión lo parsea.
- **Reuso:** importa `get_document_store()` y `build_generator()` de
  `pipeline_ciberseguridad` — ya son funciones módulo-level, no hace falta refactor. El
  import ejecuta los parches de torch de ese módulo; dentro del container ese costo ya está
  pago porque el pipeline de chat también se carga.

## Verificación

**Sin stack (pytest, como el resto de `deps/`):**

1. `_cwe_key("CWE-787") == 787` y el filtro arma el `in` con enteros.
2. Store falso con docs CWE → los 3 hallazgos resuelven su contexto con **exactamente 2
   llamadas** al store.
3. Store vacío → nivel 3 (advisory) → sigue habiendo explicación.
4. Todo vacío → `explanation` ausente, la tarjeta se renderiza igual.
5. Generador falso que devuelve basura → los campos duros del `Finding` quedan intactos.
6. JSON de entrada inválido al pipeline → error legible, no excepción.

**Con stack:**

7. Cobertura: dado un `requirements.txt`, qué porcentaje de los top-N encuentra contexto de
   nivel 1 / 2 / 3. **Ese número va antes que la prosa.**
8. Latencia extremo a extremo del panel (escaneo + lookup + 3 generaciones).

> **Estado de esta máquina (2026-09-06):** el corpus **no está indexado**. En `pgvdb` sólo
> existe `document_chunk` (tabla propia de OpenWebUI, 0 filas); no hay `ciberseguridad_docs`.
> Y el stack que está levantado bindea `/app/pipelines` a otro proyecto, no a este repo.
> **Paso 0 para probar de punta a punta: levantar el stack de este repo e indexar los XML de
> CWE** (`run_indexing.py` ya los toma por defecto; los CVE no hacen falta para esta
> iteración).

## Limitaciones honestas

- **El contexto del CVE sale del advisory, no de NVD.** Medido: 0 de 10. Se dice, no se
  disimula.
- **Los CWE están en el pretraining de cualquier modelo**, y sin las mitigaciones de MITRE la
  explicación no aporta lo que un modelo pelado no sabe. Decisión 3, asumida.
- **Sólo se explican los primeros 3.** Explicar los 144 destruiría el sentido de priorizar.
- **El advisory de OSV no está indexado** (era la Fase 3, paso 8). Por eso el nivel 3 se lee
  del `Finding` y no del store. Cuando se indexe, pasa a ser un lookup más y el código no
  cambia de forma.
- La inconsistencia de dimensión de embeddings entre
  [`pipeline_ciberseguridad.py`](../src/pipeline/pipeline_ciberseguridad.py) (2560,
  `qwen3-embedding:4b`) e [`index_nvd.py`](../src/ingestion/index_nvd.py) (1024, `bge-m3`)
  **no afecta este camino**: acá no se embebe nada. Es un argumento a favor del lookup por
  clave — funciona incluso mientras eso está sin resolver.

## Plan de implementación

| # | Paso | Estado |
|---|---|---|
| 1 | `explain/lookup.py` + normalización de `cwe_id` + tests con store falso | hecho |
| 2 | `explain/prompt.py`: contexto en cascada (CVE → CWE → advisory) y plantilla | hecho |
| 3 | `explain/summarize.py`: N llamadas, citas armadas en Python, degradación | hecho |
| 4 | `pipeline_dependencias.py`: parseo del JSON de entrada, salida JSON, reuso por import | hecho |
| 5 | `explainedProvider.ts` + ajustes nuevos + merge en el `ScanResult` | hecho |
| 6 | Script de cobertura (nivel 1/2/3) sobre un `requirements.txt` real | **pendiente** |

**Falta probarlo con el stack de este repo levantado.** Todo lo demás está verificado sin
stack: 98 tests en verde y una corrida de humo sobre el escaneo real de
`requirements_demo.txt` (2 queries al store para 3 hallazgos, cascada CWE→advisory, prompt
armado).

### Ajustes nuevos de la extensión

| Ajuste | Default | Para qué |
|---|---|---|
| `cibersec.explain` | `true` | pedir o no las explicaciones |
| `cibersec.explainTopN` | `3` | cuántas tarjetas se explican |
| `cibersec.explainModel` | `pipeline_dependencias` | qué pipeline redacta |
| `cibersec.ragApiKey` | `0p3n-w3bui` | `PIPELINES_API_KEY` del servidor |

## Relacionado

- [`escaneo_dependencias.md`](escaneo_dependencias.md) — diseño del escaneo y las 4 fases.
- [`escaneo_dependencias_setup.md`](escaneo_dependencias_setup.md) — puesta en marcha.
- [`modos_llm.md`](modos_llm.md) — de dónde sale el generador (`build_generator`).
- [`Arquitectura_RAG_Ciberseguridad.md`](Arquitectura_RAG_Ciberseguridad.md) — el desacople
  indexación/inferencia que este diseño respeta.
