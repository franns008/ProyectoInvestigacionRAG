# Escaneo de dependencias: `requirements.txt` → vulnerabilidades priorizadas

> **Estado (2026-08-26): PROPUESTO.** Plan de implementación. Las piezas de lógica pura
> (resolver de versiones y priorización) están **prototipadas y verificadas** con datos
> reales; el resto está por construir.
>
> **Restricción de diseño: no cambia la arquitectura.** Se respeta el desacople
> indexación (escribe) / inferencia (lee) descrito en
> [Arquitectura_RAG_Ciberseguridad.md](Arquitectura_RAG_Ciberseguridad.md), y el pipeline
> de chat existente ([pipeline_ciberseguridad.py](../src/pipeline/pipeline_ciberseguridad.py))
> **no se toca**. Todo lo nuevo es aditivo y se comunica por el mismo canal: pgvector.

## Objetivo

Que un desarrollador abra su `requirements.txt` en VSCode y obtenga, en una sola acción:

1. **Qué dependencias suyas son vulnerables**, resuelto de forma determinística contra la
   versión exacta que tiene fijada (no "este paquete tuvo CVEs alguna vez").
2. **Cuál arreglar primero**, priorizado por explotabilidad real y no por severidad teórica.
3. **Por qué y cómo**, explicado en lenguaje natural, con cada afirmación citada contra
   la fuente autoritativa (NVD, MITRE, CISA, el advisory de origen).

El módulo secundario "análisis de vulnerabilidades en código Python" del
[plan de trabajo](plan_de_trabajo.md) se materializa acá, pero por la vía de las
dependencias en lugar del código fuente. Ver [Justificación](#justificación).

## Justificación

### Por qué dependencias y no código fuente

El plan de trabajo proponía analizar código Python (Bandit + RAG). Se descarta para esta
etapa por dos razones:

- **Requiere un corpus de código vulnerable** para tener algo que mostrar y contra qué
  evaluar. No lo tenemos y construirlo es un proyecto en sí mismo.
- **Los CWE no son post-cutoff.** CWE-89 (inyección SQL) tiene décadas y está en los datos
  de entrenamiento de todos los modelos. Un LLM sin RAG identifica inyección SQL en código
  sin ayuda. Si la demo se monta sobre eso, el sistema con RAG empata y parece un adorno.

Las **CVE sí son post-cutoff**, y no viven en la lógica del código: viven en las versiones
de las librerías que uno importa. El artefacto natural es un manifiesto de dependencias.
Ahí el argumento del knowledge-cutoff es incontestable.

### Por qué OSV y no (solo) NVD

Medición sobre el dataset real de OSV para PyPI (descargado el 2026-08-26):

| Métrica | Valor |
|---|---|
| Advisories de vulnerabilidad (sin `MAL`) | 13.396 |
| Paquetes PyPI distintos cubiertos | 1.936 |
| **Con rangos de versión explícitos** | **99,8%** |
| Con alias a un CVE | 94,7% |
| Con severidad CVSS | 76,2% |
| Registros de paquetes maliciosos (`MAL`) | 11.676 |
| Tamaño del `all.zip` | 33 MB |

Lo decisivo es que OSV está **indexado por ecosistema + nombre de paquete** (`{"name":
"trac", "ecosystem": "PyPI", "purl": "pkg:pypi/trac"}`) y trae los rangos afectados de
forma estructurada (`events: [{introduced: "0"}, {fixed: "0.10"}]`).

NVD, en cambio, gira alrededor de **CPE**, un identificador pensado para productos
comerciales y sistemas operativos. Para librerías de un ecosistema de lenguaje eso implica:

- Los nombres CPE no coinciden con los de PyPI (hay que mantener una tabla de equivalencias).
- [`index_nvd.py`](../src/ingestion/index_nvd.py) hoy **descarta las versiones**:
  `_vendors_products()` parsea el CPE y se queda solo con `parts[3]` (vendor) y `parts[4]`
  (producto), sin tocar `versionStartIncluding` / `versionEndExcluding`.

El 94,7% de alias a CVE revela el matiz importante: **los CVE existen casi siempre en NVD;
el problema no es cobertura sino la clave de búsqueda.** OSV es la misma información
reorganizada alrededor de la clave que uno realmente tiene cuando mira un
`requirements.txt`. NVD queda como enriquecimiento (aporta el `cvss_score` numérico ya
parseado), no como prerequisito.

### Por qué hace falta priorizar (el hallazgo central)

Escaneo de un `requirements.txt` de 10 dependencias con versiones viejas fijadas:

```
CVEs que afectan tus versiones                  152
  filtrando por CVSS >= 7.0 (alta/crítica)       80   (53%)
  filtrando por EPSS >= 10% (prob. de exploit)   13   (9%)
  presentes en CISA KEV (explotadas HOY)          1   (1%)
```

**Filtrar por severidad alta deja 80 de 152: no filtra nada.** Nadie arregla 80 cosas. Y
filtrar por CVSS es exactamente lo que hace la mayoría de las herramientas.

El salto está en la diferencia entre *qué tan grave sería* (CVSS) y *qué está siendo
atacado ahora* (EPSS, KEV). Son preguntas distintas, y la segunda es la que ordena el
trabajo de un equipo real.

Esto es literalmente la **"Mejora 3 — priorización por explotabilidad real (EPSS + CISA
KEV)"** del cronograma del [plan de trabajo](plan_de_trabajo.md), fechada el 30/Ago. El
escaneo de dependencias es el caso de uso que la hace visible.

### Dónde gana el RAG

El embudo lo resuelve código determinístico (joins por identificador de CVE). El RAG hace
lo que un ranking no puede: **explicar**. Por qué esta primero, qué clase de debilidad es,
qué implica, qué hacer — con cada afirmación citada.

Sin RAG hay una tabla ordenada. Sin priorización, el RAG escribe 152 párrafos que nadie
lee. Se necesitan mutuamente, y ese es un argumento de diseño defendible.

### La cadena multi-hop

```
requirements.txt  →  advisory OSV  →  CVE (alias)  →  CWE  →  corpus MITRE
```

Verificado de punta a punta con un caso real:

```
requests==2.19.1
  → GHSA-j8r2-6x86-q33q
  → CVE-2023-32681
  → CVSS:3.1/AV:N/AC:H/PR:N/UI:R/S:C/C:H/I:N/A:N
  → CWE-200
```

Esta cadena es el desafío que el plan de trabajo declara como **"relaciones inter-documento
(multi-hop): la cadena CVE → producto afectado → mitigación no es capturada por ningún
chunk individual plano"**. El escaneo de dependencias no es una demo decorativa: es la
materialización del problema de investigación declarado.

**El 96% de los GHSA traen `cwe_ids` en `database_specific`**, así que OSV entrega el
esqueleto completo hasta el CWE. El último salto —convertir `CWE-787` en una explicación
citada— lo resuelve el corpus de MITRE **que ya está indexado**.

## El caso de la extensión y el decano

### Por qué una extensión de VSCode

El RAG hoy se usa por Open WebUI, que es un chat. Un chat, para alguien que no es del área,
se ve igual que ChatGPT: no comunica nada de lo que se construyó. Una extensión de VSCode
cambia el registro — no es "otro chatbot", es una herramienta en el lugar donde el problema
ocurre.

Es además **barato**: el servidor de Pipelines ya expone el RAG como API OpenAI-compatible
en el puerto `9099` (verificado: `GET /v1/models` lista `pipeline_ciberseguridad`). La
extensión es un cliente HTTP contra algo que ya funciona.

**Decisión: webview propio, no la API de chat participants de VSCode.** Esta última
requiere que la máquina tenga instalado Copilot Chat para hospedar el participante; la demo
no debe depender de una extensión de terceros.

### Disciplina de las dos superficies

La presentación tiene **dos visuales separadas** y la separación solo rinde si se respeta:

| Superficie | Qué muestra | Qué NO muestra |
|---|---|---|
| **Extensión VSCode** | La herramienta: hallazgos, prioridad, explicación, citas | Scores internos, etapas del pipeline, métricas |
| **Panel de evidencia** | El cuaderno de laboratorio: procedencia por retriever, scores del reranker, resultados del eval | — |

La extensión tiene que verse **aburrida y profesional**, como una herramienta que ya
existe. Toda la maquinaria vive en la segunda pantalla. Si se filtran internals a la
extensión, el conjunto se lee como proyecto de facultad; si la extensión parece un producto
y al lado hay una pantalla que revela la ingeniería que lo sostiene, el mensaje es
"construimos algo que funciona **y además** sabemos exactamente por qué funciona".

Logística sugerida: una sola pantalla partida (VSCode ~60% a la izquierda, panel a la
derecha), no dos monitores. La relación entre la acción y sus tripas tiene que verse de un
vistazo.

### El momento que se busca

La demo pasa de *"tu proyecto tiene 152 vulnerabilidades"*, que no le sirve a nadie, a
**"tu proyecto tiene 152 vulnerabilidades; empezá por esta, que se está explotando hoy"**.
Esa frase se entiende sin ninguna explicación previa.

Ejemplo de salida objetivo (todos los datos son reales y verificados; solo la redacción
del párrafo es ilustrativa):

> ⚠️ **3 dependencias con vulnerabilidades conocidas — 152 CVEs en total.**
> Priorizadas por explotabilidad real, esta es la única que se está explotando hoy:
>
> **🔴 Pillow 5.2.0 → actualizá a 10.0.1**
> CVE-2023-4863 · CVSS 8.8 · EPSS 100% · **en el catálogo CISA KEV**
>
> Desbordamiento de heap en libwebp, la librería que Pillow usa para procesar imágenes
> WebP. Un atacante puede provocar una escritura fuera de los límites de memoria enviando
> una imagen manipulada.
>
> Es una **escritura fuera de límites (CWE-787)**, de las categorías más peligrosas del
> catálogo de MITRE.
>
> **Fuentes:** NVD CVE-2023-4863 · GHSA-j7hp-h8jx-5ppr · parche Pillow #7395 · CWE-787
> (MITRE) · CISA KEV, catálogo del 26/08/2026
>
> *Las otras 151 no tienen exploit conocido circulando.*

**La frase final es la que más vale: decir que 151 no urgen es tan útil como señalar la que
sí.** Es la diferencia entre un escáner y un consultor — que es la palabra que el plan de
trabajo usa para describir el proyecto.

### Argumentos para preguntas previsibles

| Pregunta | Respuesta |
|---|---|
| *"¿No alcanza con darle búsqueda web al modelo?"* | Corpus auditable y acotado a fuentes autoritativas, todo local por confidencialidad, cita verificable en cada afirmación. |
| *"¿Y si le pregunto algo que no sabe?"* | Está diseñado para abstenerse. Anunciarlo **antes** convierte ese caso en confirmación, no en falla. Está medido: columna `correct_rejection` del eval. |
| *"¿Esto no lo hace ya Dependabot?"* | Dependabot lista. Esto prioriza por explotabilidad real y explica con fuentes citadas. Ver [limitaciones](#limitaciones-honestas). |

## Arquitectura (sin cambios)

Se respeta el desacople de [Arquitectura_RAG_Ciberseguridad.md](Arquitectura_RAG_Ciberseguridad.md):
**indexación escribe, inferencia lee, y se comunican solo por pgvector.**

```
INDEXACIÓN (batch, escribe)                    INFERENCIA (lee)
─────────────────────────────                  ─────────────────────────────
fetch_osv.py   ─┐                              pipeline_ciberseguridad.py   (chat — SIN CAMBIOS)
fetch_epss.py  ─┤→ converters.py →  pgvector ← pipeline_dependencias.py     (NUEVO)
fetch_kev.py   ─┤    run_indexing.py
(CWE completo) ─┘
```

### Dónde viven los datos de OSV

**En pgvector, como una fuente más**, con `source_type: "osv_advisory"` junto a los `cwe` y
`nvd_cve` que ya existen. Es lo que respeta el canal único de la arquitectura y reusa el
patrón de ingesta que ya está montado.

Metadata por advisory (todo filtrable, sin vectorizar):

```python
meta = {
    "source_type":   "osv_advisory",
    "osv_id":        "GHSA-j7hp-h8jx-5ppr",
    "pypi_package":  "pillow",          # normalizado PEP 503 → clave del lookup
    "ranges":        [...],             # events introduced/fixed, tal cual OSV
    "cve_ids":       ["CVE-2023-4863"],
    "cwe_ids":       ["CWE-787"],
    "cvss_vector":   "CVSS:3.1/AV:N/...",
    "cvss_score":    8.8,               # calculado desde el vector
    "epss":          1.0,               # estampado en la indexación
    "kev":           True,              # ídem
    "fixed":         "10.0.1",
    "references":    [...],
}
```

`content` = `summary` + `details` del advisory.

> **Decisión abierta — embeddings de los advisories.** El camino determinístico **no los
> necesita** (es un filtro por metadata). Embeber 13.396 advisories con
> `qwen3-embedding:4b` (2560 dims) no es gratis. Recomendación: indexarlos **sin embedding**
> en una primera pasada (el lookup por `pypi_package` funciona igual) y evaluar después si
> se quiere búsqueda semántica sobre el texto de los advisories.

> **Nota sobre EPSS/KEV.** Son volátiles (se actualizan a diario) y se estampan como
> metadata al indexar. Refrescarlos = volver a correr la indexación. Es consistente con el
> modelo batch de la arquitectura; si más adelante molesta, se resuelve con una corrida
> programada.

### El camino de inferencia

```
requirements.txt
    │
    ├─ parseo + lookup por pypi_package  →  filter_documents (NO retrieval)
    ├─ lógica de rangos PEP 440          →  advisories que afectan ESTA versión
    ├─ priorización                      →  KEV > EPSS > CVSS
    │
    ├─ CWE por id exacto                 →  filter_documents (NO retrieval)
    └─ prosa de INCIBE                   →  retrieval semántico (SÍ, acá corresponde)
    │
    └─→ prompt_builder → llm → tarjeta
```

**Punto clave: el CWE no se busca, se trae por clave.** Cuando el resolver termina ya se
tiene el string exacto `CWE-787`. Usar búsqueda semántica + full-text + RRF + cross-encoder
para encontrar un documento cuyo id exacto ya se conoce es usar una herramienta
probabilística para un problema determinístico — y el eval mide `recall_eff` 0.841, o sea
que fallaría ~1 de cada 6 veces. Con `filter_documents` es exacto o es un "no está en el
corpus" honesto.

La búsqueda semántica **sí** corresponde para la prosa de INCIBE: no hay identificador que
lleve a los párrafos sobre exposición de información, eso es genuinamente similitud.

> **Formato:** `meta.cwe_id` en el store guarda el número pelado (`787`), OSV entrega
> `CWE-787`. Hay que normalizar en el lookup.

## Contrato de salida (decisión importante)

**Python arma toda la tarjeta; el LLM escribe únicamente el párrafo explicativo.**

La versión de arreglo, el CVSS, el EPSS, el flag de KEV y todos los identificadores se
insertan por concatenación de strings desde los datos estructurados. **Nunca los genera el
modelo.**

Así el resultado es correcto por construcción: un LLM no puede equivocarse en un score que
nunca escribe. Es lo que hace la demo segura en vivo, y es coherente con el guardrail que
el `PROMPT_TEMPLATE` actual ya declara ("NEVER invent CVE/CWE identifiers, CVSS scores or
severities").

## Plan paso a paso

### Fase 1 — Datos (independientes entre sí, se pueden repartir)

1. **`src/ingestion/fetch_osv.py`** — baja
   `https://storage.googleapis.com/osv-vulnerabilities/PyPI/all.zip` a `data/raw/osv/`.
   Mismo patrón que [`fetch_nvd.py`](../src/ingestion/fetch_nvd.py) (checkpoint incluido;
   OSV publica `modified_id.csv` para incrementales). `data/raw` ya está bind-monteado como
   volumen `rawdata`.
2. **`src/ingestion/fetch_epss.py`** — `https://epss.empiricalsecurity.com/epss_scores-current.csv.gz`
   → `data/raw/epss/`. 2,5 MB, ~365.000 CVEs.
3. **`src/ingestion/fetch_kev.py`** — `https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json`
   → `data/raw/kev/`. 1,6 MB, 1.682 entradas.
4. **Catálogo completo de CWE.** Hoy hay 710 entradas (vistas parciales: top 25,
   cross-section, base elements). Bajar el XML completo de MITRE y pasarlo por el converter
   existente. **Medido: cierra un hueco real** — de los 64 CWEs que pide un escaneo de 10
   dependencias, 54 están y 10 faltan (91% de cobertura ponderada por hallazgo). Los que
   faltan son sobre todo CWEs de nivel abstracto (`CWE-664`, `CWE-669`, `CWE-670`,
   `CWE-697`) más específicos como `CWE-1321`.

### Fase 2 — Lógica pura (sin Docker, sin LLM, testeable) ← *camino crítico*

5. **`src/pipeline/deps/resolver.py`** — **ya prototipado y verificado.**
   - Normalización PEP 503 (`re.sub(r"[-_.]+", "-", name).lower()`).
   - Parseo del `requirements.txt`.
   - Algoritmo de rangos del esquema OSV: `versions[]` explícito, o recorrer `events`
     ordenados alternando estado en `introduced` / `fixed` / `last_affected`.
   - Comparación PEP 440 vía `packaging.version`.
   - *Pendiente*: extender más allá de `==` (`>=`, `~=`, extras, markers de entorno, `-r`).
6. **`src/pipeline/deps/prioritize.py`** — **ya prototipado y verificado.**
   - Base score CVSS 3.1 calculado desde el vector (OSV entrega vector, no número).
   - Join con EPSS y KEV por identificador de CVE.
   - Orden: KEV primero, después EPSS descendente, CVSS como desempate.
7. **Tests.** Ya existen fixtures en `test_requirements/`
   (`requirements_1vuln.txt`, `requirements_1de3vuln.txt`, `requirements_3vuln.txt`).
   Dado un requirements fijo, deben salir exactamente estas CVEs. **Corre en segundos, sin
   GPU, sin stack, sin LLM.** Es el único pedazo del sistema que se *verifica* en lugar de
   *medirse estadísticamente*.

### Fase 3 — Integración (necesita el stack; depende de la Fase 2)

8. **Converter de OSV** en [`indexing/converters.py`](../src/pipeline/indexing/converters.py)
   — advisory JSON → `Document` con la metadata de arriba. Id determinístico a partir del
   `osv_id`, igual que los CVE usan `sha256(cve_id)`. Enganchar en `run_indexing.py`.
9. **Lookup de CWE por id** — helper que trae el documento exacto con `filter_documents`,
   normalizando `CWE-787` ↔ `787`. Si no está: devolver vacío para que el sistema lo diga.
10. **`src/pipeline/pipeline_dependencias.py`** — clase `Pipeline` nueva. El servidor la
    levanta sola y aparece como segundo modelo en la API del `9099`, al lado del actual.
    Reusa por import `get_document_store()` y `build_generator()` de
    [`pipeline_ciberseguridad.py`](../src/pipeline/pipeline_ciberseguridad.py) — **ya son
    funciones módulo-level**, no hace falta refactor.
11. **Prompt template propio** — distinto al del chat: recibe los hallazgos ya priorizados
    y el documento de CWE, y pide **solo** el párrafo explicativo (ver
    [Contrato de salida](#contrato-de-salida-decisión-importante)).

### Fase 4 — Superficie

12. **Extensión VSCode** — comando sobre un `requirements.txt` que manda el contenido al
    `9099` seleccionando el modelo `pipeline_dependencias`, más un webview que renderiza la
    tarjeta. Se puede desarrollar contra respuestas falsas mientras la Fase 3 avanza.
13. **Panel de evidencia** — segunda superficie, con procedencia por retriever
    (`via_embedding` / `via_keyword`, columnas que el eval ya persiste), scores del reranker
    y resultados del eval.

### Reparto sugerido (4 personas)

| Persona | Trabajo |
|---|---|
| A | Fase 1 completa (pasos 1-4) |
| B | Fase 2 (pasos 5-7) ← **destraba la Fase 3** |
| C | Extensión (paso 12) contra respuestas falsas |
| D | Fase 3 (pasos 8-11), arranca cuando B termina |

Único cuello de botella real: el paso 10 no cierra hasta que 5 y 6 estén en el repo.

## Verificación

1. **Fase 2:** `pytest` sobre los fixtures de `test_requirements/`. Sin infra.
2. **Fase 3:** correr la indexación de OSV y verificar en el store
   `SELECT count(*) FROM ciberseguridad_docs WHERE meta->>'source_type' = 'osv_advisory'`
   (esperado: ~13.400).
3. **End-to-end:** mandar `requirements_3vuln.txt` al `9099` con el modelo nuevo y
   comparar la salida contra lo que da el resolver standalone. Los campos determinísticos
   deben coincidir **exactamente**.
4. **Regresión del chat:** correr `scripts/eval.sh` y confirmar delta ≈ 0. El pipeline de
   chat no se tocó; si el eval se mueve, algo se rompió.

## Limitaciones honestas

- **Alcanzabilidad.** EPSS y KEV dicen si esa CVE se explota *en el mundo*, no si el código
  propio llega a tocar la parte vulnerable de la librería. El análisis de alcanzabilidad es
  bastante más difícil y es lo que venden las herramientas comerciales. Fuera de alcance —
  conviene nombrarlo antes de que lo pregunten.
- **Solo PyPI.** OSV cubre npm, Go, crates.io y más, pero el alcance de esta etapa es
  Python.
- **Especificadores.** El resolver arranca soportando solo `==`. Un `requirements.txt` real
  trae `>=`, `~=`, extras y markers.
- **Nombres.** Aunque OSV usa nombres PyPI directamente (a diferencia de CPE), la
  normalización PEP 503 puede no cubrir todos los casos borde.
- **Cobertura de CWE.** 91% hasta que se ingiera el catálogo completo (paso 4).

## Datos de referencia (medidos el 2026-08-26)

| Fuente | Tamaño | Contenido | Actualización |
|---|---|---|---|
| OSV PyPI `all.zip` | 33 MB | 13.396 advisories + 11.676 `MAL` | continua |
| EPSS (FIRST) | 2,5 MB gz | 365.017 CVEs con probabilidad | diaria |
| CISA KEV | 1,6 MB | 1.682 CVEs explotadas confirmadas | continua |

Rendimiento del prototipo: índice OSV construido en **2,1 s**; escaneo de 10 dependencias
en **20 ms**. Todo determinístico, sin LLM ni embeddings.

> **Paquetes maliciosos.** Los 11.676 registros `MAL` son paquetes de PyPI que son
> directamente malware (típicamente typosquatting). Detectar uno en un `requirements.txt`
> es un caso de demo muy legible ("alguien tipeó mal el nombre de una librería y se instaló
> software malicioso") y un riesgo real. No es prioritario, pero está a mano.

## Relacionado

- [Arquitectura_RAG_Ciberseguridad.md](Arquitectura_RAG_Ciberseguridad.md) — el desacople
  indexación/inferencia que este plan respeta.
- [plan_de_trabajo.md](plan_de_trabajo.md) — "Mejora 3" (EPSS + KEV) y el desafío multi-hop
  que este trabajo materializa.
- [eval/eval_harness.md](eval/eval_harness.md) — el harness del chat; el camino de
  dependencias se verifica con tests unitarios, no con él.
- [ingestion_nvd_setup.md](ingestion_nvd_setup.md) — patrón de ingesta que siguen los
  fetchers nuevos.
- [reranker_cross_encoder.md](reranker_cross_encoder.md) — por qué el retrieval del chat es
  como es (y por qué el camino de dependencias no lo usa).