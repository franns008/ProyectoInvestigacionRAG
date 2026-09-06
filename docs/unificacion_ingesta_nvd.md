# Unificación de la ingesta de NVD

**Estado:** propuesto, sin aplicar · **Fecha:** 2026-09-06 · **Rama base:** `main` (`015bb2f`)

## Resumen

Hay **dos caminos de escritura distintos hacia la misma tabla de producción**
(`ciberseguridad_docs`), que leen los mismos archivos crudos y producen documentos con
distinto `content` y distinto esquema de metadata para el mismo CVE:

- `src/pipeline/indexing/run_indexing.py` + `converters.py` — el camino vigente.
- `src/ingestion/index_nvd.py` — un prototipo de julio que un refactor de agosto dejó
  huérfano y nadie volvió a tocar.

La propuesta es **eliminar `index_nvd.py`** y dejar `run_indexing.py --include-cve` como
camino único, previo diagnóstico de la base para saber si quedaron filas escritas por el
camino viejo.

## El problema

### Dos flujos, punta a punta

| | `run_indexing.py` (vigente) | `index_nvd.py` (huérfano) |
|---|---|---|
| **Origen** | `/app/pipelines/rawdata` → volumen `rawdata` → `data/raw/`, con `rglob` | `data/raw/nvd/`, con `glob("**/")` |
| **Contexto** | dentro del container `pipelines` | en el host |
| **DB** | `vdb:5432` (red docker) | `localhost:${VDB_PORT}` |
| **Ollama** | `http://ollama:11434` | `http://localhost:${OLLAMA_PORT}` |
| **Modelo / dimensión** | `qwen3-embedding:4b` / **2560** | `bge-m3` / **1024** |
| **Alcance** | PDF, DOCX, MD, TXT, XML (CWE), JSON (CVE) | sólo JSON (CVE) |
| **Dedup contra la BD** | filtra ids existentes **antes** de escribir | `OVERWRITE` incondicional |
| **Dedup entre páginas** | gana el `lastModified` más reciente | gana **el primero visto** |
| **Embedding** | `ThreadPoolExecutor`, 2 en paralelo, lotes de 64, try/except por lote | secuencial, lotes de 200, sin manejo de errores |

**Leen los mismos archivos.** El volumen `rawdata` mapea a `../data/raw/`
([`docker-compose.yml:119-124`](../infrastructure/docker-compose.yml#L119-L124)) y
`run_indexing` usa `rglob`, así que alcanza los `data/raw/nvd/<timestamp>/cves_page_*.json`
que produce `fetch_nvd.py`. No son dos fuentes de datos, son dos lectores de la misma.

### Cronología: cuál es el canónico

```
2026-07-02  index_nvd.py     ← un único commit, nunca más tocado
2026-08-04  converters.py    ← extracción de la etapa de indexing
2026-08-12  "new embbeder"   ← migración a qwen3-embedding / 2560
2026-09-06  converters.py    ← fix del cve id en el content (015bb2f)
```

`index_nvd.py` tiene **un solo commit** y es anterior tanto a la extracción del módulo
`indexing/` como a la migración de embeddings; `converters.py` lleva cinco. Además
[`docs/ingestion_nvd_setup.md:32,149`](ingestion_nvd_setup.md) todavía describe a
`index_nvd.py` como *"a crear"* y le pide usar `bge-m3` *"igual que ya usa el pipeline
existente"*: se escribió contra una especificación que la migración de agosto volvió
obsoleta.

### Lógica duplicada, pero divergente

Seis unidades de código con el mismo propósito, ~100 de las 240 líneas de `index_nvd.py`.
No son copias: son forks que **producen datos distintos**, que es peor.

| Propósito | `converters.py` | `index_nvd.py` | Diferencia |
|---|---|---|---|
| Descripción EN | `_english_description:110` | `_pick_description:77` | `.get("value","")` vs `d["value"]` → **KeyError** si falta |
| CWE ids | `_cwe_ids:171` | `_cwe_ids:102` | mismo resultado, distinto camino |
| CVSS | `_cvss:119` | `_pick_cvss:85` | **datos distintos** (ver abajo) |
| Vendors/products | `_vendors_products:149` | `_vendors_products:112` | fuente primaria distinta |
| Referencias | inline en meta `:201` | `_references:138` | index_nvd corta en 10 |
| CVE → Document | `cve_to_document:180` | `cve_to_document:150` | mismo `id`, distinto `content` y meta |

**CVSS.** No cambian sólo los nombres de las claves, cambia la semántica. `converters._cvss`
guarda v3 y v2 en campos separados más `severity`, prioriza v3.1 > v3.0 y **no contempla
v4.0**. `index_nvd._pick_cvss` colapsa todo en `cvss_score`/`cvss_vector`/`cvss_version`,
con prioridad v3.1 > v4.0 > v3.0 > v2, y **nunca escribe `severity`**.

**Vendors/products.** `index_nvd` lee primero `affected[].affectedData[]`, que **no existe en
el esquema de la API 2.0 de NVD** (es la forma de OSV); recién si viene vacío cae al parseo
de CPE que `converters` hace directamente.

**Dedup entre corridas, invertido.** `_iter_cve_records:179` ordena los archivos por nombre
—o sea, cronológicamente ascendente— y descarta el CVE ya visto: **gana la versión más
vieja**. `NVDJsonConverter` compara `lastModified` y se queda con la más reciente. En un
re-fetch incremental, que es el caso de uso de `fetch_nvd.py`, `index_nvd` conserva el dato
viejo y descarta la actualización.

### El defecto estructural

Ambos calculan `id = sha256(cve_id)`, así que **escriben sobre el mismo documento**. Pero la
interacción no es simétrica:

- [`index_nvd.py:210`](../src/ingestion/index_nvd.py#L210) escribe con `OVERWRITE` para cada
  CVE que parsea, sin condición.
- [`run_indexing.py:203-207`](../src/pipeline/indexing/run_indexing.py#L203-L207) **filtra**
  los documentos atómicos cuyo id ya existe antes de escribir. Aunque en la línea 246 use
  `OVERWRITE`, nunca llega: esos CVE ya salieron de `new_docs`.

**El camino de producción no puede reparar lo que escribió el viejo.** Reindexar con
`--include-cve` no los toca.

Consecuencia concreta y actual: los documentos escritos por `index_nvd` **no tienen
`meta.source`**, y el componente `VulnIdLookup` del canal determinístico de IDs filtra
justamente por `meta.source`. Un CVE escrito por el camino viejo es invisible para ese canal.

### Riesgo real, acotado

Conviene no exagerarlo. `PgvectorDocumentStore(embedding_dimension=1024)` hace
`CREATE TABLE IF NOT EXISTS`, que sobre una tabla existente es un no-op sin validación, pero
después intenta escribir vectores de 1024 en una columna `vector(2560)` y **Postgres rechaza
la escritura**. Sobre la base actual `index_nvd.py` no la envenena: se cae.

El riesgo tiene dos caras reales:

1. **Contaminación histórica** — si corrió en julio, cuando la tabla era de 1024, esos CVE
   quedaron con su esquema. Si la migración de agosto recreó la tabla, se limpió sola. Hay
   que mirar la base para saberlo (Fase 0).
2. **Base nueva** — en un entorno limpio, ejecutarlo antes que `run_indexing` crea la columna
   en `vector(1024)` y rompe producción de entrada.

## Evidencia medida

Sobre los 8000 CVE crudos de `data/raw/nvd/`, parseados con `NVDJsonConverter`:

| Métrica | Valor |
|---|---|
| Documentos generados | 8000 |
| Con `meta.source` | 8000 |
| Con el id en el `content` | 8000 |
| Con `cvss_v2_score` | 7927 (99,1%) |
| Con `severity` | 7927 (99,1%) |
| Con `cvss_v3_score` | **151 (1,9%)** |
| Con CVSS v4.0 | **0** |

Dos conclusiones que reordenan prioridades:

- **El soporte de v4.0 de `index_nvd` no tiene impacto medible.** Era el único punto donde el
  archivo viejo era mejor; con cero casos, deja de ser argumento para conservarlo.
- **El `PROMPT_TEMPLATE` sólo lee `cvss_v3_score`**, así que hoy muestra la nota CVSS para el
  1,9% de los CVE, teniendo el score v2 en la metadata del 99,1%. El `severity` sí llega.

## Solución propuesta

### Fase 0 — Diagnóstico (no modifica nada)

Determina si hace falta la Fase 4. **Debe correr antes de cualquier cambio de esquema**,
porque discrimina por claves de metadata que un cambio posterior enturbiaría.

```sql
-- A) total de CVE indexados
SELECT count(*) FROM ciberseguridad_docs WHERE meta->>'source_type'='nvd_cve';

-- B) escritos por index_nvd: no tienen meta.source
SELECT count(*) FROM ciberseguridad_docs
WHERE meta->>'source_type'='nvd_cve' AND NOT (meta ? 'source');

-- C) sin el id en el content (anteriores a 015bb2f)
SELECT count(*) FROM ciberseguridad_docs
WHERE meta->>'source_type'='nvd_cve' AND content NOT LIKE 'Vulnerabilidad CVE-%';

-- D) dimensión real de la columna de embeddings
SELECT atttypmod FROM pg_attribute
WHERE attrelid='ciberseguridad_docs'::regclass AND attname='embedding';
```

Si **B = 0 y C = 0**, no hay contaminación: saltear la Fase 4.

### Fase 1 — Rama

```bash
git checkout -b fix/unificar-ingesta-nvd
```

### Fase 2 — Eliminar el camino duplicado

```bash
git rm src/ingestion/index_nvd.py
```

Actualizar las tres referencias que quedan colgadas:

- [`ingestion_nvd_setup.md:32,149`](ingestion_nvd_setup.md) — describen el script como
  *"a crear"* y le piden `bge-m3`. Reemplazar por: la ingesta de CVE la hace
  `run_indexing.py --include-cve` con `qwen3-embedding:4b`.
- [`escaneo_dependencias.md:71`](escaneo_dependencias.md) — el argumento sigue siendo válido
  (NVD descarta los rangos de versión), pero hay que reapuntarlo a
  `converters._vendors_products:149`, que tiene el mismo comportamiento.

Verificación:

```bash
grep -rn 'index_nvd' --include='*.py' --include='*.md' . | grep -v '^./.git/'   # vacío
python3 -m pytest -q
```

### Fase 3 — Cerrar la brecha de CVSS

**3a. CVSS v2 en el prompt** (afecta al 99,1% de los CVE). En `PROMPT_TEMPLATE`:

```jinja
{% if document.meta.cvss_v3_score %} CVSS3={{ document.meta.cvss_v3_score }}
{% elif document.meta.cvss_v2_score %} CVSS2={{ document.meta.cvss_v2_score }}{% endif %}
```

Etiquetar la versión no es decorativo: un 10.0 en v2 no equivale a un 10.0 en v3, y el prompt
ya prohíbe inventar scores. Sin la etiqueta, el modelo presentaría una nota v2 como si fuera
CVSS actual.

**3b. CVSS v4.0** — opcional y defensivo. Cero casos hoy, pero NVD lo va a ir poblando. Si se
hace, agregar `cvss_v4_score`/`cvss_v4_vector` a `_cvss` y una rama al template. **No es
bloqueante.**

### Fase 4 — Reparar filas contaminadas *(sólo si B o C > 0)*

No alcanza con reindexar: `run_indexing.py:203-207` filtra los ids existentes antes de
escribir, así que saltea esos CVE y nunca los repara. Hay que borrarlos para que los vuelva
a crear.

```sql
BEGIN;
SELECT count(*) FROM ciberseguridad_docs
WHERE meta->>'source_type'='nvd_cve'
  AND (NOT (meta ? 'source') OR content NOT LIKE 'Vulnerabilidad CVE-%');
-- si el número coincide con B+C, confirmar el DELETE con el mismo WHERE
COMMIT;
```

Luego, dentro del container:

```bash
docker compose -f infrastructure/docker-compose.yml exec pipelines \
  python /app/pipelines/indexing/run_indexing.py --include-cve
```

### Fase 5 — Verificación

Local, sin stack (el parseo es lógica pura y hay datos crudos reales en el repo):

```bash
python3 -c "
import sys; sys.path.insert(0,'src/pipeline/indexing')
from pathlib import Path; from converters import NVDJsonConverter
docs = NVDJsonConverter().run(sorted(Path('data/raw/nvd').rglob('cves_page_*.json')))['documents']
assert all(d.meta.get('source') for d in docs)
assert all(d.content.startswith('Vulnerabilidad CVE-') for d in docs)
print(len(docs), 'docs OK')
"
```

Contra la base, repetir B y C de la Fase 0: ambas deben dar 0.

## Fuera de alcance

- **Modo host para `run_indexing`.** Es la única capacidad genuina que se pierde al borrar
  `index_nvd.py` (correr sin container). Si se necesita, la forma correcta es parametrizar
  las tres constantes de conexión, no sostener una segunda implementación del parseo de NVD.
- **Índice sobre `(meta->>'source')`** para que el filtro de `VulnIdLookup` no haga un scan
  de JSONB sobre el corpus grande.
- **Cortocircuito de los meta-prompts de OpenWebUI**, que hoy corren el pipeline completo.
