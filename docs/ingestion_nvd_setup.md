# Ingesta NVD/CVE — setup, estado y handoff

Fecha: 2026-07-01
Contexto: primer feed de la Capa 1 (datos estructurados) del corpus, según
`docs/plan_de_trabajo.md`. Este documento es el punto de entrada práctico:
qué se construyó, cómo configurarlo desde cero, qué falta, y por qué se
tomó cada decisión. Pensado para que lo siga cualquier persona del equipo
— o cualquier IA/agente — sin haber estado en la conversación original.

Documentos relacionados (leer si hace falta más contexto):
- `docs/plan_de_trabajo.md` — plan general del proyecto, las 3 capas de fuentes.
- `docs/data_transform_spec.md` — cómo transformar cada feed de Capa 1 una vez
  descargado (esquemas, joins, validaciones). Este documento (`ingestion_nvd_setup.md`)
  cubre el paso anterior: cómo *traer* el dato. Todavía no se implementó el paso de
  transformación/indexado descripto ahí para NVD (ver sección "Qué falta").
- `docs/data_sourcing_research.md` — investigación de las 5 fuentes de Capa 1
  (NVD, ATT&CK, CWE, EPSS, CISA KEV): endpoints, auth, rate limits, formato.
  Este documento solo implementa la parte de NVD; las otras 4 quedan pendientes.

---

## 1) Qué hay hecho

- `src/ingestion/fetch_nvd.py`: script que trae CVEs desde la API 2.0 de NVD y
  guarda el JSON crudo en `data/raw/nvd/`. **Esto es lo único que corre hoy.**
  No normaliza, no indexa, no toca Postgres/pgvector todavía.
- Entorno local (`.venv/`) preparado para correr scripts de ingestion.
- `infrastructure/.env.example` actualizado con la variable `NVD_API_KEY`.

## 2) Qué falta (a propósito, no es un olvido)

- **Indexado en el RAG**: falta el script (`src/ingestion/index_nvd.py`, todavía no
  escrito) que lea `data/raw/nvd/**/cves_page_*.json`, arme un `Document` de Haystack
  por CVE y lo embeba/guarde en la tabla pgvector que ya usa
  `src/pipeline/pipeline_ciberseguridad.py`. El diseño ya está acordado (ver sección 6)
  pero no implementado — si alguien retoma esto, ese es el próximo paso, no hay que
  rediscutir el diseño salvo que algo no cierre.
- **Las otras 4 fuentes de Capa 1** (ATT&CK, CWE, EPSS, CISA KEV): investigadas en
  `docs/data_sourcing_research.md`, sin script de fetch todavía.
- **Tabla SQL relacional `cves`** (la que describe `docs/data_transform_spec.md` con
  columnas normalizadas para hacer joins con EPSS/KEV): decidido explícitamente
  postergarla. Por ahora el objetivo es que el RAG pueda *responder preguntas* sobre
  CVEs vía retrieval, no hacer queries SQL de agregación. Si el proyecto llega a la
  fase de "priorización por explotabilidad" (cronograma, 30/Ago), ahí sí hace falta.

## 3) Incidente de seguridad — ya resuelto, pero hay que rotar la key

`infrastructure/.env` estaba trackeado en git desde antes de este trabajo (commit
`74b2191 "commit de envs"`). Al agregar `NVD_API_KEY` a ese archivo y commitear
(`0a20156 "Datos fetcheados de cves"`), la key quedó expuesta en texto plano en el
historial de git — y ese commit ya estaba pusheado a `origin/ciber-fetching` en GitHub
cuando se detectó.

Se corrigió hacia adelante con `git rm --cached infrastructure/.env` (commit `ad3369a`):
el archivo sigue existiendo en disco pero deja de trackearse. **Pendiente, no
automatizable sin acceso a NVD:** rotar la API key expuesta —
[nvd.nist.gov/developers/request-an-api-key](https://nvd.nist.gov/developers/request-an-api-key),
pedir una nueva y reemplazarla en el `infrastructure/.env` local de cada uno. La key
vieja sigue en el historial de git ya pusheado (no se reescribió el historial).

**Regla para todo lo que sigue: `infrastructure/.env` nunca se commitea.** Cada
persona tiene su propia copia local a partir de `infrastructure/.env.example`.

## 4) Setup desde cero (para un compañero nuevo)

Prerrequisitos: Python 3.10+, `git`, acceso al repo.

**a) Conseguir la API key de NVD (gratis):**
1. Ir a [nvd.nist.gov/developers/request-an-api-key](https://nvd.nist.gov/developers/request-an-api-key).
2. Completar el formulario (nombre, organización, email) y aceptar los Terms of Use.
3. Activar desde el link que llega por email (expira a los 7 días).
4. Cada email solo puede tener **una key activa**. No hace falta que todo el equipo
   pida una — alcanza con compartir una key por un canal seguro (no por chat/commit)
   si se prefiere una sola.

**b) Configurar el `.env`:**
```bash
cp infrastructure/.env.example infrastructure/.env
# Editar infrastructure/.env y completar NVD_API_KEY=<tu key>
# (además de DATABASE_USER/PASS, PGVECTOR_USR/PASS, PIPESERVER_KEY si vas a levantar la infra completa)
```
Este archivo **no se commitea** (ver sección 3).

**c) Crear el entorno virtual local:**
```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install requests python-dotenv haystack-ai pgvector-haystack ollama-haystack
```
Nota: `requirements.txt` en la raíz incluye además `marker-pdf` (y arrastra
`torch`/`transformers`, varios GB) — eso es para la conversión de PDFs dentro del
contenedor `pipelines` (ver `infrastructure/Dockerfile.pipelines`), **no hace falta
para trabajar con la ingesta de CVEs**. Si en algún momento se necesita correr/debuggear
esa parte en local, instalar `requirements.txt` completo:
```bash
.venv/bin/pip install -r requirements.txt
```

**d) Verificar que el entorno quedó bien armado:**
```bash
.venv/bin/python -c "
import requests, dotenv, haystack
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
print('OK')
"
```

## 5) Cómo correr el fetch de CVEs

Desde la raíz del repo, con el venv activado (o usando `.venv/bin/python` directo):

```bash
# Primera vez: trae el catálogo completo (~362k CVEs a la fecha de este doc)
.venv/bin/python src/ingestion/fetch_nvd.py --full

# Corridas siguientes: solo trae lo nuevo/modificado desde la última corrida
.venv/bin/python src/ingestion/fetch_nvd.py
```

**Qué hace exactamente** (`src/ingestion/fetch_nvd.py`):
- Pega contra `https://services.nvd.nist.gov/rest/json/cves/2.0` con la
  `NVD_API_KEY` del `.env` (header `apiKey`).
- Pagina con `resultsPerPage=2000` + `startIndex`, con `time.sleep(0.7)` entre
  requests para quedar por debajo del límite de 50 req/30s que da la API key
  (sin key el límite es 5 req/30s, por eso hace falta la key).
- Reintenta hasta 3 veces con backoff exponencial si una request falla
  (status ≠ 200).
- **Modo incremental**: guarda un checkpoint (`data/raw/nvd/_checkpoint.json`, un
  timestamp UTC) al terminar cada corrida exitosa. La siguiente corrida sin `--full`
  usa ese timestamp como `lastModStartDate` y pide solo los CVEs modificados desde
  entonces. Si el gap desde el último checkpoint supera los 120 días (límite que
  impone la API de NVD para ese filtro), el script lo parte solo en ventanas ≤120
  días — no hace falta intervención manual.
- Cada corrida escribe en una carpeta nueva con timestamp:
  `data/raw/nvd/<YYYYMMDDTHHMMSSZ>/cves_page_0000.json`, `cves_page_0001.json`, ...
  — el JSON de cada página es la respuesta cruda de la API, sin modificar.
- No borra corridas anteriores. `data/raw/nvd/` está en `.gitignore` (los dumps son
  grandes y se regeneran corriendo el script, no tiene sentido versionarlos).

**Estado al momento de escribir este documento:** hay una corrida `--full` corriendo
en background (PID visible con `ps aux | grep fetch_nvd`), fetcheando el catálogo
completo. Progreso aproximado: ~20-30s por página, 182 páginas totales → ~70-80 min.
Si se cortó o no terminó, simplemente correr de nuevo `--full` (no importa si ya hay
carpetas parciales de una corrida anterior, cada corrida es independiente).

## 6) Diseño acordado para el paso de indexado (no implementado)

Cuando se retome, el próximo script (`src/ingestion/index_nvd.py`, a crear) debe:

1. Ser standalone, sin tocar `src/pipeline/pipeline_ciberseguridad.py` (ese archivo
   ya sirve el RAG en producción vía OpenWebUI Pipelines; menor riesgo si el indexado
   estructurado vive aparte).
2. Leer todos los `data/raw/nvd/**/cves_page_*.json`.
3. Crear **un `Document` de Haystack por CVE — sin chunkear**. Una entrada CVE es
   atómica (regla general de `docs/data_transform_spec.md` para toda la Capa 1):
   nada de `DocumentSplitter` acá, eso es solo para los PDFs largos de Capa 2.
4. Separar contenido de metadata:
   - `content` (lo único que se embebe, con `OllamaDocumentEmbedder` + modelo
     `bge-m3`, igual que ya usa el pipeline existente) = la `description` del CVE en
     texto plano.
   - `meta` (filtrable vía pgvector, no se vectoriza) = `cve_id`, `cvss_v3_score`,
     `cvss_v3_vector`, `published_date`, `last_modified`, `vendor`/`products`
     (parseados de las entradas CPE), `cwe_ids` (de `weaknesses`), `references`, y
     `source_type: "nvd_cve"` para poder distinguir CVEs de PDFs en el índice.
5. Usar un `id` de documento determinístico (hash de `cve_id`, no aleatorio), para
   que reindexar solo actualice los CVEs cuyo `last_modified` cambió, en vez de
   duplicar todo en cada corrida.
6. Escribir en la **misma tabla pgvector** que ya usa el pipeline
   (`ciberseguridad_docs`, ver `DB_TABLE` en `pipeline_ciberseguridad.py`), no una
   tabla nueva — así `embedding_retriever`/`keyword_retriever` recuperan CVEs y PDFs
   juntos en la misma búsqueda híbrida, y se puede filtrar por `source_type` cuando
   haga falta (por ejemplo para las queries temporales del plan de trabajo, que
   necesitan `published_date`).

## 7) Mapa de archivos tocados en este trabajo

| Archivo | Qué cambió |
|---|---|
| `src/ingestion/fetch_nvd.py` | nuevo — fetch de CVEs desde NVD (ver sección 5) |
| `docs/data_sourcing_research.md` | nuevo — investigación de las 5 fuentes Capa 1 |
| `docs/ingestion_nvd_setup.md` | nuevo — este documento |
| `requirements.txt` | + `requests`, `python-dotenv` |
| `.gitignore` | + `data/raw/nvd/` (dumps grandes, regenerables) |
| `infrastructure/.env.example` | + `NVD_API_KEY=your_nvd_api_key` |
| `infrastructure/.env` | **desengachado de git** (`git rm --cached`), sigue existiendo local con la key real (a rotar, ver sección 3) |
| `.venv/` | nuevo, local, no versionado — ver sección 4c para recrearlo |

## 8) Comandos de referencia rápida

```bash
# Setup (una vez)
cp infrastructure/.env.example infrastructure/.env   # completar NVD_API_KEY
python3 -m venv .venv
.venv/bin/pip install requests python-dotenv haystack-ai pgvector-haystack ollama-haystack

# Fetch inicial completo
.venv/bin/python src/ingestion/fetch_nvd.py --full

# Fetch incremental (correr periódicamente, ej. cron diario — ver docs/data_sourcing_research.md
# sección "Consideraciones transversales" para la justificación de la cadencia)
.venv/bin/python src/ingestion/fetch_nvd.py

# Ver progreso de una corrida en curso
ps aux | grep fetch_nvd
find data/raw/nvd -name "cves_page_*.json" | wc -l
```
