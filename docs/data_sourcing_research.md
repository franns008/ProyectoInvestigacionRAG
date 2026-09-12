# Investigación — Cómo traer los datos (Capa 1: fuentes estructuradas)

Fecha: 2026-07-01
Autor: investigación asistida (Claude Code)

Resumen
-------
Este documento complementa `docs/data_transform_spec.md` (que asume que el dato ya fue
descargado) respondiendo la pregunta anterior en la cadena: **¿cómo se obtiene cada feed
de la Capa 1** (NVD/CVE, MITRE ATT&CK, CWE, EPSS, CISA KEV)? Para cada fuente se detalla
método de acceso, endpoint/URL, autenticación, límites de uso, formato, estrategia de
actualización incremental y librerías recomendadas. Exploit-DB (Capa 3) queda fuera de
este documento porque no es una fuente "estructurada" en el mismo sentido (mezcla código
y prosa) — se investigará por separado si hace falta.

> **Estado (2026-09-11):** los fetchers de NVD, CWE, EPSS, KEV y OSV ya están
> implementados — ver [`ingestion_fetchers.md`](ingestion_fetchers.md). Queda pendiente
> sólo ATT&CK. Lo de abajo es la investigación que los fundamenta.

Nota sobre lo ya presente en el repo: los XML en `data/raw/`
(`CrossSection.xml`, `CWEtop25.xml`, `OWASPTopTenRC2025.xml`, `WeaknessBaseElements.xml`,
`WrittenInJava.xml`) son **vistas parciales de CWE** (CWE-884, CWE-1435, CWE-1450, CWE-677,
CWE-660) descargadas manualmente, no el catálogo completo: cubren 710 de las 969
weaknesses. Sirven para explorar la estructura del XML pero no reemplazan la descarga
programática del catálogo completo (ver sección CWE). Quedaron redundantes una vez que
`fetch_cwe.py` trae el catálogo entero; no molestan porque la indexación deduplica por
CWE-ID, pero se pueden podar.

---

1) NVD / CVE
------------
**Método de acceso:** API REST (no hace falta descargar el dump completo salvo para carga
inicial masiva).

- Endpoint base: `https://services.nvd.nist.gov/rest/json/cves/2.0`
- Auth: opcional pero muy recomendable. Se pide una API key gratuita en
  [nvd.nist.gov/developers/request-an-api-key](https://nvd.nist.gov/developers/request-an-api-key)
  (activación por email, expira si no se activa en 7 días).
- **Rate limits:** 5 requests / 30s sin key, **50 requests / 30s con key**. Con el volumen
  de ~250k CVEs, sin key la carga inicial es inviable en tiempo razonable → conseguir la
  key es el primer paso práctico.
- Paginación: `startIndex` + `resultsPerPage` (máx. 2000 por página).
- Filtros relevantes: `lastModStartDate` / `lastModEndDate` (rango máx. 120 días, formato
  ISO-8601), `pubStartDate`/`pubEndDate`, `cveId`, `cveIds` (hasta 100 separados por coma),
  `keywordSearch`, `cpeName`, `cvssV3Severity`, `cweId`.
- **Actualización incremental:** guardar `lastModEndDate` del último fetch como checkpoint
  y en la próxima corrida pedir `lastModStartDate=<checkpoint>`. Encaja directamente con
  `ingestion_checkpoint` propuesto en `data_transform_spec.md`.
- Formato: JSON (CVE JSON 2.0), coincide con el ejemplo ya documentado en
  `data_transform_spec.md`.
- Librerías Python: `requests` alcanza; hay wrappers de comunidad (`nvdlib`) que ya manejan
  paginación y backoff si se quiere ahorrar boilerplate.

2) MITRE ATT&CK
---------------
**Método de acceso:** descarga estática de archivos JSON (no hay API REST, es un repo
GitHub versionado como colecciones STIX 2.1).

- Repo: [github.com/mitre-attack/attack-stix-data](https://github.com/mitre-attack/attack-stix-data)
- Tres dominios como bundles separados: `enterprise-attack/`, `mobile-attack/`, `ics-attack/`.
  El plan de trabajo pide "MITRE ATT&CK (JSON v15)" — hoy el repo va por releases más
  recientes (v19.x); conviene fijar una versión concreta (`enterprise-attack-<version>.json`)
  para reproducibilidad, no el archivo "latest" que cambia sin aviso.
- Auth: ninguna, es público.
- **Actualización:** el repo publica un `index.json` legible por máquina para detectar
  nuevos releases sin tener que hacer polling pesado.
- Formato: STIX 2.1 (JSON). Objetos relevantes: `attack-pattern` (técnicas),
  `course-of-action` (mitigaciones), `intrusion-set` (grupos/actores), `relationship`
  (edges entre todo lo anterior) — esto es lo que permite responder las preguntas
  "relacionales" del plan de trabajo (ej. "¿qué técnicas usa APT29?").
- Librería recomendada: **`mitreattack-python`** (wrapper oficial sobre `stix2`, expone
  helpers tipo `get_techniques()`, `get_groups()`, `get_relationships()` sin tener que
  navegar el grafo STIX a mano). Alternativa más low-level: `cti-python-stix2`.

3) CWE
------
**Método de acceso:** dos opciones válidas, conviene usar ambas para cosas distintas.

- **Descarga completa (para carga inicial / bulk):**
  `https://cwe.mitre.org/data/xml/cwec_latest.xml.zip` (o versión fija, ej.
  `cwec_v4.20.xml.zip`, en `https://cwe.mitre.org/data/downloads.html`). Esto es el
  catálogo completo — no confundir con las "views" (como las 5 que ya están en
  `data/raw/`), que son subconjuntos temáticos (Top 25, OWASP Top
  Ten, etc.) pensados para consulta puntual, no para poblar la base completa.
  El zip trae un único XML con nombre versionado (`cwec_v4.20.xml`), 2 MB comprimido y
  18 MB crudo. **Implementado en `src/ingestion/fetch_cwe.py`.**
- **REST API (para consultas puntuales / relaciones jerárquicas):**
  root en `https://cwe-api.mitre.org/api/v1/`, sin auth. Expone `cwe/weakness/{id}`,
  `cwe/category/{id}`, `cwe/view/{id}`, y los endpoints `children` / `parents` /
  `descendants` / `ancestors` para navegar la jerarquía padre-hijo que pide
  `data_transform_spec.md` (relaciones `parent_ids[]`). Devuelve `[]` con 200 si no hay
  relaciones, y 404 si el ID no existe.
  **No sirve para bulk:** el endpoint no documentado `cwe/weakness/all` responde 200 con
  el JSON truncado (corta a los ~10 MB), así que la carga masiva va sí o sí por el zip.
- Formato: XML (descarga completa) o JSON (API). Ambos representan el mismo esquema
  (`cwe_schema_v7.3.xsd`), por lo que el parser XML que se escriba para el bulk sirve como
  referencia de campos para el JSON de la API.
- **Actualización:** CWE no tiene cadencia diaria — se libera por versiones (hoy 4.20).
  Alcanza con re-descargar el catálogo completo en cada release nueva en vez de manejar
  incrementalidad fina como NVD.

4) EPSS
-------
**Método de acceso:** API REST (ideal para joins puntuales con `cve_id`) + snapshot CSV
diario (ideal para carga masiva/batch).

- API: `https://api.first.org/data/v1/epss`. Sin autenticación.
  Parámetros: `cve` (uno o lista separada por coma), `date` (histórico, `yyyy-mm-dd`),
  `scope=time-series` (últimos 30 días de un CVE puntual), `epss-gt`/`percentile-gt`
  (filtros por umbral — útil para la query de priorización del plan de trabajo), `order`,
  `limit`/`offset`, `envelope`/`pretty`.
- CSV snapshot diario: `epss_scores-current.csv.gz` (link en la página de datos de FIRST),
  más práctico que paginar la API cuando se quiere el feed completo del día.
- Histórico completo (desde 2021-04-14) disponible como repo GitHub separado
  (`empiricalsec/epss_scores`) si se necesita backfill para evaluación.
- **Actualización:** el feed se recalcula **diariamente** → encaja con un cron diario que
  descarga el CSV del día y hace upsert contra `cves.epss_score`, tal como ya describe
  `data_transform_spec.md`.

5) CISA KEV
-----------
**Método de acceso:** archivo estático JSON/CSV, sin API con parámetros de query.

- JSON: `https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json`
- CSV equivalente en la misma carpeta `feeds/`.
- Mirror en GitHub (`cisagov/kev-data`) sincronizado a los pocos minutos del canónico —
  útil como respaldo si el sitio de CISA da 403/rate-limit a scraping automatizado (me
  pasó al intentar traer la página HTML directamente durante esta investigación; el JSON
  del feed no tuvo ese problema).
- Auth: ninguna.
- Campos clave ya alineados con lo que pide `data_transform_spec.md`: `cveID`, `dateAdded`,
  `vendorProject`, `product`, `shortDescription`, `requiredAction`, `dueDate`,
  `knownRansomwareCampaignUse`, `notes`, y también trae `cwes` (mapping directo a CWE, un
  join gratis que el spec no tenía documentado).
- **Actualización:** CISA agrega entradas en días hábiles cuando corresponde (no es un
  cron fijo tipo "todos los días a las X"); alcanza con polling diario y comparar
  `catalogVersion`/`dateReleased` contra el último snapshot guardado para detectar cambios.

6) OSV
------
**Método de acceso:** dump estático por ecosistema (no estaba en el alcance original de
esta investigación; entró después con el escaneo de dependencias).

- Dump de PyPI: `https://storage.googleapis.com/osv-vulnerabilities/PyPI/all.zip`
  (34 MB, un JSON por advisory adentro). La lista de ecosistemas está en
  `https://osv-vulnerabilities.storage.googleapis.com/ecosystems.txt`.
- Auth: ninguna.
- **Actualización:** diaria. OSV publica un `modified_id.csv` que permitiría bajar sólo lo
  que cambió; hoy se rebaja el zip entero porque tarda segundos.
- Es la única fuente **obligatoria** del escaneo de dependencias: aporta los rangos de
  versión afectados, que es lo que NVD no da de forma utilizable por paquete.

---

Tabla comparativa
------------------

| Fuente | Acceso | Auth | Límite | Formato | Cadencia update |
|---|---|---|---|---|---|
| NVD/CVE | API REST | API key (recomendada) | 5/30s sin key, 50/30s con key | JSON | continua (usar `lastModStartDate`) |
| ATT&CK | Archivo estático (GitHub) | No | N/A | JSON (STIX 2.1) | por release (chequear `index.json`) |
| CWE | Descarga XML completa + REST API | No | No documentado | XML / JSON | por versión (hoy 4.20) |
| EPSS | API REST + CSV diario | No | No documentado | JSON / CSV | diaria |
| CISA KEV | Archivo estático JSON/CSV | No | N/A | JSON / CSV | días hábiles, sin cadencia fija |
| OSV | Dump estático por ecosistema | No | N/A | ZIP de JSONs | diaria |

Consideraciones transversales
------------------------------
- **Licencias:** las cinco fuentes son de acceso público y gratuito (NVD y CISA KEV son
  producidos por organismos de EE.UU.; ATT&CK y CWE por MITRE con términos de uso propios
  que en general piden atribución). Antes de redistribuir texto textual en respuestas del
  RAG conviene revisar el texto de términos de uso puntual de cada sitio — no se encontró
  el texto legal exacto en esta pasada de investigación, queda como pendiente si el
  proyecto necesita certeza jurídica.
- **Ninguna de las 5 fuentes requiere pago.** El único fricción real es la API key de NVD
  (gratis pero con proceso de activación por email) — conviene pedirla ahora dado que sin
  ella la carga inicial de ~250k CVEs es impracticable (5 req/30s).
- **Orden de implementación sugerido** (por dependencia de joins, ya anticipado en
  `data_transform_spec.md`): NVD primero (tabla `cves` base) → EPSS y CISA KEV (upsert por
  `cve_id`) → CWE y ATT&CK (catálogos independientes, se linkean después vía `cwes`/
  referencias cruzadas).
- **Todas encajan con el patrón "documento atómico, no chunkear"** que ya define
  `data_transform_spec.md`: cada CVE, técnica ATT&CK o CWE debe indexarse como una unidad
  con metadata filtrable, no partirse en fragmentos de tamaño fijo.

Próximos pasos
---------------
Hecho:
- Pedir la API key de NVD.
- Los parsers/fetchers viven en `src/ingestion/`, un archivo por fuente
  (`fetch_<fuente>.py`) más el orquestador `fetch_all.py`. Sólo descargan y guardan el
  raw, sin tocar la transformación — ver [`ingestion_fetchers.md`](ingestion_fetchers.md).

Pendiente:
- El fetcher de ATT&CK (es el único que falta; todavía no lo consume nada en el repo).
- Decidir el mecanismo de scheduling (cron simple vs. Airflow/Prefect) — el plan de
  trabajo no lo fija todavía.
- Incrementales de OSV vía `modified_id.csv`.

Fuentes consultadas
--------------------
- [NVD Vulnerabilities API docs](https://nvd.nist.gov/developers/vulnerabilities)
- [NVD API — Start Here / rate limits](https://nvd.nist.gov/developers/start-here)
- [attack-stix-data (GitHub)](https://github.com/mitre-attack/attack-stix-data)
- [CWE Downloads](https://cwe.mitre.org/data/downloads.html)
- [CWE REST API — Quick Start](https://github.com/CWE-CAPEC/REST-API-wg/blob/main/Quick%20Start.md)
- [EPSS data & stats](https://www.first.org/epss/data_stats)
- [EPSS API docs](https://www.first.org/epss/api)
- [CISA KEV catalog](https://www.cisa.gov/known-exploited-vulnerabilities-catalog)
- [cisagov/kev-data mirror (GitHub)](https://github.com/cisagov/kev-data)
