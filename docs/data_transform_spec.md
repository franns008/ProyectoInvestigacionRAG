# Especificación de Transformaciones — Fuentes Estructuradas

Fecha: 2026-06-12
Autor: Equipo Niños RAGtas (generado automáticamente)

Resumen
-------
Documento técnico que describe las fuentes estructuradas usadas por el RAG (según plan_de_trabajo.md: NVD, MITRE ATT&CK, CWE, EPSS, CISA KEV, Exploit-DB) y las transformaciones necesarias para convertir cada feed en artefactos listos para indexar y/o unir entre sí. Incluye: formato de entrada, campos clave, transformaciones paso a paso, ejemplos de salida (esquema SQL/JSON), validaciones y recomendaciones de monitorización.

Objetivo
--------
Entregar una especificación operativa que permita implementar pipelines de ingesta y transformación reproducibles, incrementales y auditables sin tocar código existente del proyecto. Basado en las fuentes explícitamente mencionadas en docs/plan_de_trabajo.md.

Fuentes cubiertas
-----------------
CAPA 1 — Datos estructurados / semi-estructurados:
- NVD (NIST) — CVE JSON feeds
- MITRE ATT&CK — STIX/JSON
- CWE (MITRE) — XML/HTML
- EPSS (FIRST) — CSV/JSON feed
- CISA KEV (Known Exploited Vulnerabilities) — CSV/JSON

CAPA 3 — Fuentes técnicas específicas:
- Exploit-DB — metadatos CSV + textos/code

Nota: CAPA 2 (documentos no estructurados como NIST SP, CISA Advisories, threat reports) está fuera de este documento. Eso requiere procesamiento de texto no estructurado (PDFs, OCR, etc.).

1) NVD (NIST)
-------------
Qué brinda
- Registro de CVEs con: CVE ID, descripción, referencias, configuraciones CPE, CVSS v2/v3 vector strings y score, fecha publicada/modificada, productos afectados.
Formato
- JSON (CVE JSON v1/v2) + feeds incrementales
Necesita transformación? Sí — leve

Transformaciones recomendadas
- Almacenar el JSON RAW en blob (raw_cves) con ingestion_ts y source_version.
- Normalizar campos principales a una tabla `cves`:
  - cve_id (PK)
  - description (text)
  - published_date (timestamp ISO)
  - last_modified (timestamp)
  - cvss_v2_vector, cvss_v2_score
  - cvss_v3_vector, cvss_v3_score
  - cpe_entries (array JSON normalizado)
  - references (array JSON)
  - products_affected (array de objetos {vendor,product,version_range})
  - raw_json (jsonb)
- Parsear CPEs: usar librería de parsing CPE (pypi: cpe) para extraer vendor/product/version y normalizar.
- Normalizar fechas a UTC ISO8601 y guardar ingestion_ts.
- Dedupe por cve_id y aplicar control de versiones (only update if last_modified cambia).
- Index: cve_id (unique), published_date, products_affected->vendor/product (gin jsonb).

Enriquecimiento
- Unir con EPSS y CISA KEV por cve_id (ver sección unión).
- Extraer entidades nombradas (product names) con spaCy para mejorar matching semántico.

Validaciones
- Esquema JSON válido (schema validation), cvss scores en rango 0-10, cve_id regex `^CVE-\d{4}-\d+$`.

**Ejemplo concreto: NVD JSON → BD → Haystack**

Entrada (JSON de NVD):
```json
{
  "cve": {
    "id": "CVE-2024-21762",
    "published": "2024-05-15T10:00:00Z",
    "lastModified": "2024-06-10T14:30:00Z",
    "descriptions": [
      {
        "value": "A critical Remote Code Execution vulnerability in Apache HTTP Server allows attackers to execute arbitrary code..."
      }
    ],
    "metrics": {
      "cvssMetricV31": {
        "cvssData": {
          "baseScore": 9.8,
          "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
        }
      }
    },
    "configurations": {
      "nodes": [
        {
          "cpeMatch": [
            {"criteria": "cpe:2.3:a:apache:http_server:2.4.0:*:*:*:*:*:*:*"},
            {"criteria": "cpe:2.3:a:apache:http_server:2.4.56:*:*:*:*:*:*:*"}
          ]
        }
      ]
    }
  }
}
```

Paso 1 — Parse y extracción (Python):
```python
import json
from datetime import datetime

raw_json = json.loads(nvd_response)
cve_data = raw_json['cve']

# Extraer campos principales
cve_id = cve_data['id']                    # "CVE-2024-21762"
description = cve_data['descriptions'][0]['value']
published = datetime.fromisoformat(cve_data['published'].replace('Z', '+00:00'))
cvss_score = cve_data['metrics']['cvssMetricV31']['cvssData']['baseScore']
cvss_vector = cve_data['metrics']['cvssMetricV31']['cvssData']['vectorString']

# Parsear CPEs (productos)
products = []
for cpe_match in cve_data['configurations']['nodes'][0]['cpeMatch']:
    cpe_parts = cpe_match['criteria'].split(':')  # cpe:2.3:a:vendor:product:version:...
    products.append({
        "vendor": cpe_parts[3],      # apache
        "product": cpe_parts[4],     # http_server
        "version": cpe_parts[5]      # 2.4.0 o 2.4.56
    })
```

Paso 2 — Guardar en PostgreSQL:
```sql
INSERT INTO cves (
  cve_id, 
  description, 
  published_date, 
  cvss_v3_score, 
  cvss_v3_vector,
  products,
  raw_json,
  ingestion_ts
) VALUES (
  'CVE-2024-21762',
  'A critical Remote Code Execution vulnerability in Apache HTTP Server...',
  '2024-05-15T10:00:00Z',
  9.8,
  'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H',
  '[{"vendor":"apache","product":"http_server","version":"2.4.0"},
    {"vendor":"apache","product":"http_server","version":"2.4.56"}]'::jsonb,
  '{completo JSON original}'::jsonb,
  NOW()
);
```

Resultado en BD:
```
cve_id:          CVE-2024-21762
description:     A critical Remote Code Execution vulnerability...
published_date:  2024-05-15 10:00:00 UTC
cvss_v3_score:   9.8
cvss_v3_vector:  CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H
products:        [{"vendor":"apache","product":"http_server",...}]
ingestion_ts:    2024-06-12 14:09:00 UTC
```

Paso 3 — Indexar en Haystack:
```python
from haystack.document_stores.types import Document

doc = Document(
  content="A critical Remote Code Execution vulnerability in Apache HTTP Server allows...",
  meta={
    "cve_id": "CVE-2024-21762",
    "cvss_score": 9.8,
    "published_date": "2024-05-15",
    "vendor": "apache",
    "products": "http_server 2.4.0 - 2.4.56"
  }
)

document_store.write_documents([doc])
```

Nota: **NO** se convierte el JSON a texto plano. Se **estructura** en columnas (cve_id, cvss_score, etc.) para permitir filtros. Solo la `description` se embeedea como vector. Los campos numéricos y IDs quedan como **metadata filtrable**.

2) EPSS
-------
Qué brinda
- Probabilidad (0..1) de que una CVE sea explotada durante los próximos 30 días.
Formato
- CSV o JSON API con campos: cve_id, epss_probability, percentile, last_updated
Necesita transformación? Sí — leve/mediana

Transformaciones
- Guardar raw feed (epss_raw) con ingestion_ts
- Normalizar a tabla `epss`:
  - cve_id (FK)
  - epss_score (float)
  - percentile (float)
  - feed_ts (timestamp)
- Mantener histórico: insert temporal por fecha; mark current=true para el último
- Unir con `cves` mediante upsert: actualizar campos epss_score, epss_last_seen

Validaciones
- cve_id existe en NVD (si no, ingestar igualmente pero flag `orphan=true`)
- epss_score ∈ [0,1]

3) CISA KEV
----------
Qué brinda
- Lista de CVEs confirmadas explotadas en el mundo real; campos: cve_id, date_added, vendor_product, advisory_link
Formato
- CSV/JSON
Necesita transformación? Sí — leve

Transformaciones
- Guardar raw feed
- Normalizar a tabla `cisa_kev`:
  - cve_id, date_added, advisory_url, notes
- Flag `is_kev` en tabla `cves` (boolean) y `kev_date`.
- Prioridad calculada: if is_kev true then priority += HIGH

Validaciones
- cve_id format
- date_added parsable

4) MITRE ATT&CK
---------------
Qué brinda
- Taxonomía: tácticas → técnicas → sub-técnicas; mitigaciones y detecciones; mapping a IDs ATT&CK (Txxxxx).
Formato
- STIX/JSON (oficial) y JSON simplificado
Necesita transformación? Sí — media

Transformaciones
- Convertir STIX to normalized JSON: extraer objects type `attack-pattern`, `course-of-action`, `relationship`.
- Tablas sugeridas:
  - `attack_tactics` (id, name, description)
  - `attack_techniques` (tech_id, name, description, tactic_ids[])
  - `technique_mitigations` (tech_id, mitigation_id)
- Mantener graph-relations: usar tabla `attack_relations` (from_id,to_id,type)
- Extract mapping fields: mappings to CWE, CVE references if present.

Validaciones
- Cada technique has valid ATT&CK ID
- No orphan techniques (or flag)

5) CWE
------
Qué brinda
- Definiciones de debilidades (CWE IDs), relaciones padre/hijo, mitigaciones y ejemplos.
Formato
- XML/HTML (export)
Necesita transformación? Sí — media

Transformaciones
- Parsear XML y normalizar a `cwe` table:
  - cwe_id, name, description, relationships (parent_ids[]), mitigations[]
- Mapear CWE ↔ ATT&CK ↔ CVE cuando existan referencias cruzadas.

Validaciones
- cwe_id regex `CWE-\d+`

6) Exploit-DB
-------------
Qué brinda
- Registros de exploits: id, título, CVE (a veces), fecha, descripción, código exploit
Formato
- Metadatos CSV + textos (HTML)
Necesita transformación? Sí — fuerte

Transformaciones
- Guardar raw
- Extraer metadatos a `exploitdb`:
  - exploit_id, title, date_published, cve_id (nullable), platforms, raw_text
- Separar bloques de código (delimitadores) y almacenarlos en `exploit_code` (tipo, language, code)
- Redactar/etiquetar contenido peligroso: marcar `contains_exec=true` y tratar con precaución en pipelines de generación (seguridad)
- Extraer IOCs (IPs, domains) con regex y store en `iocs` table

Validaciones
- Si cve_id existe, linkear; caso contrario, flag para revisión manual

Unión/Enriquecimiento (Join logic)
---------------------------------
- Llave primaria común: `cve_id` (fuente privilegiada: NVD). Todas las tablas deben permitir null cve_id pero mejor unir por este.
- Flujo de enriquecimiento:
  1. Ingestar NVD → construir `cves` base
  2. Ingestar EPSS → upsert `epss` y actualizar `cves.epss_score`
  3. Ingestar CISA KEV → set `cves.is_kev=true` y `cves.kev_date`
  4. Ingestar OSV/Advisories → mapear a `cves` o crear `oss_vuln` con mapping
  5. Ingestar ATT&CK/CWE → relacionar técnicas/marcos (para respuestas procedimentales)
  6. Ingestar Exploit-DB → agregar evidencia y code snippets con restricciones de uso

Esquema de ejemplo (Postgres)
----------------------------
-- Tabla cves (simplificada)
CREATE TABLE cves (
  cve_id text PRIMARY KEY,
  description text,
  published_date timestamptz,
  last_modified timestamptz,
  cvss_v3_score numeric,
  cvss_v3_vector text,
  products jsonb,
  epss_score numeric,
  is_kev boolean DEFAULT false,
  kev_date timestamptz,
  raw jsonb
);

Index recomendados: GIN en raw/jsonb, index en published_date, index en epss_score desc

Pipeline operacional (paso a paso)
---------------------------------
1. Fetch raw feed (store raw with feed metadata and checksum)
2. Validate schema + signature (if disponible)
3. Parse to normalized schema
4. Enrich (CPE parsing, spaCy NER for products, EPSS/KEV join)
5. Deduplicate / upsert using cve_id
6. Persist processed row + audit log (who/when/why)
7. Emit metrics y alertas si ingestion fails
8. Trigger index rebuild (partial/incremental) o index update

Chunking y retrievability
-------------------------
- Para datos estructurados (CVE JSON), _no_ chunkear en fragmentos pequeños; cada CVE debe ser una unidad atómica (chunk por cve_id). Esto preserva atomicidad y evita pérdida de contexto.
- Para documentos grandes (NIST SP) usar chunking semántico (fuera de este doc) — pero aquí se registran metadatos de linkeo entre CVE ↔ NIST

Calidad, tests y validaciones
-----------------------------
- Checks automáticos: schema validation, cvss ranges, cve_id format, no-null campos clave
- Tests unitarios para parsers (ej: fixtures con CVE JSON/EPSS CSV)
- End-to-end tests: ingestar nuevo CVE publicado después del cutoff y validar retrieval
- Monitoreo: ingestion lag, error rate, percent missing cve_id joins

Actualización incremental
------------------------
- Prefer incremental feeds (NVD delta feeds) con checkpointing por feed date
- Mantener `ingestion_checkpoint` por fuente
- En caso de reindex completo: snapshot raw -> reparse

Seguridad y gobernanza
----------------------
- Mantener raw_original signed if available
- Control de acceso RBAC sobre tablas con datos sensibles
- Sanitizar exploit code si va al contexto del LLM: policy para no ejecutar/emitir código sin disclaimers
- Retención legal y licence tracking (estos documentos usan CC BY-NC-SA — respetar usos)

Recomendaciones técnicas y librerías
-----------------------------------
- Python: requests, jsonschema, pandas (para CSV), xmltodict, stix2, cpe
- DB: Postgres + pgvector (recomendado por el proyecto)
- NLP: spaCy para NER, regex para IOCs
- Jobs: Airflow/Prefect/Rust cron + idempotent workers

Entregables
-----------
- Folder `ingest/` con parsers (uno por fuente)
- Documentación de API interna (contrato JSON in/out)
- SQL DDL para tablas principales
- Test fixtures para cada parser

Preguntas abiertas
------------------
- ¿Dónde almacenar vectores procesados? (Postgres+pgvector es la opción actual del proyecto)
- ¿Qué retención histórica queréis para EPSS/feeds (guardar TODO o solo último)?
- ¿Nivel de automatización permitido para Exploit-DB (procesado automático vs revisión manual)?
