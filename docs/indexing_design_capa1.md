# Diseño — De JSON crudo a Haystack Document (Capa 1)

Fecha: 2026-07-02
Autor: decisión de diseño discutida en conversación con Claude Code
Contexto: siguiente paso después de `docs/ingestion_nvd_setup.md` (que dejó el
diseño de indexado de NVD acordado pero no implementado) y de
`docs/data_sourcing_research.md` (que investigó cómo traer las 5 fuentes de
Capa 1: NVD, MITRE ATT&CK, CWE, EPSS, CISA KEV).

Este documento responde la pregunta: **una vez que tenemos el JSON/XML crudo
de una fuente de Capa 1 descargado, ¿cómo lo convertimos en algo indexable
por el RAG?** Aplica a las 5 fuentes, no solo a NVD.

---

## Decisión 1 — No todas las fuentes de Capa 1 generan un `Document`

De las 5 fuentes, solo 3 tienen prosa propia (una `description` narrativa
que tiene sentido embeber como vector):

| Fuente | ¿Genera Document propio? | Motivo |
|---|---|---|
| NVD/CVE | Sí | tiene `description` en texto libre |
| MITRE ATT&CK (técnicas) | Sí | tiene `description` de la técnica |
| CWE | Sí | tiene `description` de la debilidad |
| EPSS | No | es un score numérico (`epss_score`, `percentile`), no hay prosa que embeber |
| CISA KEV | No | es una entrada de catálogo (`vendorProject`, `dateAdded`, etc.), la única prosa (`shortDescription`) es redundante con la de NVD |

EPSS y CISA KEV son **enriquecimiento**: como ya definía
`docs/data_transform_spec.md`, actualizan el `meta` de un `Document` de CVE
que ya existe (`epss_score`, `is_kev`, `kev_date`) en vez de crear una unidad
indexable nueva. Tratarlos como Document propio duplicaría información sin
aportar contenido nuevo para retrieval semántico.

## Decisión 2 — JSON → Document directo, sin Markdown como paso intermedio

Se evaluaron dos opciones para las 3 fuentes que sí generan Document:

**Opción A (elegida) — conversión directa:**

```python
doc = Document(
  content="A critical Remote Code Execution vulnerability in Apache HTTP Server allows...",
  meta={
    "cve_id": "CVE-2024-21762",
    "cvss_score": 9.8,
    "published_date": "2024-05-15",
    "vendor": "apache",
    "products": "http_server 2.4.0 - 2.4.56",
    "source_type": "nvd_cve"
  }
)
```

Solo `content` se embebe (vector). Todo lo demás va a `meta`, que en
pgvector es una columna filtrable — no texto a re-parsear.

**Opción B (descartada) — pasar por Markdown intermedio:**

```markdown
# CVE-2024-21762
**CVSS:** 9.8
**Publicado:** 2024-05-15
**Vendor/producto:** apache / http_server 2.4.0 - 2.4.56

A critical Remote Code Execution vulnerability in Apache HTTP Server allows...
```

### Por qué se descartó la Opción B

- Los campos estructurados (CVSS, fechas, IDs, vendor/producto) son
  exactamente lo que la taxonomía del plan de trabajo necesita como
  **metadata filtrable**, no como texto embebido: las queries factuales
  ("¿qué CVSS score tiene tal CVE?") y temporales ("CVEs de los últimos 30
  días") se resuelven filtrando por `meta`, no confiando en que el embedding
  haya capturado bien un número dentro de un párrafo.
- Si se aplana todo a Markdown primero, después hace falta **volver a
  parsear ese Markdown** (regex u otro parser) para reconstruir el `meta` —
  se deshace el trabajo de aplanado que se acaba de hacer.
- Si en cambio el bloque completo (metadata + descripción) se manda entero a
  embeber como un solo texto, el vector resultante mezcla señal semántica de
  la descripción con ruido de metadata, degradando el retrieval semántico.
- Markdown sí tiene sentido como formato para **Capa 2** (NIST SP, CISA
  Advisories, threat reports) porque ahí el destino final *es* prosa larga
  que un chunker semántico debe trocear — no aplica a Capa 1, donde el
  destino final es "una fila con campos filtrables + un texto corto
  embebido".

**Conclusión:** para NVD, ATT&CK y CWE se usa conversión directa
JSON/XML → `Document` Haystack (Opción A), consistente con el diseño ya
acordado en `docs/ingestion_nvd_setup.md` para NVD. EPSS y CISA KEV no
generan Document propio, se implementan como upsert de `meta` sobre
Documents existentes.

## Próximos pasos (no implementados todavía)

- Definir el "adapter" por fuente: cada fuente tendrá su propio parser
  (`fetch_nvd.py` ya existe; faltan `fetch_attack.py`, `fetch_cwe.py`,
  `fetch_epss.py`, `fetch_kev.py`) que devuelva un dict normalizado, más una
  función común que arme el `Document` Haystack a partir de ese dict —
  pendiente de decidir si se comparte código entre fuentes desde el día uno
  o se empieza con scripts independientes y se refactoriza si aparece
  duplicación real (pregunta abierta, sin resolver aún con el equipo).
- Implementar `src/ingestion/index_nvd.py` según el diseño ya descripto en
  `docs/ingestion_nvd_setup.md` sección 6 (primero, porque NVD es la única
  fuente con fetch ya funcionando).
- Implementar el resto de los fetchers de Capa 1 según
  `docs/data_sourcing_research.md`.
