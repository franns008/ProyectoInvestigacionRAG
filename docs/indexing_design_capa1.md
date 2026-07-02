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

## Hallazgo (2026-07-02) — las queries por identificador no recuperan el CVE correcto

**Estado: limitación conocida, NO corregida todavía** (se dejó para la fase de
Mejoras del cronograma; corregirla ahora excedía el scope de "primer prototipo E2E").

**Qué se observó:** tras indexar 200 CVEs y preguntar en OpenWebUI
"¿Qué es CVE-1999-0095?", el sistema respondió con datos de OTROS CVEs (unos
"Rejected reason: DO NOT USE THIS CANDIDATE NUMBER..."), no del CVE-1999-0095 real
(el de Sendmail que sí está indexado). El circuito E2E funciona (fetch → index →
retrieval → LLM sin errores), pero el retrieval trajo los documentos equivocados.

**Evidencia (de `src/pipeline/logCiberseguridad.txt`):**
- `KEYWORD RETRIEVER: 0 documentos recuperados`.
- `EMBEDDING RETRIEVER`: top-5 = CVEs "rejected/duplicate" semánticamente parecidos
  entre sí, ninguno el CVE-1999-0095 buscado.
- El LLM contestó fielmente sobre el contexto equivocado que le llegó.

**Causa raíz:** el `content` indexado es **solo la descripción**; el `cve_id` vive
únicamente en `meta` (que se filtra pero NO se busca por texto ni se embebe).
Entonces:
- BM25 (keyword) busca sobre `content` → como "CVE-1999-0095" no está ahí, 0 matches.
- El embedding compara el significado de la pregunta contra las descripciones → un ID
  es un token opaco, no se parece semánticamente a "debug command in Sendmail" → el
  doc correcto no entra al top-k.

Esto es exactamente el desafío que `plan_de_trabajo.md` anticipa en "Terminología
multi-vocabulario": *los embeddings genéricos fallan en exact-match de identificadores
→ hybrid retrieval (BM25 + denso)*. El hybrid ya está armado, pero no alcanza si el
identificador no está en el texto buscable. Afecta al tipo de query **factual**
(fila 1 de la taxonomía del plan).

**Fix propuesto (no implementado):** incluir el `cve_id` (y a evaluar: vendor/producto)
dentro del `content` que se embebe/indexa, p.ej.:
```
content = "CVE-1999-0095: The debug command in Sendmail is enabled, allowing attackers to execute commands as root."
```
Así BM25 puede matchear el ID exacto y el embedding lo incluye. Trade-off a medir:
cuánto identificador agregar sin diluir la señal semántica de la descripción
(decisión abierta para la fase de Mejoras).

**Nota de implementación para cuando se retome:** cambiar el formato del `content`
cambia el embedding, así que hay que **re-indexar**. Pero el skip actual de
`index_nvd.py` compara `last_modified` y saltearía los CVEs "sin cambios" aunque el
formato del content haya cambiado → hará falta un `--force` (o versionar el formato
del content en `meta` y comparar eso también) para forzar el re-embed.

## Próximos pasos (no implementados todavía)

- **(Fase Mejoras)** Aplicar el fix del hallazgo de arriba (cve_id en el content) y
  medir el impacto en Hit Rate@k / MRR sobre las queries factuales por ID.
- Definir el "adapter" por fuente: cada fuente tendrá su propio parser
  (`fetch_nvd.py` ya existe; faltan `fetch_attack.py`, `fetch_cwe.py`,
  `fetch_epss.py`, `fetch_kev.py`) que devuelva un dict normalizado, más una
  función común que arme el `Document` Haystack a partir de ese dict —
  pendiente de decidir si se comparte código entre fuentes desde el día uno
  o se empieza con scripts independientes y se refactoriza si aparece
  duplicación real (pregunta abierta, sin resolver aún con el equipo).
- Completar el fetch (`fetch_nvd.py --full` se cortó en ~62 páginas / CVEs viejos)
  y correr `index_nvd.py` sin `--limit` para indexar el catálogo completo.
- Implementar el resto de los fetchers de Capa 1 según
  `docs/data_sourcing_research.md`.
