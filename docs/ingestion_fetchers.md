# Fetchers de ingesta — cómo bajar los datos crudos

Fecha: 2026-09-11

Cómo se obtiene cada feed de la Capa 1. La investigación de las fuentes está en
[`data_sourcing_research.md`](data_sourcing_research.md); el detalle de NVD (que es el
único con API key y checkpoint) en [`ingestion_nvd_setup.md`](ingestion_nvd_setup.md).

## Uso

```bash
python src/ingestion/fetch_all.py                  # OSV + EPSS + KEV + CWE (~57 MB la primera vez)
python src/ingestion/fetch_all.py --include-nvd    # + NVD (lento la primera vez)
python src/ingestion/fetch_all.py --only cwe osv   # un subconjunto
python src/ingestion/fetch_all.py --force          # rebaja todo sin preguntar
```

**Correlo seguido: es barato.** Las descargas son condicionales, así que una corrida
sobre datos al día no transfiere nada y termina en menos de un segundo:

```
=== resumen ===
sin cambios  osv   34.3 MB, del 2026-09-12
actualizado  epss  2.6 MB -> data/raw/epss/epss_scores-current.csv.gz
sin cambios  kev   1.7 MB, del 2026-09-11
sin cambios  cwe   CWE 4.20 (2026-04-30), 18.2 MB

1 de 4 con novedades (epss) -> data/raw
```

Los estados son `nuevo` (no existía), `actualizado` (cambió en el servidor),
`sin cambios` (304) y `FALLO`. El proceso sale con código 1 si alguna falló.

Cada fuente vive además en su propio archivo y se puede correr suelta:

```bash
python src/ingestion/fetch_cwe.py
python src/ingestion/fetch_nvd.py --full
```

Sólo hace falta `requests` (y `python-dotenv` para NVD). No necesita Docker, ni la base,
ni LLM.

**NVD queda afuera del set por defecto** a propósito: su carga inicial son ~250k CVEs y
domina el tiempo por órdenes de magnitud, mientras que las otras cuatro son segundos. Es
el mismo criterio que usa `run_indexing.py` con `--include-cve`.

Si una fuente falla, las demás siguen; al final se imprime un resumen y el proceso sale
con código 1 si alguna falló.

## Las fuentes

| Fuente | URL | Tamaño | Destino | Quién lo consume |
|---|---|---|---|---|
| OSV | `storage.googleapis.com/osv-vulnerabilities/PyPI/all.zip` | 34 MB | `data/raw/osv/all.zip` | escaneo de dependencias (**obligatorio**) |
| EPSS | `epss.empiricalsecurity.com/epss_scores-current.csv.gz` | 2,6 MB | `data/raw/epss/epss_scores-current.csv.gz` | escaneo (opcional) |
| KEV | `cisa.gov/.../known_exploited_vulnerabilities.json` | 1,7 MB | `data/raw/kev/known_exploited_vulnerabilities.json` | escaneo (opcional) |
| CWE | `cwe.mitre.org/data/xml/cwec_latest.xml.zip` | 2 MB → 18 MB | `data/raw/cwec_latest.xml` + zip en `data/raw/cwe/<ts>/` | indexación del RAG |
| NVD | `services.nvd.nist.gov/rest/json/cves/2.0` | ~250k CVEs | `data/raw/nvd/<ts>/cves_page_*.json` | indexación del RAG (`--include-cve`) |

Todo va a `data/raw/`, que es lo que el `docker-compose` montea como el volumen `rawdata`
→ `/app/pipelines/rawdata` (el `INPUT_DIR` del indexer) y también el `--data` que espera
`deps/cli.py`. Está en `.gitignore`: son dumps regenerables.

> `infrastructure/appdata/rawdata/` es una copia vieja que **no montea nada**. El corpus
> real es `data/raw/`.

### Por qué el XML de CWE va suelto en `data/raw/`

El indexer levanta los XML con `INPUT_DIR.glob("*.xml")` filtrando subcarpetas, así que
el catálogo tiene que quedar plano en `data/raw/`. El zip crudo sí se archiva en
`data/raw/cwe/<timestamp>/`, fuera del glob. El nombre publicado
(`cwec_latest.xml`) es fijo aunque el de adentro del zip venga versionado
(`cwec_v4.20.xml`): así cada corrida pisa la anterior en vez de acumular 18 MB por release.

## Detalles que valen la pena

**Cómo se decide si hay que bajar algo.** Se manda `If-Modified-Since` con el `mtime` del
archivo local y, si el servidor contesta 304, no se transfiere nada. Al bajar, se le fija
al archivo el `Last-Modified` del servidor, y ese mtime es el que se manda la próxima vez.
No hay archivo de estado aparte: el estado es el archivo, como en `curl -z` / `wget -N`,
así que no se puede desincronizar. Verificado contra las cuatro fuentes; `If-None-Match`
se descartó porque el ETag de CISA no vuelve igual.

**CWE no es incremental.** Se libera por versiones (hoy 4.20), no por día: cuando hay
release nueva se baja el catálogo entero, y entre releases da 304. El catálogo completo
tiene **969 weaknesses**; las 5 vistas parciales que había antes en el repo cubrían 710.

**La REST API de MITRE no sirve para bajar el catálogo.** `cwe-api.mitre.org` anda bien
para consultas puntuales (`/api/v1/cwe/version`, `/cwe/weakness/79`), pero su endpoint no
documentado `/cwe/weakness/all` devuelve HTTP 200 con el JSON **truncado** (corta a los
~10 MB). Queda como opción futura sólo para la jerarquía padre-hijo
(`parents`/`children`), que hoy no usamos.

**EPSS redirige.** La URL `-current` responde 302 al CSV del día; `requests` sigue el
redirect solo.

**Las descargas se validan.** Cada fetcher escribe a un temporal y recién ahí renombra
(una descarga cortada no deja un archivo a medias que el escáner intente leer) y verifica
un tamaño mínimo para detectar respuestas truncadas.

## Limitación conocida: lo ya indexado no se refresca

`run_indexing.py` inicializa su set de deduplicación con los documentos que ya están en
la base, así que un CWE (o un CVE) ya indexado **no se actualiza** aunque el fetcher traiga
contenido nuevo. Cuando salga CWE 4.21, correr el fetcher + la indexación va a agregar las
weaknesses nuevas pero no a corregir las viejas. Para eso hoy hay que vaciar la tabla.

## Qué falta

- **ATT&CK**: investigado en `data_sourcing_research.md`, sin fetcher — todavía no lo
  consume nada en el repo.
- **Incrementales de OSV**: OSV publica un `modified_id.csv` que permitiría no rebajar los
  34 MB enteros. Hoy se rebaja todo (tarda segundos, no molesta).
- **Jerarquía de CWE** (`parent_ids[]` de `data_transform_spec.md`): necesitaría la REST
  API, porque el converter actual sólo saca ID, nombre y descripción.
