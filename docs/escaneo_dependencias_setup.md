# Puesta en marcha del escaneo de dependencias

Cómo dejar funcionando en una máquina nueva el escaneo de `requirements.txt` y su
extensión de VSCode. Para **por qué** está diseñado así, ver
[`escaneo_dependencias.md`](escaneo_dependencias.md).

> **Nada de esto necesita el stack.** Ni Docker, ni pgvector, ni Ollama, ni GPU, ni una
> API key. El escaneo es determinístico: lógica pura sobre tres archivos descargados.
> El único momento en que hace falta red es al bajar esos archivos.

## Requisitos

| | Para qué | Mínimo |
|---|---|---|
| Python | el escáner | 3.10+ (usa `X \| Y` en anotaciones) |
| `packaging` | comparar versiones PEP 440 y parsear PEP 508 | cualquiera |
| `curl` | bajar los datos | — |
| Node.js + npm | compilar la extensión | Node 18+ |
| VSCode | correr la extensión | 1.90+ |

**El escáner no necesita Haystack.** `src/pipeline/deps/` importa solo la biblioteca
estándar y `packaging`; verificado corriendo la suite completa en un entorno con esos
dos paquetes y nada más. El `requirements.txt` del repo es para el RAG, que es otra cosa.

## Puesta en marcha

### 1. Entorno de Python

```bash
git clone <repo> && cd ProyectoInvestigacionRAG
python3 -m venv .venv
.venv/bin/pip install packaging pytest
```

Si además vas a tocar el RAG, `pip install -r requirements.txt -r requirements-dev.txt`.
Para el escaneo solo, con lo de arriba alcanza.

### 2. Datos

```bash
./scripts/fetch_deps_data.sh
```

Baja tres archivos a `data/raw/` (~37 MB, unos 6 segundos):

| Fuente | Tamaño | Qué aporta |
|---|---|---|
| `osv/all.zip` | 33 MB | 13.400 advisories de PyPI con rangos de versión |
| `epss/epss_scores-current.csv.gz` | 2,5 MB | probabilidad de explotación de 365.000 CVE |
| `kev/known_exploited_vulnerabilities.json` | 1,6 MB | las ~1.700 que CISA confirma explotadas |

Están en `.gitignore`: son dumps regenerables, no artefactos del repo. EPSS y KEV cambian
a diario, así que conviene volver a correrlo antes de una demo (`--force` rebaja todo).

### 3. Verificar

```bash
.venv/bin/python -m pytest
```

Esperado: **54 tests en menos de un segundo.** No necesitan los datos del paso 2 —
corren contra advisories reales congelados en `tests/fixtures/osv/`, para no depender
del dump del día.

Si esto pasa, la lógica está bien. Es el único pedazo del sistema que se *verifica* en
lugar de medirse estadísticamente.

### 4. Escanear desde la terminal

```bash
cd src/pipeline
../../.venv/bin/python -m deps.cli ../../requirements_demo.txt --data ../../data/raw
```

Agregá `--json` para la salida que consume la extensión, o `--top N` para acortar.

> **Ojo con el manifiesto.** Solo se escanean las líneas con pin exacto (`==`). Un
> `requirements.txt` con `>=` o `~=` sale con cero hallazgos y todas las líneas listadas
> como no escaneadas — el `requirements.txt` de este repo es justamente así. Para probar,
> usá un archivo con versiones viejas fijadas.
>
> Un caso que muestra bien el ordenamiento, porque incluye una vulnerabilidad del
> catálogo de CISA:
>
> ```
> Django==2.2.0
> Pillow==5.2.0
> requests==2.19.1
> urllib3==1.24.1
> cryptography==2.3
> ```

### 5. La extensión

```bash
cd extension
npm install
```

Después, en VSCode:

1. `File > Open Folder…` → elegí **la carpeta `extension`**, no la raíz del repo. F5 lee
   el `launch.json` que está adentro.
2. **F5**. Compila y abre una segunda ventana (*Extension Development Host*) con la raíz
   del repo ya abierta.
3. En la ventana nueva, abrí un `requirements.txt` con pines `==`.
4. Abajo a la derecha aparece **🛡 Escanear dependencias**. También está el botón en la
   barra del editor y el comando en la paleta (`Ctrl+Shift+P` → "Escanear").

La ventana original queda como consola de depuración: los errores de la extensión salen
ahí (`Ctrl+Shift+Y`), no en la de prueba.

El intérprete lo detecta solo si hay un `.venv` en la raíz del workspace. Si no, hay que
apuntarlo con el ajuste `cibersec.pythonPath`.

## Cómo funciona

```
requirements.txt
      │  clic en la barra de estado
      ▼
extensión (TypeScript)         detecta .venv, arma el comando
      │  spawn
      ▼
python -m deps.cli <archivo> --data data/raw --json
      │
      ├─ parsea el manifiesto (PEP 508); se queda solo con los pines ==
      ├─ indexa 13.400 advisories de OSV por nombre de paquete      ~0,8 s
      ├─ resuelve rangos PEP 440: qué advisories afectan ESTA versión
      ├─ deduplica por CVE y une los cwe_ids de los advisories hermanos
      ├─ calcula el CVSS 3.1 desde el vector
      ├─ cruza con EPSS y con CISA KEV
      └─ ordena: KEV > EPSS > CVSS
      │  JSON
      ▼
webview con las tarjetas priorizadas
```

Total ~1,3 s, de los cuales 0,8 s son armar el índice de OSV.

El texto de cada tarjeta es el `summary` que el propio advisory trae escrito. **Hoy no
hay lenguaje natural generado**: hay una tabla bien ordenada.

### Mapa de archivos

| Archivo | Qué hace |
|---|---|
| [`deps/resolver.py`](../src/pipeline/deps/resolver.py) | PEP 503, parseo del manifiesto, rangos de OSV, PEP 440, deduplicación por CVE |
| [`deps/prioritize.py`](../src/pipeline/deps/prioritize.py) | CVSS 3.1 desde el vector, cruce con EPSS/KEV, orden |
| [`deps/cli.py`](../src/pipeline/deps/cli.py) | entrada por línea de comandos, `--json` |
| [`tests/`](../tests/) | 54 tests con advisories reales congelados |
| [`extension/src/scan/types.ts`](../extension/src/scan/types.ts) | el contrato compartido con el RAG |
| [`extension/src/scan/localProvider.ts`](../extension/src/scan/localProvider.ts) | lanza la CLI y parsea su JSON |
| [`extension/src/panel.ts`](../extension/src/panel.ts) | el webview |

## Qué cambia cuando se integre el RAG

La costura ya está puesta, así que el cambio es acotado.

**En Python** aparece `pipeline_dependencias.py`: una clase `Pipeline` que el servidor
levanta sola y publica como un segundo modelo en el `9099`, al lado del chat. Hace el
mismo escaneo determinístico —importa el mismo `deps/`— y después trae el documento del
CWE **por clave exacta** (`filter_documents`, no búsqueda semántica: el id ya se conoce)
y le pide al LLM **solo el párrafo explicativo**.

**En la extensión** cambia el ajuste `cibersec.provider` de `local` a `rag`, y
[`ragProvider.ts`](../extension/src/scan/ragProvider.ts) —que hoy existe y lanza "no
implementado"— recibe un cuerpo que hace un POST al `9099`.

**No cambia** el contrato ni la vista. Los dos proveedores devuelven el mismo
`ScanResult`; el RAG solo completa dos campos que hoy vienen vacíos, `explanation` y
`citations`. El panel ya los renderiza si están y los omite si no.

> **La regla que no se rompe:** la versión de arreglo, el CVSS, el EPSS, el flag de KEV y
> todos los identificadores los concatena Python desde los datos estructurados. El modelo
> solo redacta el párrafo. Un LLM no puede equivocarse en un score que nunca escribe.

## Estado

| | |
|---|---|
| Fase 1 — fetchers de OSV/EPSS/KEV y catálogo CWE completo | **pendiente**. `scripts/fetch_deps_data.sh` cubre la descarga de forma provisional; falta el catálogo completo de CWE y los fetchers en Python con incrementales. |
| Fase 2 — resolver, priorización, tests | hecho |
| Fase 3 — converter de OSV a pgvector, lookup de CWE, `pipeline_dependencias` | **pendiente**. Es lo que habilita la explicación. |
| Fase 4 — extensión | hecho, contra el escáner local |

## Problemas comunes

**"No encontré el dump de OSV bajo data/raw/osv/"** — falta el paso 2.

**Cero hallazgos y todas las líneas en "No escaneadas"** — el manifiesto no tiene pines
`==`. Es el comportamiento correcto, no un error.

**`ModuleNotFoundError: No module named 'packaging'`** — falta el paso 1, o la extensión
está usando otro intérprete: revisá `cibersec.pythonPath`.

**`ModuleNotFoundError: No module named 'deps'`** — la CLI se corre desde `src/pipeline`,
que es el directorio que el container monta como `/app/pipelines`. Desde otro lado, hay
que exportar `PYTHONPATH=<repo>/src/pipeline`.

**F5 abre una ventana que parece normal** — es la correcta. Abrí un `requirements.txt` y
mirá abajo a la derecha. Si el comando tampoco aparece en la paleta, el error está en la
`Debug Console` de la ventana original.

**Los tests pasan pero el escaneo da distinto que ayer** — EPSS y KEV se actualizan a
diario. Los tests usan fixtures congelados justamente para no depender de eso.
