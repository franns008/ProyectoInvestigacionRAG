# Extensión: escaneo de dependencias

Abre un `requirements.txt`, escanea sus dependencias fijadas contra OSV y muestra los
hallazgos **priorizados por explotabilidad real** (CISA KEV → EPSS → CVSS).

Ver [`docs/escaneo_dependencias.md`](../docs/escaneo_dependencias.md) para el diseño.

## Probarla

```bash
cd extension
npm install
npm run compile
```

Después, en VSCode: abrir la carpeta `extension/` y presionar **F5**. Se abre una
ventana nueva (Extension Development Host); ahí hay que abrir **la raíz de este repo**
como workspace, abrir un `requirements.txt` y usar el botón *Escanear dependencias
vulnerables* de la barra del editor (o clic derecho sobre el archivo en el explorador).

Requisitos previos: los dumps de OSV, EPSS y CISA KEV en `data/raw/` y un intérprete de
Python con `packaging` instalado.

## Ajustes

| Ajuste | Default | Para qué |
|---|---|---|
| `cibersec.provider` | `local` | `local` = escáner determinístico. `rag` = pipeline con explicación (Fase 3, todavía no implementado). |
| `cibersec.pythonPath` | `python3` | Intérprete con el que corre el escáner. |
| `cibersec.scannerRoot` | `src/pipeline` | De dónde se importa el módulo `deps`. |
| `cibersec.dataDir` | `data/raw` | Dumps de OSV, EPSS y KEV. |
| `cibersec.ragUrl` | `http://localhost:9099` | Servidor de Pipelines. Sólo con `provider: rag`. |
| `cibersec.ragModel` | `pipeline_dependencias` | Sólo con `provider: rag`. |

Las rutas relativas se resuelven contra la raíz del workspace.

## Cómo se integra con el RAG

La extensión no sabe cómo se resuelve un escaneo: le pide el resultado a un
`ScanProvider` ([`src/scan/types.ts`](src/scan/types.ts)) y renderiza lo que recibe.

```
extension.ts → buildProvider() ─┬→ LocalScannerProvider   (hoy: python -m deps.cli --json)
                                └→ RagProvider            (Fase 3: POST al 9099)
```

Ambos devuelven el **mismo** `ScanResult`. La única diferencia es que el proveedor RAG
completa los campos opcionales `explanation` y `citations` de cada hallazgo; el panel ya
los renderiza si están y los omite si no.

Por eso integrar la Fase 3 es implementar el cuerpo de `RagProvider.scan()`. No hay que
tocar la vista, ni el contrato, ni el resto de la extensión.

**Lo que no cambia nunca:** los campos duros (versión de arreglo, CVSS, EPSS, KEV,
identificadores) los arma Python por concatenación desde los datos estructurados. El LLM
sólo escribe `explanation`. Un modelo no puede equivocarse en un score que nunca escribe.
