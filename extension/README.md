# Extensión: escaneo de dependencias

Abre un `requirements.txt`, escanea sus dependencias fijadas contra OSV y muestra los
hallazgos **priorizados por explotabilidad real** (CISA KEV → EPSS → CVSS).

Ver [`docs/escaneo_dependencias.md`](../docs/escaneo_dependencias.md) para el diseño.

## Probarla

Una sola vez, para instalar las dependencias:

```bash
cd extension
npm install
```

Después, cada vez que la quieras correr:

1. Abrí **la carpeta `extension/`** en VSCode (`File > Open Folder…`, y elegí
   `extension`, no la raíz del repo). Es importante: F5 usa el `launch.json` que está
   adentro de esa carpeta.
2. Apretá **F5**. Compila sola y abre una **segunda ventana** de VSCode
   (Extension Development Host) con la raíz del repo ya abierta.
3. En esa ventana nueva, abrí un `requirements.txt` con dependencias fijadas con `==`.
4. Apretá el botón **Escanear dependencias vulnerables** arriba a la derecha de la barra
   del editor, o clic derecho sobre el archivo en el explorador.

El panel con los resultados se abre al costado.

> La ventana original queda como consola de depuración: los `console.log` y los errores
> de la extensión aparecen ahí, no en la ventana de prueba.

Requisitos previos: los dumps de OSV, EPSS y CISA KEV en `data/raw/`. El intérprete lo
detecta solo si hay un `.venv` en la raíz del repo.

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
