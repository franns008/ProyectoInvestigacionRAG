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

El panel con los resultados se abre al costado. Cada hallazgo tiene un botón con un clip,
**Agregar al chat**, que lo adjunta al chat de la barra lateral (ícono de escudo en la barra
de actividad). Es el mismo gesto que adjuntar un archivo en Copilot: el hallazgo aparece
como una ficha arriba del input, con su CVE/OSV, paquete, versión, resumen y detalles del
advisory. Tocar de nuevo el botón, o la × de la ficha, lo quita. La clave del servidor se
mantiene en la extensión y no se entrega a los webviews.

### Chat

Hay un único chat, siempre disponible, contra el pipeline `pipeline_ciberseguridad`. Sin
adjuntos es un chat común sobre el corpus del RAG. Los adjuntos son **contexto
persistente**: viajan en cada pregunta hasta que se quitan, así que se puede comparar
varios hallazgos en la misma conversación (hasta seis). El botón **+** de la barra de
título de la vista abre un chat nuevo, sin historial ni adjuntos. La vista se puede
arrastrar a la barra lateral secundaria para tenerla a la derecha, como Copilot.

### Resultados agrupados por librería

El panel muestra primero un resumen por librería. Cada grupo indica el nombre, la cantidad
de vulnerabilidades encontradas y la versión más alta que aparece como arreglo entre sus
advisories. Esa versión máxima se usa solamente como recomendación común del grupo: cada
vulnerabilidad conserva y muestra su propia `fixed_version` original.

Antes del detalle por librería aparece una sección de prioridad inmediata con todas las
vulnerabilidades explotadas que figuran en CISA KEV. Dentro de cada librería, los hallazgos
se ordenan de mayor a menor severidad según su puntuación CVSS; los que no tienen CVSS
quedan al final.

Los grupos se pueden plegar de manera independiente desde su encabezado. Al plegar uno,
queda visible únicamente la librería, la cantidad de vulnerabilidades y la versión común
recomendada, lo que permite comparar rápidamente varias librerías.

Cuando están activadas las explicaciones, la extensión envía al LLM la primera
vulnerabilidad de cada librería siguiendo el orden de priorización del escaneo. Se explican
como máximo diez librerías; `cibersec.explainTopN` permite reducir ese número, pero nunca
ampliarlo por encima de diez.

En el chat, **Enter** envía y **Shift+Enter** agrega una línea. Las respuestas se muestran
con formato markdown (se escapa todo antes, así que el modelo no puede inyectar HTML).
**Ver contexto enviado** muestra el JSON exacto de los adjuntos que se le manda al
pipeline. Debajo de cada respuesta se listan los CVE/CWE citados, marcando los que no
venían de los hallazgos adjuntos: pueden salir del corpus del RAG o ser inventados. Los documentos que
recuperó el RAG todavía no se ven, porque el pipeline no los devuelve.

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
| `cibersec.chatModel` | `pipeline_ciberseguridad` | Pipeline RAG del chat de la barra lateral. |
| `cibersec.chatTimeoutSeconds` | `240` | Tiempo máximo de espera de cada respuesta. |
| `cibersec.explain` | `true` | Solicita explicaciones al pipeline de dependencias. Si está desactivado, el escaneo sigue funcionando sin prosa del LLM. |
| `cibersec.explainModel` | `pipeline_dependencias` | Pipeline que redacta las explicaciones. |
| `cibersec.explainTimeoutSeconds` | `240` | Tiempo máximo de espera para obtener las explicaciones. |
| `cibersec.explainTopN` | `10` | Cantidad máxima configurable de librerías explicadas. Se toma una vulnerabilidad por librería y el límite absoluto es 10. |

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
