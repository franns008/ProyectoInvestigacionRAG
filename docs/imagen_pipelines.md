# Imagen del container `pipelines`: variante slim (sin marker-pdf) y full

Este documento explica cómo está construida la imagen del container `pipelines`
([`infrastructure/Dockerfile.pipelines`](../infrastructure/Dockerfile.pipelines)), por
qué **marker-pdf es opcional** (imagen "slim" por defecto), qué se pierde/gana, y cómo
volver a la imagen "full" cuando haga falta convertir PDFs nuevos.

> **Estado (2026-07-17):** implementado. `INSTALL_MARKER=false` es el default → imagen
> **slim**. El runtime del pipeline degrada con gracia si marker no está.

## 1. Por qué existe la variante slim

La imagen del `pipelines` es pesada porque hornea modelos de ML:

| Componente | Peso aprox. | ¿Para qué? |
|---|---|---|
| Modelo del reranker (`bge-reranker-v2-m3`) | ~4.3 GB | Re-rankeo cross-encoder en cada query (siempre se usa). Ver [reranker_cross_encoder.md](reranker_cross_encoder.md). |
| **marker-pdf (modelos OCR de surya)** | **~3 GB** | Convertir **PDFs nuevos** a Markdown en runtime. |
| torch + haystack + sentence-transformers | ~4-5 GB | Base de todo. |

**El punto clave:** marker-pdf **sólo** se usa para convertir PDFs que todavía no están
en la caché `rawdata/_converted_md/`. Una vez convertidos (los 3 estudios INCIBE ya lo
están), marker **nunca vuelve a cargarse**. Tener ~3 GB de modelos OCR horneados en una
imagen que en régimen estable no los toca es puro peso muerto en disco.

Por eso marker pasó a ser **opcional vía build-arg**, apagado por defecto.

## 2. Cómo funciona el flag

En [`Dockerfile.pipelines`](../infrastructure/Dockerfile.pipelines):

```dockerfile
ARG INSTALL_MARKER=false
RUN if [ "$INSTALL_MARKER" = "true" ]; then \
        pip install --no-cache-dir marker-pdf && \
        pip install --no-cache-dir --force-reinstall torch==2.7.0 torchvision==0.22.0 --index-url ${TORCH_INDEX_URL} && \
        python -c "from marker.converters.pdf import PdfConverter; print('marker-pdf OK')" ; \
    else \
        echo "imagen SLIM sin marker-pdf" ; \
    fi
```

- **`INSTALL_MARKER=false` (default):** imagen **slim**, ~3 GB menos. No puede convertir
  PDFs nuevos en runtime.
- **`INSTALL_MARKER=true`:** imagen **full**, idéntica a la anterior. Convierte PDFs.

`torch` se instala **explícito y primero** (antes lo arrastraba marker como dependencia),
para que la imagen slim también tenga torch fijado. El overlay de GPU
([`docker-compose.nvidia.yml`](../infrastructure/docker-compose.nvidia.yml)) sigue pasando
`TORCH_INDEX_URL` para la build CUDA, sin cambios.

## 3. Por qué la imagen slim NO rompe el pipeline

`marker` se importa **lazy y dentro de un `try/except`** en
[`_get_marker_converter`](../src/pipeline/pipeline_ciberseguridad.py) (no en el top del
módulo). El flujo de conversión es tolerante a que marker no exista:

1. `_convert_pdf_with_marker` **primero mira la caché** (`_converted_md/<pdf>.md`). Si el
   `.md` existe, lo reutiliza **sin siquiera importar marker**. Este es el caso normal.
2. Sólo si el `.md` NO existe intenta cargar marker. En la imagen slim eso falla, se
   captura la excepción, se loguea un warning (`marker-pdf no disponible, salteando`) y el
   PDF **se saltea** — el resto de la indexación sigue normal, sin caídas.

Es decir: con los PDFs ya convertidos, la imagen slim se comporta **igual** que la full.
Sólo pierde la capacidad de incorporar PDFs **nuevos** automáticamente.

## 4. Qué hacer si aparece un PDF nuevo

Tres caminos, de menos a más trabajo:

1. **Rebuild puntual a full** (recomendado si es esporádico):
   ```bash
   cd infrastructure
   docker compose build --build-arg INSTALL_MARKER=true pipelines
   docker compose up -d pipelines     # indexa el PDF nuevo y lo cachea en _converted_md
   ```
   Una vez cacheado el `.md`, se puede volver a la imagen slim (rebuild sin el arg) sin
   perder la conversión ya hecha.

2. **Convertir el PDF por fuera** (con marker en un venv/host o un container efímero) y
   dejar el `.md` resultante en `data/raw/_converted_md/`. La imagen slim lo toma como
   caché normal.

3. **Dejar la imagen full de forma permanente** si vas a agregar PDFs seguido: buildeá
   siempre con `--build-arg INSTALL_MARKER=true`.

## 5. Comandos

```bash
cd infrastructure

# Imagen SLIM (default, sin marker) — ~3 GB menos
docker compose build pipelines

# Imagen FULL (con marker, convierte PDFs nuevos)
docker compose build --build-arg INSTALL_MARKER=true pipelines
```

## 6. Relación con el volumen `marker_cache`

Aparte de esto, el mount del volumen `marker_cache` (que duplicaba ~4.3 GB del modelo del
reranker en disco al crear el container) quedó **desactivado** en
[`docker-compose.yml`](../infrastructure/docker-compose.yml): el modelo ya está horneado en
la imagen y el runtime lo lee directo desde ahí. Son dos ahorros independientes:

| Cambio | Ahorro | Dónde |
|---|---|---|
| marker-pdf opcional (default off) | ~3 GB en la **imagen** | `Dockerfile.pipelines` |
| `marker_cache` desmontado | ~4.3 GB duplicados en **volumen** | `docker-compose.yml` |

## 7. Relacionado

- [reranker_cross_encoder.md](reranker_cross_encoder.md) — el reranker que se hornea en la
  imagen (esos ~4.3 GB no son opcionales: se usan en cada query).
- [arquitectura_groq.md](arquitectura_groq.md) — arquitectura general (CPU / GPU).
- [modos_llm.md](modos_llm.md) — Groq vs Ollama para la generación.
