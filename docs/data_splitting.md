# Data Splitting (Chunking) — Estado actual y mejoras propuestas

Este documento describe cómo se parte (chunkea) el corpus antes de indexarlo en el
RAG de ciberseguridad, analiza críticamente la configuración actual y detalla las
mejoras propuestas con su impacto en calidad de recuperación y en consumo de tokens
de Groq.

> **Recordatorio de arquitectura:** los **embeddings** se calculan **local** en Ollama
> (`bge-m3`, 1024 dims) y sólo la **generación** va a **Groq**. Por lo tanto el chunking
> **no consume tokens de Groq durante la indexación**; sólo influye en Groq por la
> cantidad de contexto que termina en el prompt de cada consulta. Ver
> [arquitectura_groq.md](arquitectura_groq.md).

---

## 1. Dónde vive el chunking

Todo el splitting está en un único punto del pipeline:
[`src/pipeline/pipeline_ciberseguridad.py`](../src/pipeline/pipeline_ciberseguridad.py),
método `_index_new_documents()`:

```python
splitter = DocumentSplitter(
    split_by="word",
    split_length=self.valves.split_length,   # 200
    split_overlap=self.valves.split_overlap,  # 20
)
docs = splitter.run(documents=docs)["documents"]
```

Parámetros expuestos en `Valves`:

| Parámetro       | Valor actual | Significado                         |
|-----------------|--------------|-------------------------------------|
| `split_by`      | `"word"`     | Unidad de corte: palabras           |
| `split_length`  | `200`        | Palabras por chunk (~280 tokens)    |
| `split_overlap` | `20`         | Solapamiento (10%)                  |

Un **único** splitter procesa todos los formatos de entrada (PDF→MD vía marker,
DOCX, MD nativo, TXT) con la misma configuración.

---

## 2. El corpus real

El corpus son estudios técnicos de INCIBE convertidos a Markdown estructurado. No es
prosa plana: tiene jerarquía de secciones, tablas y bloques de código.

| Documento   | Headings | Tablas (líneas `\|`) | Bloques de código | Palabras |
|-------------|----------|----------------------|-------------------|----------|
| Cloud SCI   | 19       | 18                   | 0                 | ~4.100   |
| XSS         | 27       | 12                   | 24                | ~8.200   |
| Redes 5G    | 142      | 52                   | 0                 | ~23.100  |

La estructura (`3.1 Definición de XSS`, `7.2 Programación segura`, payloads XSS en
bloques de código, tablas de configuración) es **información semántica valiosa** que
la configuración actual ignora por completo.

---

## 3. Análisis crítico de la configuración actual

1. **El corte por palabras destruye la estructura semántica.** Un límite fijo de 200
   palabras parte secciones a la mitad, corta tablas dejando filas huérfanas sin su
   encabezado (que es lo que da significado a las columnas) y trocea payloads XSS
   mid-token. Con 142 headings en la guía 5G, se están fragmentando unidades de
   significado.

2. **200 palabras es demasiado chico.** `bge-m3` admite 8192 tokens y se le están dando
   ~280. Chunks tan cortos fragmentan explicaciones autocontenidas; el retriever
   devuelve pedazos sin el contexto que el LLM necesita. Para documentos
   técnicos/expositivos lo habitual es 400-700 palabras (~512-1024 tokens).

3. **El heading queda huérfano de su contenido.** El título "7.2 Programación segura"
   es contexto muy útil para el retrieval, pero al ser sólo palabras más en el stream,
   frecuentemente cae en un chunk distinto al de su propio contenido.

4. **Corta a mitad de oración.** `split_by="word"` no respeta límites de frase.

5. **Idioma no configurado.** El corpus es español pero el tokenizador de oraciones de
   Haystack usa inglés por defecto.

6. **Fuga de contexto en el `DocumentJoiner`.** Se crea **sin `top_k`**, así que el RRF
   entrega la **unión** de ambos retrievers sin recorte → hasta ~2×`top_k` chunks
   llegan al prompt de Groq. Es la palanca de costo #1 y hoy está suelta.

7. **Inconsistencias de configuración.** El código usa `retriever_top_k=3` pero el
   `valves.json` de runtime tiene `5`; `max_tokens=512` en código vs `num_predict=1000`
   en runtime. Además, cambiar `split_length` en runtime **no re-chunkea lo ya
   indexado** (`on_valves_updated` sólo reconstruye el pipeline de query, no re-indexa).

---

## 4. Impacto en consumo de tokens (Groq)

El chunking **no** toca la cuota de embeddings (local en Ollama). Impacta en Groq
**sólo** por el contexto del prompt: `nº_chunks_recuperados × tamaño_chunk` + plantilla
+ pregunta + salida.

Estimación en español (~1,4 tokens/palabra en el tokenizer de Llama):

| Métrica                       | Actual   | Propuesta ingenua | Propuesta ajustada |
|-------------------------------|----------|-------------------|--------------------|
| Palabras/chunk                | 200      | 500               | 500                |
| Tokens/chunk                  | ~280     | ~700              | ~700               |
| Chunks al prompt              | ~6 (sin cap) | ~6 (sin cap)  | **4 (joiner cap)** |
| Input tokens/query (contexto) | ~1.700   | ~4.200            | ~2.800             |
| + plantilla + pregunta        | ~150     | ~150              | ~150               |
| Salida (`max_tokens`)         | 512      | 512               | 512                |
| **Total tokens/query**        | **~2.400** | **~4.900**      | **~3.500**         |

Conclusiones:

- Subir el chunk a 500 palabras **sin cap** en el joiner ≈ **2,5× el gasto** de Groq por
  query.
- Con un **cap de `top_k` en el joiner** baja a ~**1,5×**, con mejor contexto por pieza.
- **El límite que frena es el TPM (tokens por minuto)** del tier de Groq, no la ventana
  de contexto (Llama-4-Scout tiene ventana enorme). Más tokens/query → menos
  queries/minuto antes del `429`. El código ya mitiga con `max_retries=5` respetando el
  `retry-after`, pero eso es **latencia**, no más cuota.
- **El overlap se paga doble:** las palabras solapadas viajan en dos chunks; si ambos se
  recuperan, van dos veces al prompt. Overlap alto = tokens redundantes a Groq.

---

## 5. Mejoras propuestas

### Mejora 1 — Splitter recursivo respetando estructura (mayor impacto en calidad)

`haystack-ai 2.27` incluye `RecursiveDocumentSplitter`, que parte respetando una
jerarquía de separadores (párrafos → líneas → oraciones) en vez de cortar a ciegas.

```python
from haystack.components.preprocessors import RecursiveDocumentSplitter

splitter = RecursiveDocumentSplitter(
    split_length=500,          # palabras (~700 tokens)
    split_overlap=75,          # ~15%
    split_unit="word",
    separators=["\n\n", "\n", "sentence", " "],  # respeta párrafos y frases
)
splitter.warm_up()             # necesario para el tokenizador de oraciones
```

Elimina la mayoría de cortes a mitad de oración/tabla porque intenta primero cortar en
dobles saltos de línea (párrafos/secciones de Markdown).

### Mejora 2 — Alternativa mínima manteniendo `DocumentSplitter`

Si se prefiere no cambiar de componente:

```python
splitter = DocumentSplitter(
    split_by="period",              # o "sentence"
    split_length=6,                 # ~6 oraciones
    split_overlap=1,
    respect_sentence_boundary=True,
    language="es",                  # ¡hoy por defecto en inglés!
)
```

El `language="es"` es importante y hoy no está.

### Mejora 3 — Cap del `DocumentJoiner` (mayor impacto en costo de Groq)

Palanca de costo #1. Recorta la unión de los dos retrievers tras el RRF:

```python
DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=4)
```

Da un techo de tokens duro y predecible sobre el prompt enviado a Groq.

### Mejora 4 — Preservar el heading en cada chunk (contextual chunking)

Prependar la sección a la que pertenece el chunk antes de embeber. Simple y muy
efectivo para retrieval:

```python
for doc in split_docs:
    header = doc.meta.get("header") or ""   # requiere trackear la sección
    if header:
        doc.content = f"[{header}]\n{doc.content}"
```

Requiere un splitter markdown-aware o preservar headings en `meta`.

### Mejora 5 — Tratar tablas aparte

Las 52 líneas de tabla de la guía 5G se benefician de no ser cortadas nunca. Como
mínimo, `RecursiveDocumentSplitter` con `\n\n` las mantiene más íntegras; lo ideal sería
un pre-paso que extraiga cada tabla como documento propio.

### Mejora 6 — Re-indexar al cambiar los valves

Hoy cambiar `split_length`/`split_overlap` en runtime no tiene efecto sobre lo ya
almacenado. Añadir la re-indexación (re-chunk + re-embed + overwrite) en
`on_valves_updated` para que los parámetros sean configurables de verdad.

### Mejora 7 — Alinear parámetros código ↔ runtime

Unificar `retriever_top_k` y `max_tokens`/`num_predict` entre el código y `valves.json`
para evitar sorpresas silenciosas. El output cuenta para el TPM; 512 suele alcanzar.

---

## 6. Recomendación priorizada

| Prioridad | Mejora                                   | Beneficio principal              |
|-----------|------------------------------------------|----------------------------------|
| 1         | Cap del joiner (`top_k=4`)               | Costo Groq predecible (–~40%)    |
| 2         | Splitter recursivo (~500 palabras)       | Calidad de recuperación          |
| 3         | Re-indexar al cambiar valves             | Config utilizable de verdad      |
| 4         | Heading contextual + `language="es"`     | Precisión de recuperación        |
| 5         | Tablas aparte + alinear parámetros       | Robustez / consistencia          |

**Combo de arranque (≈80% del beneficio con poco código):** Mejora 3 (cap del joiner) +
Mejora 1 (recursivo 500 palabras) + Mejora 6 (re-indexar). Mejor contexto por query con
sólo ~1,5× el costo de Groq, manejable dentro del TPM.
