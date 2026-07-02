# Enriquecimiento de chunks antes de embeddar (Contextual Retrieval)

Este documento describe una técnica de preprocesamiento para **mejorar la
recuperación** del RAG de ciberseguridad: enriquecer cada chunk con
meta-información generada (título, resumen, preguntas hipotéticas, contexto del
documento) que se usa **al momento de indexar/embeddar**, sin contaminar el texto
que ve el LLM al generar la respuesta.

Es un plan de implementación, no un cambio ya aplicado sobre
[`src/pipeline/pipeline_ciberseguridad.py`](../src/pipeline/pipeline_ciberseguridad.py).

## 1. El problema que resuelve

En el pipeline actual, cada chunk se embeba tal cual sale del `DocumentSplitter`
(200 palabras, solape 20). Eso tiene dos limitaciones conocidas:

- **Desajuste semántico pregunta↔chunk.** El usuario escribe una *pregunta*
  ("¿cómo mitigo un XSS persistente?") pero los chunks son *texto expositivo*. Los
  vectores de una pregunta y de un párrafo que la responde no siempre quedan
  cerca.
- **Chunks opacos fuera de contexto.** Un fragmento que dice "esta configuración
  debe deshabilitarse" pierde el sujeto ("¿qué configuración? ¿de qué documento?")
  cuando se lo aísla del resto del texto.

El enriquecimiento ataca ambos: acerca la semántica del chunk a la de una pregunta
y le devuelve el contexto que el split le quitó.

## 2. Idea clave

> **Lo que se embeba no tiene por qué ser lo mismo que lo que se le pasa al LLM.**

Se enriquece el texto **que se vectoriza** (agregando metadatos generados), pero al
`PromptBuilder` le sigue llegando el `content` original. Así el vector mejora sin
ensuciar el contexto de generación ni el retriever léxico (BM25).

Meta-información que se agrega a cada chunk:

| Campo                    | Qué es                                              | Por qué ayuda a recuperar                                    |
|--------------------------|-----------------------------------------------------|-------------------------------------------------------------|
| `title`                  | Título corto del chunk (< 10 palabras)              | Ancla temática                                              |
| `summary`                | Resumen de 1–2 frases                               | Acerca la semántica del chunk a la de una pregunta         |
| `hypothetical_questions` | 3 preguntas que el chunk responde                   | **El más potente**: matchea pregunta-contra-pregunta        |
| `context`                | 1 frase que sitúa el chunk dentro del documento     | Desambigua chunks opacos (Contextual Retrieval de Anthropic)|

## 3. El mecanismo en Haystack: `meta_fields_to_embed`

`OllamaDocumentEmbedder` (como todos los embedders de Haystack) acepta
`meta_fields_to_embed`: antes de vectorizar, concatena esos campos del `meta` con
el `content`, pero **`doc.content` queda intacto** para el prompt.

```python
doc_embedder = OllamaDocumentEmbedder(
    model=self.valves.embedding_model,
    url=OLLAMA_URL,
    batch_size=32,
    meta_fields_to_embed=["title", "summary", "hypothetical_questions", "context"],
    embedding_separator="\n",
)
```

Esto cubre el "cómo se usa al indexar". Falta el "cómo se generan esos metadatos".

## 4. Dónde encaja en el flujo de indexación

El paso de enriquecimiento va **entre el split y el embedding**, dentro de
`_index_new_documents()`
([`pipeline_ciberseguridad.py:288`](../src/pipeline/pipeline_ciberseguridad.py#L288)),
justo después de filtrar `new_docs` (así solo se enriquecen chunks nuevos):

```
PDFs ─marker─▶ Markdown ─▶ DocumentSplitter ─▶ filtro new_docs
                                                     │
                                                     ▼
                                      _enrich_chunks (LLM Groq)  ◀── PASO NUEVO
                                                     │
                                                     ▼
                              OllamaDocumentEmbedder (meta_fields_to_embed)
                                                     │
                                                     ▼
                                              pgvector (vdb)
```

## 5. Componente de enriquecimiento (esbozo)

Genera los metadatos con una llamada al LLM de Groq por chunk, pidiendo JSON:

```python
ENRICH_TEMPLATE = """Sos un asistente que prepara fragmentos de documentos técnicos de
ciberseguridad para un sistema de búsqueda. Dado el fragmento, devolvé JSON con:
- "title": título breve (< 10 palabras)
- "summary": resumen de 1-2 frases
- "hypothetical_questions": lista de 3 preguntas que este fragmento responde

Fragmento:
{{ chunk }}

Respondé SOLO con el JSON."""

def _enrich_chunks(self, docs: list[Document]) -> list[Document]:
    import json
    enricher = OpenAIGenerator(
        api_key=Secret.from_env_var("GROQ_API_KEY"),
        api_base_url=GROQ_BASE_URL,
        model=self.valves.llm_model,
        generation_kwargs={"temperature": 0.0, "response_format": {"type": "json_object"}},
    )
    builder = PromptBuilder(template=ENRICH_TEMPLATE)
    for doc in docs:
        try:
            prompt = builder.run(chunk=doc.content)["prompt"]
            reply  = enricher.run(prompt=prompt)["replies"][0]
            meta   = json.loads(reply)
            doc.meta["title"]   = meta.get("title", "")
            doc.meta["summary"] = meta.get("summary", "")
            qs = meta.get("hypothetical_questions", [])
            doc.meta["hypothetical_questions"] = " ".join(qs) if isinstance(qs, list) else str(qs)
        except Exception as e:
            logger.warning(f"No se pudo enriquecer chunk {doc.id}: {e}")
    return docs
```

Y en `_index_new_documents()`, después de calcular `new_docs` y antes del embedding:

```python
logger.info(f"Enriqueciendo {len(new_docs)} chunks nuevos...")
new_docs = self._enrich_chunks(new_docs)
```

## 6. Consideraciones para este proyecto

1. **Costo / rate-limit de Groq.** El enriquecimiento es **1 llamada LLM por
   chunk**. Con `split_length=200` habrá muchos chunks por documento. Mitigantes:
   - Corre solo sobre `new_docs` (la indexación ya es incremental) → costo de una
     sola vez por documento.
   - Considerar un modelo chico/barato para esta tarea (p. ej.
     `llama-3.1-8b-instant`), distinto del de generación.
   - Vigilar el límite TPM de Groq; si hace falta, agregar `time.sleep` o batching.

2. **Guardar lo enriquecido en `meta`, NO en `content`.** Si se mete en `content`,
   se contamina el contexto del LLM y el `keyword_retriever` (BM25) empieza a
   matchear texto sintético. Con `meta` + `meta_fields_to_embed`, solo el embedding
   retriever lo aprovecha; el keyword y el prompt siguen sobre texto real.

3. **El campo `context` necesita el documento completo.** "Situar el chunk en el
   documento" (Contextual Retrieval de Anthropic) requiere pasarle al LLM el
   documento entero —o un resumen— además del chunk. Es el paso de mayor impacto
   pero también el más caro; conviene dejarlo como **segunda iteración**.

4. **Reindexado necesario.** Como cambia *lo que se embeba*, los chunks ya
   indexados no tienen los campos nuevos. Para una comparación limpia, vaciar la
   tabla `ciberseguridad_docs` y reindexar todo una vez.

## 7. Plan de adopción sugerido

1. **Iteración 1 (barata, alto impacto):** `title` + `summary` +
   `hypothetical_questions` vía `meta_fields_to_embed`. Reindexar y medir.
2. **Iteración 2 (Contextual Retrieval completo):** agregar `context` con el
   documento completo como referencia.
3. **Medición:** comparar los logs de `_log_retrieved_docs`
   ([`pipeline_ciberseguridad.py:156`](../src/pipeline/pipeline_ciberseguridad.py#L156))
   antes/después sobre un set fijo de preguntas, mirando qué fuentes y scores
   recupera cada retriever.

## 8. Referencias

- Anthropic — *Introducing Contextual Retrieval*:
  <https://www.anthropic.com/news/contextual-retrieval>
- Haystack — `meta_fields_to_embed` en los document embedders:
  <https://docs.haystack.deepset.ai/docs/ollamadocumentembedder>
