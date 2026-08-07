# Análisis: el keyword retriever trae siempre 0 documentos

## Síntoma

En el pipeline de RAG ([src/pipeline/pipeline_ciberseguridad.py](../src/pipeline/pipeline_ciberseguridad.py)),
el `PgvectorKeywordRetriever` devuelve `0` documentos en casi todas las queries,
mientras que el `PgvectorEmbeddingRetriever` (búsqueda densa) sí recupera 3.

## Evidencia (logs reales)

De [src/pipeline/logCiberseguridad.txt](../src/pipeline/logCiberseguridad.txt):

| Query | Keyword retriever |
|-------|-------------------|
| `QUe es la ciberseguridad?` | **3 documentos** ✅ |
| `¿Cuáles son las principales amenazas de ciberseguridad en las redes 5G?` | **0 documentos** ❌ |
| `¿Cómo pueden las empresas proteger sus redes 5G...?` | **0 documentos** ❌ |
| Meta-prompts de OpenWebUI (título, tags, follow-ups en inglés) | **0 documentos** ❌ |

El dato clave: **no es que siempre traiga 0** — con una query corta (`Que es la ciberseguridad`)
trajo 3. Con preguntas más largas, siempre 0. Eso apunta directo a la causa.

## Causa raíz

El problema es la combinación de **dos** factores en cómo `pgvector-haystack` implementa
la búsqueda por palabras clave. La consulta SQL que usa (confirmado en el código del paquete
instalado, `document_stores/pgvector/document_store.py`) es:

```sql
SELECT *, ts_rank_cd(to_tsvector(language, content), query) AS score
FROM tabla, plainto_tsquery(language, %s) query
WHERE to_tsvector(language, content) @@ query
ORDER BY score DESC
LIMIT top_k
```

### Factor 1 — `language` está en `"english"` (el default)

En [`_get_document_store()`](../src/pipeline/pipeline_ciberseguridad.py#L198) creamos el
`PgvectorDocumentStore` **sin** pasar `language`, y el default del paquete es:

```python
language: str = "english"
```

Postgres usa esa configuración de Full-Text Search tanto para **indexar** (índice GIN
`to_tsvector('english', content)`) como para **consultar** (`plainto_tsquery('english', ...)`).
Al aplicar el diccionario **inglés** sobre texto en **español**:

- **No se eliminan las stopwords en español.** Palabras como *cuáles, son, las, de, en, la, que*
  no son stopwords en inglés, así que **se conservan como lexemas obligatorios** de la búsqueda.
- **El stemming es el inglés**, que no unifica variantes del español
  (*amenazas* ≠ *amenaza*, *redes* ≠ *red*), reduciendo aún más las coincidencias.

### Factor 2 — `plainto_tsquery` une TODOS los términos con AND (`&`)

`plainto_tsquery` genera una tsquery donde **todos** los lexemas van unidos con `&` (AND).
Es decir, para que un chunk haga match, **debe contener todas y cada una** de las palabras
de la pregunta (ya lematizadas). El `WHERE ... @@ query` es un filtro duro: si falta un solo
lexema, el documento no aparece.

### Por qué esto explica exactamente los logs

Los chunks son de ~200 palabras ([`split_length=200`](../src/pipeline/pipeline_ciberseguridad.py#L95)).

- `Que es la ciberseguridad` → lexemas (config inglesa): `que & es & la & ciberseguridad`.
  *que*, *la* aparecen en casi todos los chunks en español, y *ciberseguridad* es el tema del
  corpus → **hay chunks que contienen los 4 → 3 resultados**.
- `¿Cuáles son las principales amenazas de ciberseguridad en las redes 5G?` → se convierte en
  algo como `cual & son & las & principal & amenaza & de & ciberseguridad & en & red & 5g`.
  Que **un único chunk de 200 palabras** contenga simultáneamente *todos* esos términos es
  casi imposible → **0 resultados**.

Cuantas más palabras tiene la pregunta, más restrictivo es el AND y más probable es el 0.
Ese es exactamente el patrón observado.

> **Ruido adicional:** los "meta-prompts" que OpenWebUI manda por el mismo `pipe()` (generar
> título, tags y follow-ups) son textos largos en inglés. Pasan por los retrievers y, como son
> enormes, el AND garantiza 0 en keyword. No son queries reales del usuario, pero ensucian el log
> y gastan cómputo.

## El rol del Document Joiner (por qué "tapa" el problema)

El [`DocumentJoiner`](../src/pipeline/pipeline_ciberseguridad.py#L348) usa
`join_mode="reciprocal_rank_fusion"`. Esto es importante para entender por qué el bug
pasó desapercibido y qué implica:

1. **El joiner no falla ni se queja.** Cuando keyword devuelve 0, RRF simplemente fusiona
   `0 + 3` documentos. El prompt igual recibe los 3 del retriever denso, la respuesta sale
   coherente, y por eso "parece que anda". **La búsqueda híbrida se degrada silenciosamente
   a búsqueda puramente densa**: perdemos todo el aporte del canal léxico (matching exacto de
   siglas/términos como *IPsec*, *EAP-AKA'*, *GDPR*, *5G*, nombres propios) que es justamente
   donde el keyword retriever debería brillar frente a los embeddings.

2. **El tamaño del contexto es inestable.** El joiner no tiene `top_k` fijo, así que devuelve
   *todos* los documentos únicos fusionados: **6** cuando keyword aporta 3 (caso línea 99 del log)
   y **3** cuando aporta 0. El prompt al LLM cambia de tamaño según si el keyword matcheó o no.

3. **RRF ignora los scores crudos** (usa el ranking), lo cual es correcto — pero también significa
   que los scores minúsculos del keyword (0.026, 0.006 en el log) no son el problema; el problema
   es aguas arriba, en cuántos documentos entran al joiner.

**Conclusión sobre el joiner:** está funcionando bien; el defecto es del keyword retriever.
Pero conviene, al arreglar el keyword, fijar un `top_k` en el joiner para que el prompt reciba
una cantidad estable de documentos.

## Soluciones (priorizadas)

### 1. (Principal) Configurar `language="spanish"` en el DocumentStore

En [`_get_document_store()`](../src/pipeline/pipeline_ciberseguridad.py#L198):

```python
return PgvectorDocumentStore(
    connection_string=Secret.from_token(DB_CONNECTION),
    embedding_dimension=EMBEDDING_DIMENSION,
    table_name=DB_TABLE,
    keyword_index_name="ciberseguridad_keyword_index",
    language="spanish",   # <-- FTS en español: quita stopwords y lematiza en español
)
```

`spanish` es una configuración de FTS que viene por defecto en PostgreSQL. Esto:
- elimina las stopwords españolas de la query (se van *cuáles, son, las, de, en...*),
  dejando solo los términos con contenido → el AND deja de ser imposible de satisfacer;
- lematiza en español (*amenazas*→*amenaza*, *redes*→*red*).

> ⚠️ **Importante — hay que recrear el índice GIN.** El índice de keyword ya existente fue
> construido con `to_tsvector('english', content)`. Cambiar solo el parámetro `language` hace que
> la query use `spanish` pero el índice siga en `english` (no se usará o no matcheará). Opciones:
> - **DROP del índice** de keyword en Postgres y dejar que el store lo recree en español, o
> - re-inicializar la tabla (`recreate_table=True` una vez, que borra los datos → hay que
>   re-indexar los documentos), o
> - dropear tabla/índice manualmente y correr de nuevo la indexación.
>
> Como los embeddings ya están calculados, lo más barato es dropear **solo el índice de keyword**
> y que se regenere; no hace falta recalcular embeddings.

### 2. (Complementaria) Fijar `top_k` en el DocumentJoiner

Para que el prompt reciba una cantidad estable de documentos independientemente de cuántos aporte
cada canal:

```python
pipeline.add_component("document_joiner", DocumentJoiner(
    join_mode="reciprocal_rank_fusion",
    top_k=self.valves.retriever_top_k,   # p.ej. 3–5 documentos finales
))
```

### 3. (Opcional) No mandar los meta-prompts de OpenWebUI al RAG

Los prompts internos de OpenWebUI (título, tags, follow-ups) llegan a `pipe()` y disparan
recuperación innecesaria. Se pueden detectar (empiezan con `### Task:`) y responder sin pasar por
los retrievers, ahorrando cómputo y limpiando el log. Es una mejora aparte, no arregla el bug.

### 4. (Evaluar) Suavizar la lógica AND de la query

`plainto_tsquery` (AND estricto) lo impone el paquete y no es configurable desde nuestro código
sin subclasear el retriever. Si tras el fix con `spanish` las preguntas largas siguen siendo
demasiado restrictivas, una opción es pre-procesar la query (extraer solo términos clave /
entidades antes de pasarla a `keyword_retriever`) para reducir el número de términos AND-eados.
Con el cambio de idioma esto suele no ser necesario, porque las stopwords ya se eliminan solas.

## Cómo verificar el arreglo

1. Aplicar la solución 1 (y recrear el índice de keyword).
2. Repetir la query larga: `¿Cuáles son las principales amenazas de ciberseguridad en las redes 5G?`.
3. En [logCiberseguridad.txt](../src/pipeline/logCiberseguridad.txt) el bloque
   `[KEYWORD RETRIEVER]` debe mostrar > 0 documentos, y `[DOCUMENT JOINER]` una cantidad estable.

## Resumen

| | |
|---|---|
| **Causa raíz** | FTS en `english` (default) sobre corpus en español → no se filtran stopwords españolas; sumado a que `plainto_tsquery` une todos los términos con AND, las preguntas largas exigen que un mismo chunk contenga todas las palabras → 0 resultados. |
| **Por qué a veces trae 3** | Queries cortas (*"Que es la ciberseguridad"*) tienen pocos términos, y esos términos son ubicuos en el corpus, así que el AND se satisface. |
| **Rol del joiner** | Correcto, pero enmascara el fallo: RRF degrada la búsqueda híbrida a solo-densa sin avisar, y el contexto queda de tamaño variable. |
| **Fix principal** | `language="spanish"` en el `PgvectorDocumentStore` + recrear el índice GIN de keyword. |
