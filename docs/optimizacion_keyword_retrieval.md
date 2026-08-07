# Plan de decisión: optimización del keyword retrieval

## Contexto y problema

La búsqueda es híbrida: `PgvectorEmbeddingRetriever` (semántica) + `PgvectorKeywordRetriever`
(full-text) unidos por `DocumentJoiner` (reciprocal rank fusion) en
[pipeline_ciberseguridad.py](../src/pipeline/pipeline_ciberseguridad.py).

El análisis de los logs reales (`src/pipeline/logCiberseguridad.txt`) mostró **dos fallas que
se combinan** y dejan la mitad keyword inútil:

1. **El keyword retriever hace AND de todos los términos de la query.** Recibe el
   `user_message` crudo (la pregunta entera en español). Como exige que *todas* las palabras
   estén en un mismo documento, nunca matchea un doc CWE (en inglés) → **0 hits en toda
   pregunta real**. Verificado en vivo:

   | Query pasada al keyword retriever | hits |
   |---|---|
   | `CWE-89` | 1 ✓ |
   | `SQL injection` | 2 (CWE-564, CWE-89) ✓ |
   | `CWE-89 prevenirla` | 0 |
   | `Explicame la vulnerabilidad CWE-89 y cómo prevenirla.` | 0 |

2. **Los embeddings no mapean número de CWE → concepto.** `bge-m3` no sabe que CWE-89 =
   SQL Injection, así que preguntar por ID exacto (`"Explicame CWE-89"`) trae CWEs no
   relacionados (CWE-391/1295/290). El único branch que puede clavar un ID exacto es el
   keyword… que está roto por (1).

**Consecuencia:** hoy la búsqueda es *de facto* solo-embeddings, y las consultas por
identificador exacto (CWE-89, CVE-2023-…) fallan. El objetivo de este plan es decidir **cómo
transformar la query del usuario en un input efectivo para el keyword retriever**.

## Objetivo

Dado el `user_message`, producir una **query keyword** que:
- Conserve identificadores exactos (`CWE-\d+`, `CVE-\d{4}-\d+`) → aciertos determinísticos.
- Extraiga los términos/entidades relevantes (ej. "SQL injection", "buffer overflow", "XSS").
- Descarte stopwords, signos y relleno en español que rompen el AND del full-text.
- No degrade la latencia de forma inaceptable.

## Opciones a decidir

### Opción A — Análisis estático del string (sin LLM)
Preprocesado por reglas antes del keyword retriever:
- Regex para IDs: `CWE-\d+`, `CVE-\d{4}-\d+` (y normalización de variantes: "cwe 89" → "CWE-89").
- Quitar stopwords español/inglés + signos de puntuación e interrogación.
- Opcional: extracción de keyphrases (RAKE/YAKE, o simplemente n-gramas de sustantivos).
- Pasar el resultado como query keyword (idealmente con semántica OR, ver "Nota técnica").

**Pros:** latencia ~0, costo $0, determinístico, sin dependencia de red, fácil de testear.
**Contras:** frágil con sinónimos y lenguaje natural ("dejar credenciales en el código" no
llega a "CWE-798 Hardcoded Credentials"); mantener listas de stopwords; no capta intención.

### Opción B — Preprocesado con una llamada al LLM (Groq)
Una primera llamada barata (modelo chico, ej. `llama-3.1-8b-instant`) que reciba el
`user_message` y devuelva JSON estructurado, p. ej.:
```json
{ "intent": "explicar_y_prevenir",
  "cwe_ids": ["CWE-89"],
  "cve_ids": [],
  "keywords": ["SQL injection", "input validation", "prevention"] }
```
Los `keywords` (en inglés, para matchear los docs CWE) alimentan el keyword retriever; los IDs
van por el fast-path exacto.

**Pros:** entiende intención y sinónimos, traduce español→inglés (clave, porque los CWE están
en inglés), robusto ante frases largas; unifica extracción de IDs + keyphrases.
**Contras:** +1 llamada por query (latencia +~300-800 ms y costo/quota de Groq — ya vimos 429s
que motivaron `max_retries=5`); no determinístico; puede alucinar IDs (hay que validar contra
el store).

### Opción C — Híbrido (recomendada como default a evaluar)
- **Fast-path estático** siempre: regex de `CWE-\d+`/`CVE-…` → si hay IDs, keyword retriever
  exacto garantizado (barato, cubre el caso que hoy más falla).
- **LLM solo cuando aporta:** si NO hay IDs explícitos y la query es de lenguaje natural,
  llamar al LLM para extraer keywords/intención (en inglés).
- Cachear el preprocesado por query para no repetir la llamada (las tareas internas de
  OpenWebUI repiten prompts).

**Pros:** cubre lo determinístico gratis y reserva el costo del LLM para donde rinde.
**Contras:** más lógica y dos caminos que mantener/testear.

## Criterios de decisión

| Criterio | Peso | A (estático) | B (LLM) | C (híbrido) |
|---|---|---|---|---|
| Aciertos por ID exacto | alto | ✓ | ✓ (si valida) | ✓ |
| Robustez ante lenguaje natural / sinónimos | alto | ✗ | ✓ | ✓ |
| Traducción es→en (docs CWE en inglés) | alto | ✗ | ✓ | ✓ |
| Latencia | medio | ✓✓ | ✗ | ~ |
| Costo / quota Groq (riesgo 429) | medio | ✓✓ | ✗ | ~ |
| Determinismo / testeabilidad | medio | ✓✓ | ✗ | ~ |

**Recomendación:** empezar por **A** para el fast-path de IDs (resuelve ya el caso más
sangrante con costo cero) y evaluar sumar **B** solo para queries sin ID — es decir, converger
a **C**. Decisión final pendiente de: (a) tolerancia de latencia del equipo, (b) presupuesto de
llamadas a Groq, (c) cuánto pesa el retrieval por concepto en español vs por ID.

## Nota técnica: además del preprocesado

Aunque se extraigan bien las keywords, conviene revisar el **modo de match** del
`PgvectorKeywordRetriever`. Hoy el full-text ANDea; con OR (o `websearch_to_tsquery`) varias
keywords sueltas matchearían aunque no estén todas en el mismo doc. Y la config de idioma del
`to_tsvector` importa: los docs son mixtos (CWE en inglés, PDFs INCIBE en español). Evaluar
indexar/consultar con la config de idioma adecuada o `simple`.

## Verificación (para cualquiera de las opciones)

Reusar el banco de queries que ya ejercitamos contra el store real
(`ciberseguridad_docs`, 965 docs):
1. **Por ID exacto:** `"Explicame CWE-89"`, `"riesgos de CWE-798"` → el keyword retriever debe
   traer el doc CWE correcto en el top-k (hoy da 0).
2. **Por concepto en español:** `"inyección SQL"`, `"buffer overflow"`, `"credenciales en el
   código"` → deben aparecer los CWE correspondientes junto a los PDFs.
3. **Regresión:** las preguntas 100% conceptuales (XSS/5G/cloud) no deben empeorar.
4. Medir en el log el bloque `[KEYWORD RETRIEVER]`: dejar de ver `0 documentos recuperados` en
   consultas con términos indexados.

## Relacionado (mismo análisis de logs, fuera de este plan)
- Filtrar chunks de PDF casi-vacíos (~14% son whitespace de tablas de marker; umbral
  `alnum < 20`, el mismo que ya usa el `XMLCWEConverter`).
- Subir `retriever_top_k` (3 → 6-8) para que CWEs y PDFs convivan en el top-k.
