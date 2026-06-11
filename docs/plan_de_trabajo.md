# Plan de trabajo definitivo

**Plan de Trabajo - Grupo Niños RAGtas**

**Tema**

Sistema RAG consultor de ciberseguridad en lenguaje natural. El sistema responde preguntas sobre vulnerabilidades, debilidades y amenazas a partir de un corpus heterogéneo de fuentes autoritativas (CVE/NVD, CWE, MITRE ATT&CK, EPSS/CISA KEV, documentos NIST, advisories), con citación obligatoria en cada respuesta y generación con LLM 100% local. La heterogeneidad del corpus es el challenge de ingeniería central del proyecto. Módulo secundario (nice-to-have): análisis de vulnerabilidades en código Python.

**Integrantes del grupo**

- Luca Giordano
- Francisco Suarez
- Valentino Turconi
- Miguel Miesionznik

**Corpus**

El corpus está justificado técnicamente en tres capas de dificultad creciente:

**Capa 1 — Datos estructurados / semi-estructurados (API-accesibles, actualizables):**

- NVD / CVE (JSON desde [nvd.nist.gov](http://nvd.nist.gov)): ~250k entradas con campos fijos. Formato ideal para evaluar chunking de datos estructurados.
- MITRE ATT&CK (JSON v15): jerarquía táctica/técnica/sub-técnica/mitigación/detección. Permite evaluar si el chunking preserva relaciones jerárquicas.
- CWE (XML, [mitre.org](http://mitre.org)): debilidades de software con categorías y relaciones.
- EPSS + CISA KEV (feeds diarios): probabilidad de explotación y vulnerabilidades explotadas confirmadas. Habilitan queries de priorización real.

**Capa 2 — Documentos no estructurados (PDFs/HTML):**

- NIST Special Publications (SP 800-x): guías normativas en prosa técnica densa, con referencias cruzadas.
- CISA Advisories: alertas sobre amenazas activas, formato semi-estructurado con IOCs.
- Threat reports públicos (Mandiant, CrowdStrike, Recorded Future): narrativa larga con ATT&CK mappings embebidos.

**Capa 3 — Fuentes técnicas específicas (mayor desafío de parsing):**

- Exploit-DB: código + descripción mezclados. Permite evaluar si el chunker separa correctamente código y prosa.
- Security Stack Exchange (dump público): pares pregunta/respuesta útiles para construir ground truth sintético.

La heterogeneidad de estas tres capas **es el desafío de ingeniería**.

**Tipos de preguntas**

Taxonomía de queries que el sistema debe cubrir, con complejidad creciente:

| Tipo | Ejemplo | Desafío RAG principal |
| --- | --- | --- |
| **Factual** | "¿Qué CVSS score tiene CVE-2024-21762?" | Recuperación precisa de entidad puntual; riesgo de stale data |
| **Relacional** | "¿Qué técnicas ATT&CK usa el grupo APT29?" | Multi-hop: actor → técnica → sub-técnica |
| **Comparativo** | "¿Cuál es la diferencia entre CWE-79 y CWE-89?" | Recuperar dos chunks distintos y contrastarlos coherentemente |
| **Procedimental** | "¿Cómo mitigamos T1078 (Valid Accounts)?" | Mitigaciones + contexto de detección; respuesta multi-parte |
| **Contextual** | "Tengo Windows con RDP expuesto, ¿qué priorizo?" | Query expansion necesaria; múltiples fuentes |
| **Temporal** | "¿Qué CVEs críticas afectan Apache en los últimos 30 días?" | Temporal retrieval + recency ranking |

**Objetivos**

- Reducir el knowledge-cutoff del LLM: respuestas actualizadas y fundamentadas en fuentes autoritativas, con citación obligatoria en cada respuesta (anti-alucinación).
- Cubrir la taxonomía completa de queries de la tabla anterior.
- Demostrar mejora cuantitativa iteración a iteración en retrieval, calidad E2E y velocidad.
- Operar enteramente **local**, preservando la **confidencialidad** de las consultas.

**Desafíos**

- **Chunking heterogéneo:** una entrada CVE es atómica (no debe dividirse); un NIST SP de 300 páginas requiere segmentación semántica. El mismo chunker no puede aplicarse uniformemente → se necesita un *document router* que clasifique el tipo de fuente y aplique la estrategia correspondiente. Este problema no tiene estándar en la literatura: definir métricas intrínsecas de calidad de chunk (boundary precision, information density, self-containedness) es el **aporte académico del proyecto**.
- **Temporal drift / index staleness:** los CVEs se publican a diario y el knowledge base puede volverse incorrecto en semanas → requiere pipeline de ingesta continua. El RAG debe seguir respondiendo correctamente con vulnerabilidades nuevas no vistas en el pretraining del modelo.
- **Terminología multi-vocabulario:** el mismo concepto aparece con siglas, IDs (CVE-2024-XXXX, T1078, CWE-89) y en prosa narrativa. Los embeddings genéricos fallan en exact-match de identificadores → hybrid retrieval (BM25 + denso) desde el baseline.
- **Relaciones inter-documento (multi-hop):** la cadena CVE → producto afectado → mitigación (NIST SP) → técnica ATT&CK no es capturada por ningún chunk individual plano.
- **Evaluación sin contaminación paramétrica:** NVD y ATT&CK ya están en el pretraining de los LLMs. Para separar conocimiento recuperado de conocimiento paramétrico, el ground truth se construye con CVEs publicadas después del cutoff del modelo.
- **LLM 100% local (restricción de diseño):** un consultor de ciberseguridad recibe queries con datos potencialmente sensibles (infraestructura, configs, vulnerabilidades expuestas) → enviarlas a una API en la nube es inaceptable por confidencialidad. Un modelo de 7B tiene menor razonamiento y mayor tendencia a alucinar que modelos cloud; se compensa con retrieval fuerte, citación obligatoria y abstención calibrada. Implica: ventana de contexto acotada (top-k chico, reranking importa más), throughput medible sobre hardware modesto (tokens/s y P50/P95 son métricas de primer orden), y cuantización Q4/Q5 como eje calidad-velocidad.
- **Prompt injection vía corpus:** si el corpus incluye texto no curado (ej: descripciones de Exploit-DB), hay superficie de ataque real. OWASP GenAI Top 10 lista Prompt Injection como LLM01:2025, explicitando que RAG y fine-tuning no eliminan este riesgo.
- **Confidencialidad y data leakage:** si el sistema se extiende a documentos internos de una organización (políticas, configs), el riesgo de filtrar información entre sesiones o usuarios es no trivial. RAG no aísla automáticamente el acceso por rol.
- **Simplificación de salida:** el sistema debe poder explicar conceptos técnicos complejos (CVSSv3, CWE, tácticas ATT&CK) en lenguaje comprensible e indicar acciones concretas a tomar.

**Cronograma**

- Selección y preprocesamiento de los datos (15/Jun)
    - Corpus Capa 1: NVD/CVE (JSON), CWE (XML), MITRE ATT&CK (JSON v15), EPSS + CISA KEV.
    - Corpus Capa 2: NIST SP, CISA Advisories, threat reports públicos.
    - Parsers separados por tipo de fuente (base del document router).
    - (Secundario) Dataset PythonSecurityEval para el módulo de código.
- Primer prototipo E2E (30/Jun)
    - Pipeline completo: retrieval híbrido BM25 + embeddings bge-m3 con fusión RRF, índice en pgvector (Postgres), generación con LLM local vía Ollama (candidatos: Qwen3 7B o Llama 3.3 8B, cuantización Q4_K_M).
- Tests automatizados para medir rendimiento (15/Jul). Pensamos medir:
    - Velocidad: tokens/segundo (throughput) y latencia P50/P95 por etapa (chunker, embedder, retriever, LLM).
    - Relevancia del retrieval: Hit Rate@k, MRR y nDCG sobre un ground truth sintético de CVEs publicadas después del cutoff del modelo (anti-contaminación paramétrica).
    - Calidad E2E: faithfulness y answer relevance con RAGAS (Es et al., arXiv 2309.15217).
    - Calidad intrínseca del chunking: boundary precision e information density (entidades nombradas / tokens, vía spaCy), sin pasar por el pipeline completo.
    - Sanity check de dominio: accuracy sobre CyberMetric (arXiv 2402.07688) o CTIBench (arXiv 2406.07599).
- Mejora 1 — chunking semántico con document router (30/Jul)
    - Reemplazar fixed-size por chunking semántico con estrategia diferenciada por tipo de fuente (JSON estructurado vs prosa técnica vs advisories). Comparación cuantitativa contra el baseline.
- Mejora 2 — cross-encoder reranker (15/Ago)
    - Agregar un reranker tras el retrieval híbrido para reordenar los candidatos recuperados. Medir delta en Recall@5 y nDCG.
- Mejora 3 — priorización por explotabilidad real (30/Ago)
    - Incorporar EPSS + CISA KEV al razonamiento del sistema para responder queries de priorización ("¿qué vulnerabilidades debo atender primero?"). Evaluar contra casos donde CVSS solo no alcanza.
- (Si hay tiempo) Módulo de código Python: Bandit como SAST + RAG para explicación, contexto CWE y remediación citada. Evaluar con precision/recall/F1 sobre PythonSecurityEval.
- Práctica de la demo para expo-ciencia (2/Oct) ← fecha fija para todos los grupos; las anteriores pueden ajustarse según qué etapas lleven más tiempo.

---

## Versión humanizada

**Plan de Trabajo - Grupo Niños RAGtas**

**Tema**

Queremos construir un sistema que funcione como consultor de ciberseguridad: le hacés una pregunta en lenguaje natural ("¿qué vulnerabilidades críticas afectan a Apache este mes?", "¿cómo mitigamos este tipo de ataque?") y te responde con información actualizada, citando de dónde sacó cada dato.

El problema que resuelve es concreto: los modelos de lenguaje tienen una fecha de corte de conocimiento y no saben nada de vulnerabilidades publicadas después de eso. Nosotros lo conectamos a fuentes que se actualizan a diario (como el catálogo de CVEs del NIST o las alertas de CISA), así el sistema siempre trabaja con información fresca.

Todo corre en local, sin mandar datos a ninguna API en la nube. Eso es importante porque un consultor de seguridad recibe preguntas sobre infraestructura propia, configs internas y vulnerabilidades expuestas — ese tipo de información no debería salir de la organización.

**Integrantes del grupo**

- Luca Giordano
- Francisco Suarez
- Valentino Turconi
- Miguel Miesionznik

**De dónde vienen los datos**

Las fuentes que le damos al sistema no son arbitrarias. Las organizamos en tres capas según qué tan difícil son de procesar:

La primera capa son los datos más estructurados: el catálogo de CVEs del NIST (más de 250.000 vulnerabilidades en formato JSON), el framework ATT&CK de MITRE (que mapea tácticas y técnicas de los atacantes), el catálogo CWE de debilidades de software, y dos feeds que nos dicen cuáles vulnerabilidades se están explotando actualmente en el mundo real (EPSS y CISA KEV). Son los más fáciles de ingerir y los que más valor aportan.

La segunda capa son documentos en prosa: las guías técnicas del NIST (que pueden tener 300 páginas), las alertas de CISA, y reportes de threat intelligence de empresas como Mandiant o CrowdStrike. Son más difíciles de procesar que un JSON, pero dan contexto que los datos estructurados solos no tienen.

La tercera capa son las más complejas: Exploit-DB mezcla código de exploit con descripción en el mismo documento, y Stack Exchange tiene un formato de pregunta/respuesta que además nos sirve para armar ejemplos de evaluación.

Lo interesante de este corpus es que cada tipo de documento necesita una estrategia de procesamiento distinta — un CVE es una entrada atómica que no se puede dividir, pero una guía NIST de 300 páginas hay que segmentarla de forma inteligente. Ese problema de "¿cómo procesás fuentes tan distintas de forma consistente?" es el aporte técnico central de nuestra investigación.

**Qué tipo de preguntas tiene que poder responder**

Pensamos en seis tipos de consultas con dificultad creciente:

Las más simples son las factuales: el usuario pide un dato puntual ("¿qué score CVSS tiene tal CVE?"). El desafío acá es la precisión y que el dato no esté desactualizado.

Las relacionales requieren conectar información de varias fuentes: "¿qué técnicas de ataque usa el grupo APT29?" implica buscar el actor, sus técnicas, y las sub-técnicas asociadas — todo en documentos distintos.

Las comparativas necesitan traer dos chunks distintos y contrastarlos: "¿cuál es la diferencia entre XSS e inyección SQL?".

Las procedimentales son las más útiles en la práctica: "¿cómo mitigamos esta técnica de ataque?" requiere juntar la descripción del ataque, las mitigaciones recomendadas y el contexto de detección.

Las contextuales son las más complejas: el usuario describe su situación ("tengo Windows con RDP expuesto") y el sistema tiene que entender qué está preguntando realmente antes de buscar.

Las temporales agregan la variable tiempo: "¿qué CVEs críticas salieron en los últimos 30 días?", lo que requiere ordenar los resultados por fecha.

**Objetivos**

Básicamente queremos demostrar cuatro cosas con este proyecto: que podemos darle al modelo información actualizada que no tiene en su entrenamiento y que el rendimiento del mismo se mantenga al incorporar nuevos datos, que el sistema puede responder los seis tipos de preguntas que describimos, y que todo funciona sin depender de ningún servicio externo.

**Desafíos**

El desafío más interesante — y el que creemos que tiene valor académico — es el de procesar fuentes tan heterogéneas de forma coherente. No existe hoy una métrica estándar para evaluar si un fragmento de texto está bien dividido. Nosotros proponemos dos: *boundary precision* (si el corte respeta límites semánticos reales) e *information density* (cuánta información útil hay por cada token). Eso es algo que la literatura todavía no tiene resuelto.

El modelo corriendo en local también es un desafío real: un modelo de 7B alucina más que uno de 70B o que GPT-4. La solución no es ignorarlo sino compensarlo: cada respuesta debe citar sus fuentes, y si el sistema no encuentra evidencia suficiente, tiene que decirlo en lugar de inventar.

Un riesgo que no queremos ignorar es el de prompt injection: si el corpus incluye texto de Exploit-DB (que puede contener instrucciones maliciosas embebidas), ese texto podría intentar manipular al modelo. Es el vector de ataque #1 en sistemas RAG según OWASP 2025.

**Cronograma**

- Selección y preprocesamiento de los datos (15/Jun): armamos los parsers para cada tipo de fuente y cargamos el corpus inicial.
- Primer prototipo completo (30/Jun): un sistema que funciona de punta a punta — desde la pregunta hasta la respuesta con citas — aunque no esté optimizado.
- Tests automatizados (15/Jul). Medimos cuatro cosas:
    - qué tan rápido responde (tokens por segundo, latencia por etapa)
    - qué tan bien recupera los documentos relevantes (Hit Rate, MRR, nDCG)
    - qué tan buenas son las respuestas (usando RAGAS, que no necesita respuestas de referencia)
    - qué tan bien están divididos los textos (con las métricas de chunking que propusimos). Para los tests de retrieval usamos solo CVEs publicadas después de la fecha de corte del modelo, así no podemos hacer trampa.
- Adaptamos los componentes del RAG según los hallazgos otorgados por los tests (30/Jul).
- Nuestra idea es lograr incorporar la funcionalidad de análisis de código y no solo chat en lenguaje natural.
    - El RAG recibe código y detecta vulnerabilidades.
- Demo expo-ciencia (2/Oct): fecha fija para todos los grupos.