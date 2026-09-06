"""
title: Rag de Ciberseguridad (Inferencia)
author: Valentino, Miguel y Luca
version: 2.0 (Arquitectura Desacoplada)
requirements: haystack-ai, pgvector-haystack, ollama-haystack, sentence-transformers
"""

import torch as _torch
import torch.nn.utils.rnn as _rnn

# Patch 1: is_bf16_supported(including_emulation=...) agregado en torch 2.4
if hasattr(_torch.cuda, 'is_bf16_supported'):
    _orig_bf16 = _torch.cuda.is_bf16_supported
    def _patched_bf16(*args, **kwargs):
        kwargs.pop('including_emulation', None)
        return _orig_bf16(*args, **kwargs)
    _torch.cuda.is_bf16_supported = _patched_bf16

# Patch 2: pad_sequence(padding_side=...) agregado en torch 2.5
_orig_pad_sequence = _rnn.pad_sequence
def _patched_pad_sequence(sequences, batch_first=False, padding_value=0.0, **kwargs):
    kwargs.pop('padding_side', None)
    return _orig_pad_sequence(sequences, batch_first=batch_first, padding_value=padding_value, **kwargs)
_rnn.pad_sequence = _patched_pad_sequence
_torch.nn.utils.rnn.pad_sequence = _patched_pad_sequence

from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import logging
import os
import re

# Configurar el logger específico para nuestra app
logger = logging.getLogger("HaystackRAG_Query")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    file_handler = logging.FileHandler('/app/pipelines/logCiberseguridad.txt')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

from haystack import Pipeline as HaystackPipeline, Document, component
from haystack.utils import Secret
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.retrievers.pgvector import (
    PgvectorEmbeddingRetriever,
    PgvectorKeywordRetriever,
)
from haystack.components.builders import PromptBuilder
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.generators.ollama import OllamaGenerator

# --- Configuración fija de infraestructura ---
DB_CONNECTION       = "postgresql://avdbuser:avdbpass@vdb:5432/pgvdb"
OLLAMA_URL          = "http://ollama:11434"
DB_TABLE            = "ciberseguridad_docs"
EMBEDDING_DIMENSION = 2560  # nativa de qwen3-embedding:4b (sin truncar vía MRL)
DEFAULT_OLLAMA_LLM  = "qwen2.5:3b-instruct"

PROMPT_TEMPLATE = """
You are a cybersecurity assistant. Answer strictly from the provided context.

The context may contain three kinds of sources:
- CWE entries (MITRE): generic weakness *classes*, identified as CWE-<number>.
- CVE entries (NVD): specific, dated vulnerabilities in concrete products, identified as
  CVE-YYYY-NNNN, often with a CVSS score, severity and affected vendors/products.
- INCIBE guides: prose from cybersecurity best-practice reports.

Rules:
- Do NOT confuse a CWE (category) with a CVE (concrete instance).
- Cite the exact identifier (CWE-x / CVE-x) whenever a statement comes from such an entry.
- NEVER invent CVE/CWE identifiers, CVSS scores or severities. Use only values present in the context.
- If the context does not contain the answer, say explicitly that you don't know.
- Answer in the language of the question, NOT the language of the documents.

Context:
{% for document in documents %}
- {% if document.meta.cve_id %}[{{ document.meta.cve_id }}]{% if document.meta.severity %} severity={{ document.meta.severity }}{% endif %}{% if document.meta.cvss_v3_score %} CVSS={{ document.meta.cvss_v3_score }}{% endif %}{% if document.meta.cwe_ids %} (related: {{ document.meta.cwe_ids | join(", ") }}){% endif %}{% if document.meta.products %} affects: {{ document.meta.products | join(", ") }}{% endif %}
  {% endif %}{{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""

# ======================================================================
# Detector estático de keywords para el retriever full-text
# ======================================================================
_CWE_ID_RE = re.compile(r"cwe[\s\-_]?(\d{1,5})", re.IGNORECASE)
_CVE_ID_RE = re.compile(r"cve[\s\-_]?(\d{4})[\s\-_]?(\d{4,7})", re.IGNORECASE)

_SECURITY_TERMS = (
    "cross-site scripting", "cross site scripting", "xss",
    "cross-site request forgery", "csrf", "ssrf",
    "sql injection", "inyección sql", "inyeccion sql", "sqli",
    "command injection", "code injection", "os command injection",
    "buffer overflow", "desbordamiento de búfer", "desbordamiento de buffer",
    "integer overflow", "use after free", "null pointer",
    "remote code execution", "rce", "xxe", "xml external entity",
    "path traversal", "directory traversal", "lfi", "rfi",
    "open redirect", "idor", "insecure deserialization", "deserialization",
    "privilege escalation", "escalada de privilegios",
    "authentication bypass", "hardcoded credentials", "credenciales",
    "information disclosure", "sensitive information", "información sensible",
    "denial of service", "ddos", "race condition",
    "clickjacking", "phishing", "ransomware", "malware",
    "man-in-the-middle", "mitm", "spoofing",
)

_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "a", "ante",
    "con", "contra", "en", "entre", "hacia", "hasta", "para", "por", "según", "sin", "sobre",
    "tras", "y", "o", "u", "e", "que", "qué", "cual", "cuál", "cuales", "cuáles", "como",
    "cómo", "cuando", "cuándo", "donde", "dónde", "es", "son", "ser", "está", "están", "hay",
    "tiene", "tienen", "puede", "pueden", "más", "muy", "me", "te", "se", "le", "lo", "su",
    "sus", "mi", "tu", "este", "esta", "esto", "estos", "estas", "eso", "existen", "tipos",
    "explicame", "explícame", "dame", "decime", "contame", "quiero", "saber", "prevenirla",
    "prevenir", "acerca", "información", "informacion", "the", "a", "an", "of", "to", "in", 
    "on", "for", "and", "or", "is", "are", "be", "what", "which", "how", "when", "where", 
    "why", "can", "could", "does", "do", "about", "tell", "give", "explain", "types", "exist",
    "vulnerabilidad", "vulnerabilidades", "vulnerability", "descripcion", "descripción",
    "description", "cwe", "cve", "weakness", "debilidad",
}

def _extract_vuln_ids(text: str) -> list[str]:
    """IDs en ORDEN DE APARICIÓN en el texto, no agrupados por tipo.

    Antes se recorrían todos los CWE y después todos los CVE, así que el orden
    reflejaba el tipo y no la pregunta: "¿qué CVE explota CWE-502?" ponía siempre
    el CWE primero. El orden importa porque decide el desempate y el corte por
    top_k en VulnIdLookup (y, vía RRF, cuánto pesa cada uno).
    """
    found: list[tuple[int, str]] = []
    for m in _CWE_ID_RE.finditer(text):
        found.append((m.start(), f"CWE-{int(m.group(1))}"))
    for m in _CVE_ID_RE.finditer(text):
        found.append((m.start(), f"CVE-{m.group(1)}-{m.group(2)}"))
    found.sort(key=lambda t: t[0])
    seen, out = set(), []
    for _, vid in found:
        if vid not in seen:
            seen.add(vid)
            out.append(vid)
    return out


# Marcadores de los prompts internos de OpenWebUI (generación de títulos, tags...).
# No se heredan IDs desde ellos: traen <chat_history> embebido y contaminarían la query.
_META_PROMPT_MARKERS = ("### Task:", "<chat_history>")
_HISTORY_TURNS = 4   # ~2 intercambios hacia atrás
_MAX_VULN_IDS  = 6


def _message_text(msg: dict) -> str:
    """El content de OpenWebUI puede ser str o una lista de partes (multimodal)."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def resolve_vuln_ids(
    user_message: str,
    messages: list[dict] | None = None,
    history_turns: int = _HISTORY_TURNS,
    max_ids: int = _MAX_VULN_IDS,
) -> tuple[list[str], str]:
    """IDs del turno actual; si no hay ninguno, los hereda de los últimos turnos.

    El turno actual manda: sólo se mira hacia atrás cuando no trae ningún ID, así
    un cambio explícito de tema nunca queda contaminado. Dentro del historial se
    acumulan del más reciente al más viejo, y ese orden ES el peso: VulnIdLookup
    lo respeta y el RRF le da 1/(k+rango) a cada documento.

    Devuelve (ids, origen); `origen` se loguea para poder auditar el arrastre.
    """
    current = _extract_vuln_ids(user_message)
    if current:
        return current[:max_ids], "turno"
    if not messages:
        return [], "ninguno"

    inherited: list[str] = []
    for msg in reversed(messages[-history_turns:]):
        text = _message_text(msg)
        if not text or text.strip() == user_message.strip():
            continue  # OpenWebUI repite el turno actual al final de messages
        if any(marker in text for marker in _META_PROMPT_MARKERS):
            continue
        for vid in _extract_vuln_ids(text):
            if vid not in inherited:
                inherited.append(vid)
    if not inherited:
        return [], "ninguno"
    return inherited[:max_ids], "historial"

def _find_security_terms(text_low: str) -> list[str]:
    found = []
    for term in _SECURITY_TERMS:
        if " " in term or "-" in term:
            if term in text_low:
                found.append(term)
        elif re.search(rf"\b{re.escape(term)}\b", text_low):
            found.append(term)
    return found

_EMBED_TASK_INSTRUCTION = "Given a question, retrieve passages that answer the question"

def build_embedding_query(user_message: str) -> str:
    """qwen3-embedding es instruction-tuned: prefijar la query (no los documentos)
    con la tarea de retrieval mejora el recall ~1-5% frente a no usarlo."""
    return f"Instruct: {_EMBED_TASK_INSTRUCTION}\nQuery: {user_message.strip()}"


def build_keyword_query(user_message: str) -> str:
    text = user_message.strip()
    ids = _extract_vuln_ids(text)
    if ids:
        return ids[0]
    low = text.lower()
    terms = _find_security_terms(low)
    if terms:
        return max(terms, key=len)
    tokens = re.findall(r"[a-záéíóúñü0-9][a-záéíóúñü0-9\-]*", low)
    keywords = [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]
    return " ".join(keywords) or text


@component
class VulnIdLookup:
    """Canal determinístico de IDs: resuelve CWE-x / CVE-x por metadata.

    Un ID es una clave primaria disfrazada, no tiene sentido recuperarlo por
    similitud de coseno ni por BM25. Además el PgvectorKeywordRetriever usa
    plainto_tsquery, que AND-ea los lexemas: con dos IDs ningún documento los
    contiene a ambos y devuelve 0. Por eso este canal sale de BM25.

    Se filtra por `meta.source` y no por `cwe_id`/`cve_id` porque `cwe_id` es int
    y `cve_id` es str, mientras que `source` es str en ambos y guarda exactamente
    la forma canónica que produce _extract_vuln_ids ("CWE-79", "CVE-2021-44228").
    Las guías INCIBE tienen un nombre de archivo en `source`, así que no colisionan.

    `ids` es opcional a propósito: los scripts de eval que no lo pasan siguen
    corriendo sin tocarlos (el lookup simplemente no aporta candidatos).
    """

    def __init__(self, document_store, top_k: int = 10):
        self.document_store = document_store
        self.top_k = top_k

    @component.output_types(documents=list[Document])
    def run(self, ids: list[str] | None = None):
        ids = ids or []
        if not ids:
            return {"documents": []}
        wanted = ids[: self.top_k]
        docs = self.document_store.filter_documents(
            filters={"field": "meta.source", "operator": "in", "value": wanted}
        )
        rank = {vid: i for i, vid in enumerate(wanted)}
        docs.sort(key=lambda d: rank.get(d.meta.get("source"), 10 ** 6))
        return {"documents": docs[: self.top_k]}


# ======================================================================
# Motor RAG
# ======================================================================
def get_document_store(
    table_name: str = DB_TABLE,
    keyword_index_name: str = "ciberseguridad_keyword_index",
) -> PgvectorDocumentStore:
    """`table_name`/`keyword_index_name` son overrideables para instanciar stores
    paralelos (p. ej. un experimento de chunking en una tabla
    `ciberseguridad_docs_chunk_<estrategia>`) sin tocar la tabla de producción.
    Los defaults reproducen exactamente el store de producción. Ver
    docs/data_splitting.md y src/pipeline/eval/run_chunking_experiment.py.
    """
    return PgvectorDocumentStore(
        connection_string=Secret.from_token(DB_CONNECTION),
        embedding_dimension=EMBEDDING_DIMENSION,
        table_name=table_name,
        keyword_index_name=keyword_index_name,
    )

def build_generator(valves):
    """La generación corre siempre en el Ollama local (ver docs/arquitectura.md).

    El modelo sale de `LLM_MODEL` si está seteada; si no, del valve `llm_model`
    (que ya trae `DEFAULT_OLLAMA_LLM` como default). La variable de entorno gana
    para poder cambiar de modelo sin tocar la UI de OpenWebUI.
    """
    v = valves
    model = os.getenv("LLM_MODEL") or v.llm_model
    return OllamaGenerator(
        model=model,
        url=OLLAMA_URL,
        timeout=120,
        generation_kwargs={
            "num_predict": v.max_tokens,
            "temperature": v.temperature,
        },
    )

def build_rag_pipeline(store: PgvectorDocumentStore, valves, include_llm: bool = True) -> HaystackPipeline:
    """Arma el grafo de query (retrievers híbridos + RRF + reranker + generación).

    `include_llm=False` arma el pipeline SOLO hasta el reranker (sin prompt_builder ni
    LLM): útil para medir retrieval puro (recall@k / source_recall del experimento de
    chunking) sin cargar el generador ni pagar la latencia de generación. El runtime de
    producción usa el default (True). Ver src/pipeline/eval/run_chunking_experiment.py.
    """
    v = valves
    pipeline = HaystackPipeline()
    pipeline.add_component("text_embedder",       OllamaTextEmbedder(model=v.embedding_model, url=OLLAMA_URL))
    pipeline.add_component("embedding_retriever", PgvectorEmbeddingRetriever(document_store=store, top_k=v.retriever_top_k))
    pipeline.add_component("keyword_retriever",   PgvectorKeywordRetriever(document_store=store, top_k=v.retriever_top_k))
    pipeline.add_component("id_lookup",           VulnIdLookup(document_store=store, top_k=getattr(v, "id_lookup_top_k", 10)))
    pipeline.add_component("document_joiner",     DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=v.retriever_top_k * 2))
    pipeline.add_component("ranker",              SentenceTransformersSimilarityRanker(model=v.ranker_model, top_k=v.ranker_top_k))

    pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
    pipeline.connect("embedding_retriever",     "document_joiner")
    pipeline.connect("keyword_retriever",       "document_joiner")
    pipeline.connect("id_lookup",               "document_joiner")
    pipeline.connect("document_joiner",         "ranker.documents")

    if include_llm:
        pipeline.add_component("prompt_builder", PromptBuilder(template=PROMPT_TEMPLATE))
        pipeline.add_component("llm",            build_generator(v))
        pipeline.connect("ranker",          "prompt_builder.documents")
        pipeline.connect("prompt_builder",  "llm")

    return pipeline


class Pipeline:

    class Valves(BaseModel):
        llm_model:       str   = DEFAULT_OLLAMA_LLM
        embedding_model: str   = "qwen3-embedding:4b"
        retriever_top_k: int   = 15
        ranker_model:    str   = "BAAI/bge-reranker-v2-m3"
        ranker_top_k:    int   = 4
        id_lookup_top_k: int   = 10
        max_tokens:      int   = 512
        temperature:     float = 0.5
        # NOTA: split_length y split_overlap se movieron al script de indexación.

    def __init__(self):
        self.name   = "RAG ciberseguridad"
        self.valves = self.Valves()
        
        # Conectamos a la BD solo para recuperar datos
        self.store  = self._get_document_store()
        self.rag_pipeline = self._build_rag_pipeline()
        
        logger.info("Pipeline RAG inicializado. Listo para responder.")

    async def on_startup(self):
        # Ya no indexamos. Solo verificamos cuántos documentos existen.
        try:
            total = len(self.store.filter_documents())
            logger.info(f"Conexión a store verificada. Documentos en BD: {total}")
        except Exception as e:
            logger.error(f"Error conectando a la BD en el arranque: {e}")

    async def on_valves_updated(self):
        logger.info("Valves actualizados — reconstruyendo pipeline...")
        self.rag_pipeline = self._build_rag_pipeline()

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:
        
        logger.info(f"Ejecutando RAG para: {user_message}")

        keyword_query = build_keyword_query(user_message)
        logger.info(f"[KEYWORD QUERY] -> {keyword_query!r}")

        vuln_ids, ids_origin = resolve_vuln_ids(user_message, messages)
        logger.info(f"[VULN IDS] -> {vuln_ids} (origen={ids_origin})")

        result = self.rag_pipeline.run(
            {
                "text_embedder":     {"text": build_embedding_query(user_message)},
                "keyword_retriever": {"query": keyword_query},
                "id_lookup":         {"ids": vuln_ids},
                "ranker":            {"query": user_message},
                "prompt_builder":    {"question": user_message},
            },
            include_outputs_from={
                "embedding_retriever",
                "keyword_retriever",
                "id_lookup",
                "document_joiner",
                "ranker",
                "prompt_builder",
            },
        )

        self._log_retrieved_docs(result)
        self._log_token_usage(result)

        return result["llm"]["replies"][0]

    def _log_token_usage(self, result: dict) -> None:
        sep = "-" * 60
        prompt = result.get("prompt_builder", {}).get("prompt") or ""
        n_docs = len(result.get("ranker", {}).get("documents", []))
        meta_list = result.get("llm", {}).get("meta") or []
        meta = meta_list[0] if meta_list else {}

        prompt_tokens = completion_tokens = total_tokens = None
        usage = meta.get("usage")
        if isinstance(usage, dict):
            prompt_tokens     = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens      = usage.get("total_tokens")
        else:
            prompt_tokens     = meta.get("prompt_eval_count")
            completion_tokens = meta.get("eval_count")

        if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

        logger.info(f"{sep}")
        logger.info(f"[TOKENS] docs_al_prompt={n_docs} | prompt_chars={len(prompt)} | prompt_words={len(prompt.split())}")
        if prompt_tokens is None and completion_tokens is None:
            model = meta.get("model", "desconocido")
            logger.info(f"[TOKENS] uso no reportado por el generador (model={model}).")
        else:
            logger.info(f"[TOKENS] prompt={prompt_tokens} | completion={completion_tokens} | total={total_tokens} | model={meta.get('model', 'desconocido')}")
        logger.info(f"{sep}")

    def _log_retrieved_docs(self, result: dict) -> None:
        sep = "-" * 60

        emb_docs = result.get("embedding_retriever", {}).get("documents", [])
        logger.info(f"{sep}")
        logger.info(f"[EMBEDDING RETRIEVER] {len(emb_docs)} documentos recuperados:")
        for i, doc in enumerate(emb_docs):
            score   = doc.score if doc.score is not None else "N/A"
            source  = doc.meta.get("source") or doc.meta.get("file_path", "desconocido")
            snippet = (doc.content or "")[:200].replace("\n", " ")
            logger.info(f"  [{i+1}] score={score} | fuente={source}\n        snippet: {snippet}...")

        kw_docs = result.get("keyword_retriever", {}).get("documents", [])
        logger.info(f"{sep}")
        logger.info(f"[KEYWORD RETRIEVER] {len(kw_docs)} documentos recuperados:")
        for i, doc in enumerate(kw_docs):
            score   = doc.score if doc.score is not None else "N/A"
            source  = doc.meta.get("source") or doc.meta.get("file_path", "desconocido")
            snippet = (doc.content or "")[:200].replace("\n", " ")
            logger.info(f"  [{i+1}] score={score} | fuente={source}\n        snippet: {snippet}...")

        lookup_docs = result.get("id_lookup", {}).get("documents", [])
        logger.info(f"{sep}")
        logger.info(f"[ID LOOKUP] {len(lookup_docs)} documentos por metadata:")
        for i, doc in enumerate(lookup_docs):
            logger.info(f"  [{i+1}] source={doc.meta.get('source')}")

        joined_docs = result.get("document_joiner", {}).get("documents", [])
        logger.info(f"{sep}")
        logger.info(f"[DOCUMENT JOINER] {len(joined_docs)} candidatos fusionados (RRF):")
        for i, doc in enumerate(joined_docs):
            source  = doc.meta.get("source") or doc.meta.get("file_path", "desconocido")
            snippet = (doc.content or "")[:300].replace("\n", " ")
            logger.info(f"  [{i+1}] fuente={source}\n        contenido: {snippet}...")

        ranked_docs = result.get("ranker", {}).get("documents", [])
        logger.info(f"{sep}")
        logger.info(f"[RANKER] {len(ranked_docs)} documentos re-rankeados enviados al prompt:")
        for i, doc in enumerate(ranked_docs):
            score   = f"{doc.score:.4f}" if doc.score is not None else "N/A"
            source  = doc.meta.get("source") or doc.meta.get("file_path", "desconocido")
            snippet = (doc.content or "")[:300].replace("\n", " ")
            logger.info(f"  [{i+1}] score={score} | fuente={source}\n        contenido: {snippet}...")
        logger.info(f"{sep}")

    def _get_document_store(self) -> PgvectorDocumentStore:
        return get_document_store()

    def _build_rag_pipeline(self) -> HaystackPipeline:
        return build_rag_pipeline(self.store, self.valves)