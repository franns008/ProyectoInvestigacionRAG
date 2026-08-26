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

from haystack import Pipeline as HaystackPipeline
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
from haystack.components.generators.openai import OpenAIGenerator
from haystack_integrations.components.generators.ollama import OllamaGenerator

# --- Configuración fija de infraestructura ---
DB_CONNECTION       = "postgresql://avdbuser:avdbpass@vdb:5432/pgvdb"
OLLAMA_URL          = "http://ollama:11434"
GROQ_BASE_URL       = "https://api.groq.com/openai/v1"
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
    ids: list[str] = []
    for m in _CWE_ID_RE.finditer(text):
        ids.append(f"CWE-{int(m.group(1))}")
    for m in _CVE_ID_RE.finditer(text):
        ids.append(f"CVE-{m.group(1)}-{m.group(2)}")
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out

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

def _llm_provider() -> str:
    return os.getenv("LLM_PROVIDER", "groq").strip().lower()

def build_generator(valves):
    v = valves
    provider = _llm_provider()
    if provider == "ollama":
        model = os.getenv("LLM_MODEL") or DEFAULT_OLLAMA_LLM
        return OllamaGenerator(
            model=model,
            url=OLLAMA_URL,
            timeout=120,
            generation_kwargs={
                "num_predict": v.max_tokens,
                "temperature": v.temperature,
            },
        )
    model = os.getenv("LLM_MODEL") or v.llm_model
    return OpenAIGenerator(
        api_key=Secret.from_env_var("GROQ_API_KEY"),
        api_base_url=GROQ_BASE_URL,
        model=model,
        max_retries=5,
        timeout=60.0,
        generation_kwargs={
            "max_tokens":  v.max_tokens,
            "temperature": v.temperature,
        },
    )

def build_rag_pipeline(store: PgvectorDocumentStore, valves, include_llm: bool = True) -> HaystackPipeline:
    """Arma el grafo de query (retrievers híbridos + RRF + reranker + generación).

    `include_llm=False` arma el pipeline SOLO hasta el reranker (sin prompt_builder ni
    LLM): útil para medir retrieval puro (recall@k / source_recall del experimento de
    chunking) sin exigir GROQ_API_KEY ni pagar la latencia de generación. El runtime de
    producción usa el default (True). Ver src/pipeline/eval/run_chunking_experiment.py.
    """
    v = valves
    pipeline = HaystackPipeline()
    pipeline.add_component("text_embedder",       OllamaTextEmbedder(model=v.embedding_model, url=OLLAMA_URL))
    pipeline.add_component("embedding_retriever", PgvectorEmbeddingRetriever(document_store=store, top_k=v.retriever_top_k))
    pipeline.add_component("keyword_retriever",   PgvectorKeywordRetriever(document_store=store, top_k=v.retriever_top_k))
    pipeline.add_component("document_joiner",     DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=v.retriever_top_k * 2))
    pipeline.add_component("ranker",              SentenceTransformersSimilarityRanker(model=v.ranker_model, top_k=v.ranker_top_k))

    pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
    pipeline.connect("embedding_retriever",     "document_joiner")
    pipeline.connect("keyword_retriever",       "document_joiner")
    pipeline.connect("document_joiner",         "ranker.documents")

    if include_llm:
        pipeline.add_component("prompt_builder", PromptBuilder(template=PROMPT_TEMPLATE))
        pipeline.add_component("llm",            build_generator(v))
        pipeline.connect("ranker",          "prompt_builder.documents")
        pipeline.connect("prompt_builder",  "llm")

    return pipeline


class Pipeline:

    class Valves(BaseModel):
        llm_model:       str   = "meta-llama/llama-4-scout-17b-16e-instruct"
        embedding_model: str   = "qwen3-embedding:4b"
        retriever_top_k: int   = 15
        ranker_model:    str   = "BAAI/bge-reranker-v2-m3"
        ranker_top_k:    int   = 4
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

        result = self.rag_pipeline.run(
            {
                "text_embedder":     {"text": build_embedding_query(user_message)},
                "keyword_retriever": {"query": keyword_query},
                "ranker":            {"query": user_message},
                "prompt_builder":    {"question": user_message},
            },
            include_outputs_from={
                "embedding_retriever",
                "keyword_retriever",
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