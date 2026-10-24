"""
title: Rag de Ciberseguridad (Inferencia)
author: Valentino, Miguel y Luca
version: 2.0 (Arquitectura Desacoplada)
requirements: haystack-ai, pgvector-haystack, ollama-haystack, sentence-transformers
"""

from dataclasses import dataclass
from typing import List, Optional, Union, Generator, Iterator
from queue import Queue
from pydantic import BaseModel, computed_field
import logging
import os
import re
import sys
import threading
import time
import uuid

# El servidor carga cada pipeline con `spec_from_file_location`, que no agrega su
# directorio a `sys.path`: sin esto `import chat` falla y el archivo termina en
# `failed/`. Mismo arreglo que en pipeline_dependencias.py.
_PIPELINES_DIR = os.path.dirname(os.path.abspath(__file__))
if _PIPELINES_DIR not in sys.path:
    sys.path.insert(0, _PIPELINES_DIR)

from chat import extract_findings_context, findings_vuln_ids, format_findings, select_history
from common.log import get_logger
from common.llm import DEFAULT_OLLAMA_LLM, effective_num_ctx, llm_model
from common.store import OLLAMA_URL, get_document_store
from common.text import finding_identifier, message_text

logger = get_logger("HaystackRAG_Query", file="/app/pipelines/logCiberseguridad.txt", propagate=False)

from haystack import Pipeline as HaystackPipeline, Document, component
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

_WARM_UP_LOCK = threading.Lock()

# Componentes intermedios cuya salida se pide a Haystack para loguearla.
_LOGGED_OUTPUTS = {
    "embedding_retriever",
    "keyword_retriever",
    "id_lookup",
    "document_joiner",
    "ranker",
    "prompt_builder",
}

# Qué documentos se loguean de cada componente: (componente, encabezado, con score, ancho).
_LOGGED_DOCS = (
    ("embedding_retriever", "[EMBEDDING RETRIEVER] {n} documentos recuperados:", True, 200),
    ("keyword_retriever", "[KEYWORD RETRIEVER] {n} documentos recuperados:", True, 200),
    ("id_lookup", "[ID LOOKUP] {n} documentos por metadata:", False, 200),
    ("document_joiner", "[DOCUMENT JOINER] {n} candidatos fusionados (RRF):", False, 300),
    ("ranker", "[RANKER] {n} documentos re-rankeados enviados al prompt:", True, 300),
)

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
- The conversation so far is only there to understand follow-ups ("it", "this vulnerability").
  Facts must still come from the context or the vulnerabilities under discussion.

{% if findings %}
Vulnerabilities under discussion (attached by the user from their dependency scan; they are the topic of the conversation):
{{ findings }}
{% endif %}

Context:
{% for document in documents %}
- {% if document.meta.cve_id %}[{{ document.meta.cve_id }}]{% if document.meta.severity %} severity={{ document.meta.severity }}{% endif %}{% if document.meta.cvss_v3_score %} CVSS={{ document.meta.cvss_v3_score }}{% endif %}{% if document.meta.cwe_ids %} (related: {{ document.meta.cwe_ids | join(", ") }}){% endif %}{% if document.meta.products %} affects: {{ document.meta.products | join(", ") }}{% endif %}
  {% endif %}{{ document.content }}
{% endfor %}

{% if history %}
Conversation so far:
{% for turn in history %}{{ "User" if turn.role == "user" else "Assistant" }}: {{ turn.content }}
{% endfor %}
{% endif %}
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
    "prevenir", "acerca", "información", "informacion", "the", "an", "of", "to", "in", 
    "on", "for", "and", "or", "is", "are", "be", "what", "which", "how", "when", "where", 
    "why", "can", "could", "does", "do", "about", "tell", "give", "explain", "types", "exist",
    "vulnerabilidad", "vulnerabilidades", "vulnerability", "descripcion", "descripción",
    "description", "cwe", "cve", "weakness", "debilidad",
}

def extract_vuln_ids(text: str) -> list[str]:
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


_HISTORY_TURNS = 4   # ~2 intercambios hacia atrás
_MAX_VULN_IDS  = 6


def resolve_vuln_ids(
    user_message: str,
    messages: list[dict] | None = None,
) -> tuple[list[str], str]:
    """IDs del turno actual; si no hay ninguno, los hereda de los últimos turnos.

    El turno actual manda: sólo se mira hacia atrás cuando no trae ningún ID, así
    un cambio explícito de tema nunca queda contaminado. Dentro del historial se
    acumulan del más reciente al más viejo, y ese orden ES el peso: VulnIdLookup
    lo respeta y el RRF le da 1/(k+rango) a cada documento.

    Devuelve (ids, origen); `origen` se loguea para poder auditar el arrastre.
    """
    current = extract_vuln_ids(user_message)
    if current:
        return current[:_MAX_VULN_IDS], "turno"
    if not messages:
        return [], "ninguno"

    inherited: list[str] = []
    for msg in reversed(messages[-_HISTORY_TURNS:]):
        text = message_text(msg)
        if not text or text.strip() == user_message.strip():
            continue  # messages trae el turno actual al final
        for vid in extract_vuln_ids(text):
            if vid not in inherited:
                inherited.append(vid)
    if not inherited:
        return [], "ninguno"
    return inherited[:_MAX_VULN_IDS], "historial"

class _RequestLog(logging.LoggerAdapter):
    """Prefija cada línea con el id de la corrida (`[rid]`)."""

    def process(self, msg, kwargs):
        return f"[{self.extra['rid']}] {msg}", kwargs


def _one_line(text: str) -> str:
    return (text or "").strip().replace("\n", " ⏎ ")


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


def build_ranker_query(user_message: str, vuln_ids: list[str]) -> str:
    """Query contra la que puntúa el cross-encoder.

    Si los IDs se heredaron del historial ("dame un ejemplo de esta vulnerabilidad"),
    la pregunta no los nombra y el ranker deja los docs de VulnIdLookup por debajo
    de CVEs al azar. Se agregan los IDs que el texto no menciona.
    """
    mentioned = set(extract_vuln_ids(user_message))
    missing = [vid for vid in vuln_ids if vid not in mentioned]
    if not missing:
        return user_message
    return f"{user_message.strip()} ({', '.join(missing)})"


def build_keyword_query(user_message: str) -> str:
    text = user_message.strip()
    ids = extract_vuln_ids(text)
    if ids:
        return ids[0]
    low = text.lower()
    terms = _find_security_terms(low)
    if terms:
        return max(terms, key=len)
    tokens = re.findall(r"[a-záéíóúñü0-9][a-záéíóúñü0-9\-]*", low)
    keywords = [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]
    return " ".join(keywords) or text


@dataclass(frozen=True)
class Query:
    """Lo que el pipeline saca de una pregunta. Lo usan `pipe()` y el eval, así los dos
    corren exactamente el mismo grafo con los mismos inputs."""

    question: str
    keyword_query: str
    vuln_ids: list[str]
    ids_origin: str
    ranker_query: str
    findings: list[dict]
    history: list[dict]

    def inputs(self, include_llm: bool = True) -> dict:
        """Los inputs de `build_rag_pipeline`; sin LLM no hay `prompt_builder`."""
        inputs = {
            "text_embedder":     {"text": build_embedding_query(self.question)},
            "keyword_retriever": {"query": self.keyword_query},
            "id_lookup":         {"ids": self.vuln_ids},
            "ranker":            {"query": self.ranker_query},
        }
        if include_llm:
            inputs["prompt_builder"] = {
                "question": self.question,
                "findings": format_findings(self.findings),
                "history":  self.history,
            }
        return inputs


def build_query(
    user_message: str,
    messages: list[dict] | None = None,
    history_max_messages: int = 0,
    history_max_chars: int = 0,
) -> Query:
    """Una pregunta del chat (con su conversación y hallazgos adjuntos) o del eval (sola)."""
    findings = extract_findings_context(messages)
    vuln_ids, ids_origin = resolve_vuln_ids(user_message, messages)
    # Los IDs de los hallazgos van DETRÁS: si la pregunta nombra otro CVE, ese manda.
    for vid in findings_vuln_ids(findings):
        if vid not in vuln_ids:
            vuln_ids.append(vid)
    return Query(
        question=user_message,
        keyword_query=build_keyword_query(user_message),
        vuln_ids=vuln_ids,
        ids_origin=ids_origin,
        ranker_query=build_ranker_query(user_message, vuln_ids),
        findings=findings,
        history=select_history(
            messages, user_message, max_messages=history_max_messages, max_chars=history_max_chars,
        ),
    )


@component
class VulnIdLookup:
    """Canal determinístico de IDs: resuelve CWE-x / CVE-x por metadata.

    Un ID es una clave primaria disfrazada, no tiene sentido recuperarlo por
    similitud de coseno ni por BM25. Además el PgvectorKeywordRetriever usa
    plainto_tsquery, que AND-ea los lexemas: con dos IDs ningún documento los
    contiene a ambos y devuelve 0. Por eso este canal sale de BM25.

    Se filtra por `meta.source` y no por `cwe_id`/`cve_id` porque `cwe_id` es int
    y `cve_id` es str, mientras que `source` es str en ambos y guarda exactamente
    la forma canónica que produce extract_vuln_ids ("CWE-79", "CVE-2021-44228").
    Las guías INCIBE tienen un nombre de archivo en `source`, así que no colisionan.
    """

    def __init__(self, document_store, top_k: int = 10):
        self.document_store = document_store
        self.top_k = top_k

    @component.output_types(documents=list[Document])
    def run(self, ids: list[str]):
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
def build_generator(valves):
    """La generación corre siempre en el Ollama local (ver docs/arquitectura.md).

    El modelo sale de `LLM_MODEL` si está seteada; si no, del valve `llm_model`
    (que ya trae `DEFAULT_OLLAMA_LLM` como default). La ventana, igual: `LLM_NUM_CTX`
    le gana al valve `num_ctx`, y 0 toma la del Modelfile. Las variables de entorno
    ganan para poder cambiar de modelo sin tocar los valves.
    """
    v = valves
    model = llm_model(v)
    generation_kwargs = {
        "num_predict": v.max_tokens,
        "temperature": v.temperature,
    }
    # Sin num_ctx, Ollama usa su default y, si el prompt no entra, recorta el PRINCIPIO
    # sin avisar: justo las reglas. Por eso se manda explícito aunque salga del Modelfile.
    num_ctx = effective_num_ctx(v, OLLAMA_URL)
    if num_ctx:
        generation_kwargs["num_ctx"] = num_ctx
    return OllamaGenerator(
        model=model,
        url=OLLAMA_URL,
        timeout=120,
        generation_kwargs=generation_kwargs,
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
    pipeline.add_component("id_lookup",           VulnIdLookup(document_store=store, top_k=v.id_lookup_top_k))
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
        # Mismo override que run_indexing.py: query e indexación DEBEN usar el mismo
        # modelo de embeddings o los vectores no matchean.
        embedding_model: str   = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:4b")
        retriever_top_k: int   = 15
        ranker_model:    str   = "BAAI/bge-reranker-v2-m3"
        ranker_top_k:    int   = 4
        id_lookup_top_k: int   = 10
        max_tokens:      int   = 512
        temperature:     float = 0.5
        num_ctx:         int   = 8192  # ventana de Ollama: prompt + respuesta
        # Historial del chat que entra al prompt. El hallazgo de la extensión (mensaje
        # system) va aparte y entra siempre. 0 desactiva el historial.
        history_max_messages: int = 6
        history_max_chars:    int = 600
        # NOTA: split_length y split_overlap se movieron al script de indexación.

        @computed_field
        @property
        def effective_num_ctx(self) -> Optional[int]:
            """La ventana con la que corre el generador (`LLM_NUM_CTX` le gana a `num_ctx`).

            De sólo lectura: sale en `GET /valves` para el medidor de contexto de la
            extensión, pero no en el schema de edición, y el servidor la ignora al
            restaurar valves.json.
            """
            return effective_num_ctx(self, OLLAMA_URL)

    def __init__(self):
        self.name   = "RAG ciberseguridad"
        self.valves = self.Valves()
        
        # Conectamos a la BD solo para recuperar datos
        self.store  = get_document_store()
        self.rag_pipeline = self._build_rag_pipeline()
        
        logger.info("Pipeline RAG inicializado. Listo para responder.")

    async def on_startup(self):
        # Ya no indexamos. Solo verificamos cuántos documentos existen.
        try:
            total = self.store.count_documents()
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
        
        # Puede haber corridas concurrentes (varios chats o clientes a la vez):
        # sin un id por request sus líneas quedan intercaladas y no se pueden separar.
        log = _RequestLog(logger, {"rid": uuid.uuid4().hex[:8]})
        t0 = time.perf_counter()
        stream = bool(body.get("stream"))

        log.info(f"[INICIO] pregunta={_one_line(user_message)}")

        query = build_query(
            user_message,
            messages,
            history_max_messages=self.valves.history_max_messages,
            history_max_chars=self.valves.history_max_chars,
        )
        log.info(f"[KEYWORD QUERY] -> {query.keyword_query!r}")
        for finding in query.findings:
            log.info(f"[HALLAZGO] {finding_identifier(finding)} | paquete={finding.get('package')} {finding.get('installed_version')}")
        log.info(f"[HISTORIAL] {len(query.history)} mensajes al prompt")
        log.info(f"[VULN IDS] -> {query.vuln_ids} (origen={query.ids_origin}{f', +{len(query.findings)} hallazgo(s)' if query.findings else ''})")
        log.info(f"[RANKER QUERY] -> {_one_line(query.ranker_query)!r}")

        if stream:
            return self._stream(query.inputs(), log, t0)
        return self._run(query.inputs(), log, t0)

    def _run(
        self,
        inputs: dict,
        log: logging.LoggerAdapter,
        t0: float,
        streaming_callback=None,
    ) -> str:
        """Corre el pipeline de Haystack, loguea lo recuperado y devuelve la respuesta."""
        if streaming_callback is not None:
            inputs = {**inputs, "llm": {"streaming_callback": streaming_callback}}
        try:
            result = self.rag_pipeline.run(inputs, include_outputs_from=_LOGGED_OUTPUTS)
            answer = result["llm"]["replies"][0]
            self._log_retrieved_docs(log, result)
            self._log_token_usage(log, result)
            log.info(f"[RESPUESTA] {_one_line(answer)}")
            log.info(f"[FIN] {time.perf_counter() - t0:.1f}s")
        except Exception:
            log.exception(f"[ERROR] falló tras {time.perf_counter() - t0:.1f}s")
            raise
        return answer

    def _stream(
        self,
        inputs: dict,
        log: logging.LoggerAdapter,
        t0: float,
    ) -> Iterator[str]:
        """Ejecuta Haystack en segundo plano y entrega los tokens al servidor HTTP."""
        events: Queue[object] = Queue()
        finished = object()

        def on_chunk(chunk) -> None:
            content = getattr(chunk, "content", "")
            if content:
                events.put(content)

        def run() -> None:
            try:
                self._run(inputs, log, t0, streaming_callback=on_chunk)
            except Exception as error:  # _run ya lo logueó
                events.put(error)
            finally:
                events.put(finished)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        while True:
            event = events.get()
            if event is finished:
                break
            if isinstance(event, BaseException):
                raise event
            yield event

    def _log_token_usage(self, log: logging.LoggerAdapter, result: dict) -> None:
        prompt = result.get("prompt_builder", {}).get("prompt") or ""
        n_docs = len(result.get("ranker", {}).get("documents", []))
        meta_list = result.get("llm", {}).get("meta") or []
        meta = meta_list[0] if meta_list else {}
        # OllamaGenerator ya convierte los contadores de Ollama al formato `usage` de OpenAI.
        usage = meta.get("usage") or {}
        model = meta.get("model", "desconocido")

        log.info(f"[TOKENS] docs_al_prompt={n_docs} | prompt_chars={len(prompt)} | prompt_words={len(prompt.split())}")
        if not usage:
            log.info(f"[TOKENS] uso no reportado por el generador (model={model}).")
        else:
            log.info(f"[TOKENS] prompt={usage.get('prompt_tokens')} | completion={usage.get('completion_tokens')} | total={usage.get('total_tokens')} | model={model}")

    def _log_retrieved_docs(self, log: logging.LoggerAdapter, result: dict) -> None:
        # Una línea por documento: con saltos de línea las continuaciones quedan
        # sin request id y no se pueden filtrar con grep.
        for component, header, with_score, width in _LOGGED_DOCS:
            docs = result.get(component, {}).get("documents", [])
            log.info(header.format(n=len(docs)))
            for i, doc in enumerate(docs):
                source = doc.meta.get("source") or doc.meta.get("file_path", "desconocido")
                snippet = " ".join((doc.content or "")[:width].split())
                score = ""
                if with_score:
                    score = f"score={doc.score:.4f} | " if doc.score is not None else "score=N/A | "
                log.info(f"  [{i+1}] {score}fuente={source} | {snippet}...")

    def _build_rag_pipeline(self) -> HaystackPipeline:
        pipeline = build_rag_pipeline(self.store, self.valves)
        # Warm-up eager y bajo lock: si se deja al primer run(), dos requests
        # concurrentes cargarían dos CrossEncoder a la vez en threads distintos, y
        # eso rompe con "Cannot copy out of meta tensor".
        with _WARM_UP_LOCK:
            pipeline.warm_up()
        return pipeline