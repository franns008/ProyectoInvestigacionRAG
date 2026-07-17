"""
title: Rag de Ciberseguridad
author: Valentino, Miguel y Luca
version: 1.4
requirements: haystack-ai, pgvector-haystack, ollama-haystack, sentence-transformers, pypdf, python-docx, markdown-it-py, mdit-plain
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
from pathlib import Path
from pydantic import BaseModel
import hashlib
import json
import logging
import os
import re

# Configurar el logger específico para nuestra app
logger = logging.getLogger("HaystackRAG")
logger.setLevel(logging.INFO)
logger.propagate = False  # Evita que el root logger intercepte este logger

# Configurar el archivo de salida solo para este logger
if not logger.handlers:
    file_handler = logging.FileHandler('/app/pipelines/logCiberseguridad.txt')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

from haystack import Pipeline as HaystackPipeline, Document
from haystack.utils import Secret
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.retrievers.pgvector import (
    PgvectorEmbeddingRetriever,
    PgvectorKeywordRetriever,
)
from haystack.components.builders import PromptBuilder
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.converters import (
    DOCXToDocument, MarkdownToDocument, TextFileToDocument
)
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder, OllamaDocumentEmbedder
from haystack.components.generators.openai import OpenAIGenerator
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.document_stores.types import DuplicatePolicy

# --- Configuración fija de infraestructura ---
DB_CONNECTION       = "postgresql://avdbuser:avdbpass@vdb:5432/pgvdb"
OLLAMA_URL          = "http://ollama:11434"
GROQ_BASE_URL       = "https://api.groq.com/openai/v1"
DB_TABLE            = "ciberseguridad_docs"
EMBEDDING_DIMENSION = 1024
INPUT_DIR           = Path("/app/pipelines/rawdata")
CONVERTED_DIR       = INPUT_DIR / "_converted_md"   # caché de PDFs ya convertidos

# --- Proveedor del LLM de generación (ver docs/modos_llm.md) ---
# El repo soporta dos modos de generación, elegidos con la env var LLM_PROVIDER:
#   - "groq"   (default): generación por API (Groq, endpoint OpenAI-compatible).
#                         Corre en cualquier máquina (CPU); requiere GROQ_API_KEY.
#   - "ollama"          : generación local en Ollama (orientado a GPU Nvidia).
#                         No requiere GROQ_API_KEY. El modelo se elige con LLM_MODEL.
# Los EMBEDDINGS siempre son locales en Ollama (bge-m3) en ambos modos.
DEFAULT_OLLAMA_LLM  = "qwen2.5:3b-instruct"   # ~2 GB VRAM; override con LLM_MODEL

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
- If the context does not contain the answer, say you don't know. Do not make one up.
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
# Converters de fuentes de ciberseguridad → haystack.Document
#
# Ambos producen documentos ATÓMICOS (un CWE / un CVE = un Document, con id
# determinístico). NO se chunkean con el DocumentSplitter y se deduplican por
# id, a diferencia de PDFs/DOCX/MD/TXT. Ver docs/indexing_design_capa1.md.
# ======================================================================

class XMLCWEConverter:
    """XML de MITRE (esquema CWE, namespace cwe-7) → un Document por Weakness."""

    def run(self, sources: list[Path]):
        import xml.etree.ElementTree as ET
        from haystack import Document

        documents = []
        skipped = 0
        ns = {'cwe': 'http://cwe.mitre.org/cwe-7'}

        for file_path in sources:
            try:
                tree = ET.parse(file_path)
            except ET.ParseError as e:
                logger.error(f"[XMLCWE] No se pudo parsear {file_path.name}: {e}")
                continue

            root = tree.getroot()

            for weakness in root.findall('.//cwe:Weakness', ns):
                cwe_id = weakness.get('ID')
                name = weakness.get('Name')

                # Si falta el ID, no podemos generar un doc.id único ni confiable — se descarta
                if not cwe_id:
                    skipped += 1
                    continue

                desc_elem = weakness.find('cwe:Description', ns)
                description = self._extract_text(desc_elem)

                # Fallback: si Description viene vacía, probamos con Extended_Description
                if not description:
                    ext_elem = weakness.find('cwe:Extended_Description', ns)
                    description = self._extract_text(ext_elem)

                name = (name or "").strip()
                description = (description or "").strip()

                content = f"Vulnerabilidad CWE-{cwe_id}: {name}\nDescripción: {description}".strip()

                # Filtro de calidad: descarta chunks sin sustancia real
                # (sin nombre Y sin descripción, o contenido alfanumérico insuficiente)
                alnum_chars = sum(c.isalnum() for c in content)
                if alnum_chars < 20:
                    logger.warning(
                        f"[XMLCWE] Descartado CWE-{cwe_id} por contenido insuficiente "
                        f"(alnum_chars={alnum_chars}): '{content[:80]}'"
                    )
                    skipped += 1
                    continue

                doc = Document(
                    id=f"cwe-{cwe_id}",
                    content=content,
                    meta={
                        "source_type": "cwe_weakness",
                        "source": f"CWE-{cwe_id}",
                        "cwe_id": int(cwe_id) if cwe_id.isdigit() else cwe_id,
                        "name": name or "Sin nombre",
                        "source_file": str(file_path),
                    }
                )
                documents.append(doc)

        logger.info(
            f"[XMLCWE] {len(documents)} documentos CWE generados, "
            f"{skipped} descartados por datos insuficientes."
        )
        return {"documents": documents}

    @staticmethod
    def _extract_text(elem) -> str:
        """
        Extrae todo el texto de un elemento XML, incluyendo el de sub-tags anidados
        (ej. <xhtml:p>, <xhtml:div> dentro de <Description>). A diferencia de
        elem.text (que solo trae el texto directo del nodo), esto recorre todo
        el árbol y concatena cada fragmento de texto encontrado.
        """
        if elem is None:
            return ""
        parts = [t.strip() for t in elem.itertext() if t and t.strip()]
        return " ".join(parts)


# --- NVD CVE (JSON) → Document ---------------------------------------------
# Portado de src/ingestion/index_nvd.py (rama ciber-fetching-turco): SOLO la
# lógica de conversión. El embedding/escritura a pgvector lo hace el pipeline.
CWE_RE = re.compile(r"^CWE-\d+$")


def _english_description(cve: dict) -> str:
    """Descripción en inglés (lo único que se embebe). Fallback: la primera que haya."""
    descriptions = cve.get("descriptions", [])
    for d in descriptions:
        if d.get("lang") == "en":
            return d.get("value", "")
    return descriptions[0].get("value", "") if descriptions else ""


def _cvss(metrics: dict) -> dict:
    """Extrae score/vector/severity. Prioriza v3.1 > v3.0 para 'v3', guarda v2 aparte.

    Cada métrica en el JSON de NVD es una LISTA; tomamos el primer elemento
    (habitualmente el 'Primary' de nvd@nist.gov).
    """
    out = {
        "cvss_v3_score": None, "cvss_v3_vector": None,
        "cvss_v2_score": None, "cvss_v2_vector": None,
        "severity": None,
    }
    for key in ("cvssMetricV31", "cvssMetricV30"):
        entries = metrics.get(key)
        if entries:
            data = entries[0].get("cvssData", {})
            out["cvss_v3_score"] = data.get("baseScore")
            out["cvss_v3_vector"] = data.get("vectorString")
            out["severity"] = data.get("baseSeverity") or entries[0].get("baseSeverity")
            break
    entries_v2 = metrics.get("cvssMetricV2")
    if entries_v2:
        data = entries_v2[0].get("cvssData", {})
        out["cvss_v2_score"] = data.get("baseScore")
        out["cvss_v2_vector"] = data.get("vectorString")
        if out["severity"] is None:
            out["severity"] = entries_v2[0].get("baseSeverity")
    return out


def _vendors_products(cve: dict) -> tuple[list[str], list[str]]:
    """Vendors y productos únicos, parseados de los CPE de configurations.

    CPE 2.3 = cpe:2.3:<part>:<vendor>:<product>:<version>:...  (índices 3 y 4).
    """
    vendors, products = set(), set()
    for config in cve.get("configurations", []):
        for node in config.get("nodes", []):
            for match in node.get("cpeMatch", []):
                parts = match.get("criteria", "").split(":")
                if len(parts) > 4:
                    vendor, product = parts[3], parts[4]
                    if vendor and vendor != "*":
                        vendors.add(vendor)
                    if product and product != "*":
                        products.add(product)
    return sorted(vendors), sorted(products)


def _cwe_ids(cve: dict) -> list[str]:
    ids = set()
    for weakness in cve.get("weaknesses", []):
        for desc in weakness.get("description", []):
            value = desc.get("value", "")
            if CWE_RE.match(value):
                ids.add(value)
    return sorted(ids)


def cve_to_document(cve: dict):
    """Convierte un objeto 'cve' crudo de NVD en un Document de Haystack.

    Devuelve None si el CVE no tiene descripción utilizable (nada que embeber).
    """
    from haystack import Document

    cve_id = cve.get("id")
    description = _english_description(cve)
    if not cve_id or not description.strip():
        return None

    vendors, products = _vendors_products(cve)
    meta = {
        "source_type": "nvd_cve",
        "source": cve_id,
        "cve_id": cve_id,
        "published_date": cve.get("published"),
        "last_modified": cve.get("lastModified"),
        "vendors": vendors,
        "products": products,
        "cwe_ids": _cwe_ids(cve),
        "references": [r.get("url") for r in cve.get("references", []) if r.get("url")],
        **_cvss(cve.get("metrics", {})),
    }
    # id determinístico: reindexar el mismo CVE actualiza la fila, no la duplica.
    doc_id = hashlib.sha256(cve_id.encode()).hexdigest()
    return Document(id=doc_id, content=description, meta=meta)


class NVDJsonConverter:
    """Páginas JSON crudas de la NVD (cves_page_*.json de fetch_nvd.py) → un
    Document atómico por CVE. Ante un CVE repetido entre páginas, gana el de
    lastModified más reciente."""

    def run(self, sources: list[Path]):
        by_id: dict[str, dict] = {}
        for path in sources:
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.error(f"[NVD] No se pudo leer {Path(path).name}: {e}")
                continue
            for entry in data.get("vulnerabilities", []):
                cve = entry.get("cve", {})
                cve_id = cve.get("id")
                if not cve_id:
                    continue
                prev = by_id.get(cve_id)
                if prev is None or cve.get("lastModified", "") >= prev.get("lastModified", ""):
                    by_id[cve_id] = cve

        documents = []
        skipped = 0
        for cve in by_id.values():
            doc = cve_to_document(cve)
            if doc is None:
                skipped += 1
                continue
            documents.append(doc)

        logger.info(
            f"[NVD] {len(documents)} documentos CVE generados, "
            f"{skipped} descartados por falta de descripción."
        )
        return {"documents": documents}


# ======================================================================
# Detector estático de keywords para el retriever full-text
#
# El PgvectorKeywordRetriever ANDea todos los términos de la query, así que
# pasarle la pregunta cruda en lenguaje natural da 0 hits: ninguna frase en
# español matchea un doc CWE en inglés (ver docs/optimizacion_keyword_retrieval.md).
# Este preprocesador reduce la query a señal de alta precisión, priorizando:
#   1) IDs de vulnerabilidad (CWE-/CVE-): exactos y ultra-distintivos.
#   2) Términos de seguridad conocidos (XSS, SQL injection, ...).
#   3) Fallback: tokens de contenido sin stopwords ni ruido de dominio.
# ======================================================================

# IDs de vulnerabilidad, tolerante a variantes: "CWE-89", "cwe 89", "cwe89".
_CWE_ID_RE = re.compile(r"cwe[\s\-_]?(\d{1,5})", re.IGNORECASE)
_CVE_ID_RE = re.compile(r"cve[\s\-_]?(\d{4})[\s\-_]?(\d{4,7})", re.IGNORECASE)

# Términos de seguridad de alta señal (bilingüe). Se permiten multi-palabra: el
# full-text ANDea sus tokens, que co-ocurren en el doc objetivo. Ampliable.
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

# Stopwords ES/EN + ruido de dominio (palabras presentes en casi todos los docs,
# que no discriminan y sólo rompen el AND del full-text).
_STOPWORDS = {
    # español
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "a", "ante",
    "con", "contra", "en", "entre", "hacia", "hasta", "para", "por", "según", "sin", "sobre",
    "tras", "y", "o", "u", "e", "que", "qué", "cual", "cuál", "cuales", "cuáles", "como",
    "cómo", "cuando", "cuándo", "donde", "dónde", "es", "son", "ser", "está", "están", "hay",
    "tiene", "tienen", "puede", "pueden", "más", "muy", "me", "te", "se", "le", "lo", "su",
    "sus", "mi", "tu", "este", "esta", "esto", "estos", "estas", "eso", "existen", "tipos",
    "explicame", "explícame", "dame", "decime", "contame", "quiero", "saber", "prevenirla",
    "prevenir", "acerca", "información", "informacion",
    # inglés
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are", "be", "what",
    "which", "how", "when", "where", "why", "can", "could", "does", "do", "about", "tell",
    "give", "explain", "types", "exist",
    # ruido de dominio (en casi todos los docs)
    "vulnerabilidad", "vulnerabilidades", "vulnerability", "descripcion", "descripción",
    "description", "cwe", "cve", "weakness", "debilidad",
}


def _extract_vuln_ids(text: str) -> list[str]:
    """IDs CWE/CVE normalizados a forma canónica (CWE-89, CVE-2023-1234), únicos y en orden."""
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
    """Términos de _SECURITY_TERMS presentes en el texto (ya en minúsculas)."""
    found = []
    for term in _SECURITY_TERMS:
        if " " in term or "-" in term:
            if term in text_low:
                found.append(term)
        elif re.search(rf"\b{re.escape(term)}\b", text_low):
            # límite de palabra para siglas cortas (dos/rce/lfi) y evitar substrings
            found.append(term)
    return found


def build_keyword_query(user_message: str) -> str:
    """Transforma la pregunta del usuario en una query de alta precisión para el
    keyword retriever (que ANDea todos los términos). Ver el encabezado de sección
    y docs/optimizacion_keyword_retrieval.md.
    """
    text = user_message.strip()

    ids = _extract_vuln_ids(text)
    if ids:
        # Un ID basta y da precisión máxima; con varios se toma el primero para no
        # romper el AND del full-text (una pregunta suele apuntar a un único ID).
        return ids[0]

    low = text.lower()
    terms = _find_security_terms(low)
    if terms:
        # El término más específico = el más largo (colapsa "xss" y su forma larga
        # en una sola frase coherente que co-ocurre en el doc objetivo).
        return max(terms, key=len)

    # Fallback: contenido sin stopwords ni ruido de dominio. Si queda vacío se
    # devuelve el texto original (comportamiento no peor que antes).
    tokens = re.findall(r"[a-záéíóúñü0-9][a-záéíóúñü0-9\-]*", low)
    keywords = [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]
    return " ".join(keywords) or text


# ======================================================================
# Factories del pipeline (módulo-level)
#
# Extraídas de los métodos de Pipeline para poder instanciar el store y el
# pipeline de query SIN arrastrar el runtime de Open WebUI (Valves, carga de
# marker-pdf, indexación). Las usa el harness de evaluación (eval/) para correr
# headless. La clase Pipeline delega en ellas → el runtime no cambia.
# Ver docs/eval/eval_harness.md.
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
    """Proveedor de generación activo: 'groq' (default) u 'ollama'. Ver docs/modos_llm.md."""
    return os.getenv("LLM_PROVIDER", "groq").strip().lower()


def build_generator(valves):
    """Devuelve el generador del LLM según LLM_PROVIDER (leído en tiempo de build, para
    que on_valves_updated y el eval headless respeten el modo activo).

    - groq   (default): OpenAIGenerator contra el endpoint OpenAI-compatible de Groq.
                        Sólo aquí se lee GROQ_API_KEY (Secret) → el modo ollama no la exige.
    - ollama          : OllamaGenerator local. Modelo = LLM_MODEL o DEFAULT_OLLAMA_LLM.
    """
    v = valves
    provider = _llm_provider()

    if provider == "ollama":
        model = os.getenv("LLM_MODEL") or DEFAULT_OLLAMA_LLM
        return OllamaGenerator(
            model=model,
            url=OLLAMA_URL,
            timeout=120,   # generación local puede tardar más que la API
            generation_kwargs={
                "num_predict": v.max_tokens,   # Ollama usa num_predict, no max_tokens
                "temperature": v.temperature,
            },
        )

    # Default: Groq (API Key). LLM_MODEL puede overridear el valve si se quiere.
    model = os.getenv("LLM_MODEL") or v.llm_model
    return OpenAIGenerator(
        api_key=Secret.from_env_var("GROQ_API_KEY"),
        api_base_url=GROQ_BASE_URL,
        model=model,
        max_retries=5,      # default 2 — el cliente respeta el retry-after de Groq ante 429
        timeout=60.0,
        generation_kwargs={
            "max_tokens":  v.max_tokens,
            "temperature": v.temperature,
        },
    )


def build_rag_pipeline(store: PgvectorDocumentStore, valves, include_llm: bool = True) -> HaystackPipeline:
    """Arma el pipeline de query: retrievers híbridos + joiner (RRF) + reranker + prompt + LLM.

    Los retrievers recuperan ancho (retriever_top_k), el joiner fusiona por RRF y el
    cross-encoder reranker (SentenceTransformersSimilarityRanker) re-puntúa y deja los
    ranker_top_k mejores docs en el prompt.

    `valves` es cualquier objeto con los atributos de Pipeline.Valves que usa el pipeline:
    embedding_model, retriever_top_k, ranker_model, ranker_top_k, llm_model, max_tokens,
    temperature. El generador (Groq o Ollama local) lo elige build_generator según LLM_PROVIDER.

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
    pipeline.add_component("document_joiner",     DocumentJoiner(
        join_mode="reciprocal_rank_fusion",
        top_k=v.retriever_top_k * 2,   # cap de candidatos que entran al reranker
    ))
    pipeline.add_component("ranker",              SentenceTransformersSimilarityRanker(
        model=v.ranker_model,
        top_k=v.ranker_top_k,          # techo duro de docs que llegan al prompt
    ))

    pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
    pipeline.connect("embedding_retriever",     "document_joiner")
    pipeline.connect("keyword_retriever",       "document_joiner")
    pipeline.connect("document_joiner",         "ranker.documents")

    if include_llm:
        pipeline.add_component("prompt_builder", PromptBuilder(template=PROMPT_TEMPLATE))
        pipeline.add_component("llm", build_generator(v))
        pipeline.connect("ranker",          "prompt_builder.documents")
        pipeline.connect("prompt_builder",  "llm")

    return pipeline


class Pipeline:

    class Valves(BaseModel):
        # Generación por API (Groq, endpoint compatible con OpenAI)
        llm_model:       str   = "meta-llama/llama-4-scout-17b-16e-instruct"
        # Embeddings locales en Ollama (bge-m3 = 1024 dims, coincide con EMBEDDING_DIMENSION)
        embedding_model: str   = "bge-m3"
        retriever_top_k: int   = 15    # recuperamos ancho; el reranker filtra
        # Cross-encoder reranker (compañero local de bge-m3, corre en CPU)
        ranker_model:    str   = "BAAI/bge-reranker-v2-m3"
        ranker_top_k:    int   = 4     # docs que realmente llegan al prompt (techo duro)
        max_tokens:      int   = 512
        temperature:     float = 0.5
        split_length:    int   = 200
        split_overlap:   int   = 20

    def __init__(self):
        self.name   = "RAG ciberseguridad"
        self.valves = self.Valves()
        self.store  = self._get_document_store()

        # marker-pdf carga modelos pesados (~3 GB residentes) y SOLO se usa para
        # convertir PDFs sin cachear. No se carga acá: se hace lazy en la primera
        # conversión real (ver _get_marker_converter) y se libera tras indexar. En
        # estado estable (PDFs ya convertidos en _converted_md) nunca se carga.
        self._marker_converter = None
        self._marker_load_failed = False

        self.rag_pipeline = self._build_rag_pipeline()
        logger.info("Indexando documentos en __init__...")
        self._index_new_documents()

    async def on_startup(self):
        logger.info("Indexando documentos nuevos si los hay...")
        self._index_new_documents()

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
        total = len(self.store.filter_documents())
        logger.info(f"Documentos en store al momento de la query: {total}")
        logger.info(f"Ejecutando RAG para: {user_message}")

        # El keyword retriever recibe una query depurada (IDs / términos de
        # seguridad / contenido sin stopwords), no la pregunta cruda.
        keyword_query = build_keyword_query(user_message)
        logger.info(f"[KEYWORD QUERY] -> {keyword_query!r}")

        result = self.rag_pipeline.run(
            {
                "text_embedder":     {"text": user_message},
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

        # --- Debug: documentos recuperados por cada retriever ---
        self._log_retrieved_docs(result)

        # --- Medición de tokens (relevancia del reranker: menos docs → menos tokens) ---
        self._log_token_usage(result)

        return result["llm"]["replies"][0]

    def _log_token_usage(self, result: dict) -> None:
        """Loguea el uso de tokens de la generación al txt.

        Groq (OpenAIGenerator) y Ollama (OllamaGenerator) reportan el uso con claves
        distintas en el meta del reply; normalizamos ambos a prompt/completion/total.
        También logueamos el tamaño del prompt (chars/palabras) y cuántos docs entraron,
        para poder medir el efecto del reranker en los tokens enviados al LLM.
        """
        sep = "-" * 60

        # Tamaño del prompt efectivo (lo que se le manda al LLM) y nº de docs al prompt.
        prompt = result.get("prompt_builder", {}).get("prompt") or ""
        n_docs = len(result.get("ranker", {}).get("documents", []))

        # Uso de tokens: el meta es una lista (un dict por reply); tomamos el primero.
        meta_list = result.get("llm", {}).get("meta") or []
        meta = meta_list[0] if meta_list else {}

        prompt_tokens = completion_tokens = total_tokens = None
        usage = meta.get("usage")
        if isinstance(usage, dict):
            # Groq / OpenAI-compatible
            prompt_tokens     = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens      = usage.get("total_tokens")
        else:
            # Ollama: prompt_eval_count (entrada) / eval_count (salida)
            prompt_tokens     = meta.get("prompt_eval_count")
            completion_tokens = meta.get("eval_count")

        if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

        logger.info(f"{sep}")
        logger.info(
            f"[TOKENS] docs_al_prompt={n_docs} | prompt_chars={len(prompt)} | "
            f"prompt_words={len(prompt.split())}"
        )
        if prompt_tokens is None and completion_tokens is None:
            model = meta.get("model", "desconocido")
            logger.info(f"[TOKENS] uso no reportado por el generador (model={model}).")
        else:
            logger.info(
                f"[TOKENS] prompt={prompt_tokens} | completion={completion_tokens} | "
                f"total={total_tokens} | model={meta.get('model', 'desconocido')}"
            )
        logger.info(f"{sep}")

    def _log_retrieved_docs(self, result: dict) -> None:
        """Loguea en detalle los documentos que cada retriever pasó al joiner."""

        sep = "-" * 60

        # Embedding retriever
        emb_docs = result.get("embedding_retriever", {}).get("documents", [])
        logger.info(f"{sep}")
        logger.info(f"[EMBEDDING RETRIEVER] {len(emb_docs)} documentos recuperados:")
        for i, doc in enumerate(emb_docs):
            score   = doc.score if doc.score is not None else "N/A"
            source  = doc.meta.get("source") or doc.meta.get("file_path", "desconocido")
            snippet = (doc.content or "")[:200].replace("\n", " ")
            logger.info(f"  [{i+1}] score={score:.4f} | fuente={source}")
            logger.info(f"        snippet: {snippet}...")

        # Keyword retriever
        kw_docs = result.get("keyword_retriever", {}).get("documents", [])
        logger.info(f"{sep}")
        logger.info(f"[KEYWORD RETRIEVER] {len(kw_docs)} documentos recuperados:")
        for i, doc in enumerate(kw_docs):
            score   = doc.score if doc.score is not None else "N/A"
            source  = doc.meta.get("source") or doc.meta.get("file_path", "desconocido")
            snippet = (doc.content or "")[:200].replace("\n", " ")
            logger.info(f"  [{i+1}] score={score} | fuente={source}")
            logger.info(f"        snippet: {snippet}...")

        # Joiner (candidatos fusionados que entran al reranker)
        joined_docs = result.get("document_joiner", {}).get("documents", [])
        logger.info(f"{sep}")
        logger.info(f"[DOCUMENT JOINER] {len(joined_docs)} candidatos fusionados (RRF) al reranker:")
        for i, doc in enumerate(joined_docs):
            source  = doc.meta.get("source") or doc.meta.get("file_path", "desconocido")
            snippet = (doc.content or "")[:300].replace("\n", " ")
            logger.info(f"  [{i+1}] fuente={source}")
            logger.info(f"        contenido: {snippet}...")

        # Reranker (lo que realmente llega al prompt)
        ranked_docs = result.get("ranker", {}).get("documents", [])
        logger.info(f"{sep}")
        logger.info(f"[RANKER] {len(ranked_docs)} documentos re-rankeados enviados al prompt:")
        for i, doc in enumerate(ranked_docs):
            score   = f"{doc.score:.4f}" if doc.score is not None else "N/A"
            source  = doc.meta.get("source") or doc.meta.get("file_path", "desconocido")
            snippet = (doc.content or "")[:300].replace("\n", " ")
            logger.info(f"  [{i+1}] score={score} | fuente={source}")
            logger.info(f"        contenido: {snippet}...")
        logger.info(f"{sep}")

    # ------------------------------------------------------------------
    # Infraestructura
    # ------------------------------------------------------------------

    def _get_document_store(self) -> PgvectorDocumentStore:
        return get_document_store()

    # ------------------------------------------------------------------
    # Conversión de PDFs con marker-pdf
    # ------------------------------------------------------------------

    def _get_marker_converter(self):
        """Carga marker-pdf on-demand y cachea la instancia. Devuelve el converter o
        None si falla (una sola vez; no reintenta en cada PDF). Cargar estos modelos
        cuesta ~3 GB de RAM, así que solo se hace cuando hay que convertir de verdad."""
        if self._marker_converter is not None:
            return self._marker_converter
        if self._marker_load_failed:
            return None
        logger.info("Cargando modelos de marker-pdf (lazy)...")
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            self._marker_converter = PdfConverter(artifact_dict=create_model_dict())
            logger.info("marker-pdf cargado correctamente.")
        except Exception as e:
            import traceback
            logger.error(f"No se pudo cargar marker-pdf: {e}")
            logger.error(f"Traceback completo:\n{traceback.format_exc()}")
            self._marker_load_failed = True
            self._marker_converter = None
        return self._marker_converter

    def _release_marker_converter(self) -> None:
        """Libera los modelos de marker-pdf de memoria tras indexar (recupera ~3 GB)."""
        if self._marker_converter is None:
            return
        logger.info("Liberando modelos de marker-pdf de memoria.")
        self._marker_converter = None
        import gc
        gc.collect()

    def _convert_pdf_with_marker(self, pdf_path: Path) -> Path | None:
        """
        Convierte un PDF a Markdown usando marker-pdf y lo guarda en CONVERTED_DIR.
        Si el .md ya existe (caché), lo reutiliza SIN cargar marker en memoria.
        Retorna el path del .md, o None si falla.
        """
        CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
        md_path = CONVERTED_DIR / (pdf_path.stem + ".md")

        # Cache hit: NO cargamos marker (el gran ahorro de RAM en estado estable).
        if md_path.exists():
            logger.info(f"Reutilizando conversión cacheada: {md_path.name}")
            return md_path

        # Cache miss: recién acá cargamos marker (lazy).
        converter = self._get_marker_converter()
        if converter is None:
            logger.warning(f"marker-pdf no disponible, salteando: {pdf_path.name}")
            return None

        logger.info(f"Convirtiendo con marker-pdf: {pdf_path.name}")
        try:
            from marker.output import text_from_rendered
            rendered    = converter(str(pdf_path))
            markdown_text, _, _ = text_from_rendered(rendered)
            md_path.write_text(markdown_text, encoding="utf-8")
            logger.info(f"Conversión exitosa → {md_path.name}")
            return md_path
        except Exception as e:
            logger.error(f"Error convirtiendo {pdf_path.name}: {e}")
            return None

    # ------------------------------------------------------------------
    # Carga de documentos locales
    # ------------------------------------------------------------------

    def _load_local_documents(self) -> tuple[list[Document], list[Document]]:
        """Devuelve (splittables, atomicos).

        - splittables: PDF/DOCX/MD/TXT → se chunkean con DocumentSplitter y se
          deduplican por (file_path, _split_id).
        - atomicos:    CWE (XML) y CVE (JSON) → id determinístico, NO se chunkean
          y se deduplican por id. Ver docs/indexing_design_capa1.md.
        """
        splittable: list[Document] = []
        atomic: list[Document] = []

        # --- PDFs: convertir primero a .md con marker-pdf ---
        pdf_files = list(INPUT_DIR.glob("*.pdf"))
        if pdf_files:
            md_paths     = []
            source_names = []   # para preservar el nombre del PDF original en meta

            for pdf_path in pdf_files:
                md_path = self._convert_pdf_with_marker(pdf_path)
                if md_path is not None:
                    md_paths.append(md_path)
                    source_names.append(pdf_path)

            if md_paths:
                md_converter = MarkdownToDocument()
                result = md_converter.run(sources=md_paths)
                for doc, original_pdf in zip(result["documents"], source_names):
                    # Apuntamos file_path al PDF original para que la deduplicación
                    # funcione igual que antes (basada en el archivo fuente real)
                    doc.meta["file_path"] = str(original_pdf)
                    doc.meta["source"]    = original_pdf.name
                splittable.extend(result["documents"])

        # --- DOCX, MD nativo y TXT: misma lógica que antes ---
        other_converters = [
            ("*.docx", DOCXToDocument()),
            ("*.md",   MarkdownToDocument()),
            ("*.txt",  TextFileToDocument()),
        ]
        for pattern, converter in other_converters:
            # Excluir los .md generados por marker (viven en CONVERTED_DIR, no en INPUT_DIR)
            files = [f for f in INPUT_DIR.glob(pattern) if f.parent == INPUT_DIR]
            if not files:
                continue
            result = converter.run(sources=files)
            splittable.extend(result["documents"])

        # --- XML de CWE → Documents atómicos (esquema MITRE cwe-7) ---
        xml_files = [f for f in INPUT_DIR.glob("*.xml") if f.parent == INPUT_DIR]
        if xml_files:
            atomic.extend(XMLCWEConverter().run(sources=xml_files)["documents"])

        # --- JSON de CVE (NVD) → Documents atómicos ---
        # fetch_nvd.py deja las páginas en rawdata/nvd/<corrida>/cves_page_*.json
        cve_files = sorted(INPUT_DIR.rglob("cves_page_*.json"))
        if cve_files:
            atomic.extend(NVDJsonConverter().run(sources=cve_files)["documents"])

        return splittable, atomic

    # ------------------------------------------------------------------
    # Indexación incremental
    # ------------------------------------------------------------------

    def _index_new_documents(self) -> None:
        if not INPUT_DIR.exists():
            logger.error(f"Directorio de entrada no encontrado: {INPUT_DIR}")
            return

        splittable_docs, atomic_docs = self._load_local_documents()

        # La conversión de PDFs ya terminó: si marker se cargó, lo liberamos antes
        # del embedding (paso también intensivo en memoria) para no acumular ~3 GB.
        self._release_marker_converter()

        if not splittable_docs and not atomic_docs:
            logger.warning("No se encontraron documentos en rawdata/")
            return

        # Solo los splittables (PDF/DOCX/MD/TXT) se chunkean. Los atómicos
        # (CWE/CVE) ya vienen dimensionados y con id determinístico.
        if splittable_docs:
            splitter = DocumentSplitter(
                split_by="word",
                split_length=self.valves.split_length,
                split_overlap=self.valves.split_overlap,
            )
            splittable_docs = splitter.run(documents=splittable_docs)["documents"]

        stored = self.store.filter_documents()

        # Dedup de splittables: por (file_path, _split_id), como antes.
        existing_keys = {
            (doc.meta.get("file_path"), doc.meta.get("_split_id"))
            for doc in stored
            if doc.meta.get("file_path") is not None
        }
        new_splittable = [
            doc for doc in splittable_docs
            if (doc.meta.get("file_path"), doc.meta.get("_split_id")) not in existing_keys
        ]

        # Dedup de atómicos: por id determinístico. Evita re-embeber en cada
        # arranque (contra el store) Y colapsa el mismo CWE repetido entre varios
        # catálogos XML dentro del mismo batch (gana la primera aparición).
        seen_ids = {doc.id for doc in stored}
        new_atomic = []
        for doc in atomic_docs:
            if doc.id in seen_ids:
                continue
            seen_ids.add(doc.id)
            new_atomic.append(doc)

        new_docs = new_splittable + new_atomic
        if not new_docs:
            logger.info("Sin documentos nuevos para indexar.")
            total = len(stored)
            logger.info(f"Total de documentos en el store: {total}")
            return

        logger.info(
            f"Embedding de {len(new_docs)} documentos nuevos "
            f"({len(new_splittable)} chunks + {len(new_atomic)} atómicos)..."
        )
        doc_embedder = OllamaDocumentEmbedder(
            model=self.valves.embedding_model,
            url=OLLAMA_URL,
            batch_size=32,
        )
        docs_with_embeddings = doc_embedder.run(new_docs)
        self.store.write_documents(
            docs_with_embeddings["documents"],
            policy=DuplicatePolicy.OVERWRITE,
        )
        total = len(self.store.filter_documents())
        logger.info(f"Indexación completa: {len(new_docs)} documentos agregados.")
        logger.info(f"Total de documentos en el store: {total}")

    # ------------------------------------------------------------------
    # Pipeline de RAG
    # ------------------------------------------------------------------

    def _build_rag_pipeline(self) -> HaystackPipeline:
        return build_rag_pipeline(self.store, self.valves)
