"""
title: Rag de Ciberseguridad
author: Valentino, Miguel y Luca
version: 1.3
requirements: haystack-ai, pgvector-haystack, ollama-haystack, pypdf, python-docx, markdown-it-py, mdit-plain
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
import logging

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
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.converters import (
    DOCXToDocument, MarkdownToDocument, TextFileToDocument
)
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder, OllamaDocumentEmbedder
from haystack.components.generators.openai import OpenAIGenerator
from haystack.document_stores.types import DuplicatePolicy

# --- Configuración fija de infraestructura ---
DB_CONNECTION       = "postgresql://avdbuser:avdbpass@vdb:5432/pgvdb"
OLLAMA_URL          = "http://ollama:11434"
GROQ_BASE_URL       = "https://api.groq.com/openai/v1"
DB_TABLE            = "ciberseguridad_docs"
EMBEDDING_DIMENSION = 1024
INPUT_DIR           = Path("/app/pipelines/rawdata")
CONVERTED_DIR       = INPUT_DIR / "_converted_md"   # caché de PDFs ya convertidos

PROMPT_TEMPLATE = """
Given the following information, answer the question.
YOU SHOULD ANSWER IN THE LANGUAGE OF THE QUESTION, NOT THE LANGUAGE OF THE DOCUMENTS.
If you don't know the answer, say you don't know. Do not make up an answer.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""


class Pipeline:

    class Valves(BaseModel):
        # Generación por API (Groq, endpoint compatible con OpenAI)
        llm_model:       str   = "llama-3.1-8b-instant"
        # Embeddings locales en Ollama (bge-m3 = 1024 dims, coincide con EMBEDDING_DIMENSION)
        embedding_model: str   = "bge-m3"
        retriever_top_k: int   = 5
        max_tokens:      int   = 1000
        temperature:     float = 0.5
        split_length:    int   = 200
        split_overlap:   int   = 20

    def __init__(self):
        self.name   = "RAG ciberseguridad"
        self.valves = self.Valves()
        self.store  = self._get_document_store()
        
        logger.info("Cargando modelos de marker-pdf...")
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            self._marker_converter = PdfConverter(artifact_dict=create_model_dict())
            logger.info("marker-pdf cargado correctamente.")
        except Exception as e:
            import traceback
            logger.error(f"No se pudo cargar marker-pdf: {e}")
            logger.error(f"Traceback completo:\n{traceback.format_exc()}")
            self._marker_converter = None

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

        result = self.rag_pipeline.run(
            {
                "text_embedder":     {"text": user_message},
                "keyword_retriever": {"query": user_message},
                "prompt_builder":    {"question": user_message},
            },
            include_outputs_from={
                "embedding_retriever",
                "keyword_retriever",
                "document_joiner",
            },
        )

        # --- Debug: documentos recuperados por cada retriever ---
        self._log_retrieved_docs(result)

        return result["llm"]["replies"][0]

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

        # Joiner (lo que realmente llega al prompt)
        joined_docs = result.get("document_joiner", {}).get("documents", [])
        logger.info(f"{sep}")
        logger.info(f"[DOCUMENT JOINER] {len(joined_docs)} documentos enviados al prompt:")
        for i, doc in enumerate(joined_docs):
            source  = doc.meta.get("source") or doc.meta.get("file_path", "desconocido")
            snippet = (doc.content or "")[:300].replace("\n", " ")
            logger.info(f"  [{i+1}] fuente={source}")
            logger.info(f"        contenido: {snippet}...")
        logger.info(f"{sep}")

    # ------------------------------------------------------------------
    # Infraestructura
    # ------------------------------------------------------------------

    def _get_document_store(self) -> PgvectorDocumentStore:
        return PgvectorDocumentStore(
            connection_string=Secret.from_token(DB_CONNECTION),
            embedding_dimension=EMBEDDING_DIMENSION,
            table_name=DB_TABLE,
            keyword_index_name="ciberseguridad_keyword_index",
        )

    # ------------------------------------------------------------------
    # Conversión de PDFs con marker-pdf
    # ------------------------------------------------------------------

    def _convert_pdf_with_marker(self, pdf_path: Path) -> Path | None:
        """
        Convierte un PDF a Markdown usando marker-pdf y lo guarda en CONVERTED_DIR.
        Si el .md ya existe (caché), lo reutiliza sin reconvertir.
        Retorna el path del .md, o None si falla.
        """
        if self._marker_converter is None:
            logger.warning(f"marker-pdf no disponible, salteando: {pdf_path.name}")
            return None

        CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
        md_path = CONVERTED_DIR / (pdf_path.stem + ".md")

        if md_path.exists():
            logger.info(f"Reutilizando conversión cacheada: {md_path.name}")
            return md_path

        logger.info(f"Convirtiendo con marker-pdf: {pdf_path.name}")
        try:
            from marker.output import text_from_rendered
            rendered    = self._marker_converter(str(pdf_path))
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

    def _load_local_documents(self) -> list[Document]:
        docs: list[Document] = []

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
                docs.extend(result["documents"])

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
            docs.extend(result["documents"])

        return docs

    # ------------------------------------------------------------------
    # Indexación incremental
    # ------------------------------------------------------------------

    def _index_new_documents(self) -> None:
        if not INPUT_DIR.exists():
            logger.error(f"Directorio de entrada no encontrado: {INPUT_DIR}")
            return

        docs = self._load_local_documents()
        if not docs:
            logger.warning("No se encontraron documentos en rawdata/")
            return

        splitter = DocumentSplitter(
            split_by="word",
            split_length=self.valves.split_length,
            split_overlap=self.valves.split_overlap,
        )
        docs = splitter.run(documents=docs)["documents"]

        existing_keys = {
            (doc.meta.get("file_path"), doc.meta.get("_split_id"))
            for doc in self.store.filter_documents()
            if doc.meta.get("file_path") is not None
        }

        new_docs = [
            doc for doc in docs
            if (doc.meta.get("file_path"), doc.meta.get("_split_id")) not in existing_keys
        ]

        if not new_docs:
            logger.info("Sin documentos nuevos para indexar.")
            total = len(self.store.filter_documents())
            logger.info(f"Total de documentos en el store: {total}")
            return

        logger.info(f"Embedding de {len(new_docs)} documentos nuevos...")
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
        logger.info(f"Indexación completa: {len(new_docs)} chunks agregados.")
        logger.info(f"Total de documentos en el store: {total}")

    # ------------------------------------------------------------------
    # Pipeline de RAG
    # ------------------------------------------------------------------

    def _build_rag_pipeline(self) -> HaystackPipeline:
        v = self.valves
        pipeline = HaystackPipeline()

        pipeline.add_component("text_embedder",       OllamaTextEmbedder(model=v.embedding_model, url=OLLAMA_URL))
        pipeline.add_component("embedding_retriever", PgvectorEmbeddingRetriever(document_store=self.store, top_k=v.retriever_top_k))
        pipeline.add_component("keyword_retriever",   PgvectorKeywordRetriever(document_store=self.store, top_k=v.retriever_top_k))
        pipeline.add_component("document_joiner",     DocumentJoiner(
            join_mode="reciprocal_rank_fusion",
        ))
        pipeline.add_component("prompt_builder",      PromptBuilder(template=PROMPT_TEMPLATE))
        pipeline.add_component("llm", OpenAIGenerator(
            api_key=Secret.from_env_var("GROQ_API_KEY"),
            api_base_url=GROQ_BASE_URL,
            model=v.llm_model,
            generation_kwargs={
                "max_tokens":  v.max_tokens,
                "temperature": v.temperature,
            },
        ))

        pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
        pipeline.connect("embedding_retriever",     "document_joiner")
        pipeline.connect("keyword_retriever",       "document_joiner")
        pipeline.connect("document_joiner",         "prompt_builder.documents")
        pipeline.connect("prompt_builder",          "llm")

        return pipeline
