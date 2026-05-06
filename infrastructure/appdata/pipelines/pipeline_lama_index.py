"""
title: RAG lama index Pipeline
description: Pipeline de ejemplo que utiliza LlamaIndex para cargar documentos locales y Haystack para construir un pipeline RAG.
author: Francisco
version: 1.1
requirements: haystack-ai, pgvector-haystack, ollama-haystack, pypdf, python-docx, markdown-it-py, mdit-plain, docx2txt, llama-index
"""

from typing import List, Union, Generator, Iterator
from pathlib import Path
from pydantic import BaseModel
import logging

# Configurar el logger
logging.basicConfig(
    filename='/app/pipelines/logLlamaIndex.txt',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    force=True
)
logger = logging.getLogger("LlamaIndexRAG")

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
from llama_index.core import SimpleDirectoryReader, Document as LlamaDocument
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder, OllamaDocumentEmbedder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.document_stores.types import DuplicatePolicy

# --- Configuración fija de infraestructura ---
DB_CONNECTION      = "postgresql://avdbuser:avdbpass@vdb:5432/pgvdb"
OLLAMA_URL         = "http://ollama:11434"
DB_TABLE           = "local_docs"
EMBEDDING_DIMENSION = 1024
INPUT_DIR          = Path("/app/pipelines/rawdata")

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
        llm_model:       str   = "llama3.1:8b"
        embedding_model: str   = "bge-m3"
        retriever_top_k: int   = 5
        num_ctx:         int   = 2048
        num_predict:     int   = 1000
        temperature:     float = 0.5
        split_length:    int   = 200
        split_overlap:   int   = 20

    def __init__(self):
        self.name   = "RAG lama index"
        self.valves = self.Valves()
        self.store  = self._get_document_store()
        self.rag_pipeline = self._build_rag_pipeline()

    async def on_startup(self):
        logger.info("Indexando documentos nuevos si los hay...")
        self._index_new_documents()

    async def on_valves_updated(self):
        logger.info("Valves actualizados — reconstruyendo pipeline...")
        self.rag_pipeline = self._build_rag_pipeline()

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        logger.debug(f"[DEBUG-PIPE] Iniciando ejecución de RAG Pipeline.")
        logger.debug(f"[DEBUG-PIPE] Mensaje del usuario: '{user_message}'")
        logger.debug(f"[DEBUG-PIPE] ID Modelo entrante: {model_id} | Usando configurado: {self.valves.llm_model}")
        logger.debug(f"[DEBUG-PIPE] Historial de mensajes recibido (cantidad): {len(messages)}")

        # Habilitamos la extracción de los outputs intermedios agregando `include_outputs_from`
        result = self.rag_pipeline.run({
            "text_embedder": {"text": user_message},
            "keyword_retriever": {"query": user_message},
            "prompt_builder": {"question": user_message},
        }, include_outputs_from={"prompt_builder", "embedding_retriever", "keyword_retriever"})

        # --- RECOPILACION DE METRICAS ---
        # 1. Documentos Recuperados
        emb_docs = result.get("embedding_retriever", {}).get("documents", [])
        kw_docs = result.get("keyword_retriever", {}).get("documents", [])
        total_retrieved = len(emb_docs) + len(kw_docs)
        logger.info(f"[METRICS] Documentos recuperados por embeddings: {len(emb_docs)}")
        if emb_docs:
            logger.debug("[DEBUG-DOCS] --- Detalles Documentos Embeddings ---")
            for i, doc in enumerate(emb_docs, 1):
                filepath = doc.meta.get("file_path", "Desconocido")
                score = getattr(doc, "score", "N/A")
                logger.debug(f"  {i}. Archivo: {filepath} | Score: {score}")

        logger.info(f"[METRICS] Documentos recuperados por keywords: {len(kw_docs)}")
        if kw_docs:
            logger.debug("[DEBUG-DOCS] --- Detalles Documentos Keywords ---")
            for i, doc in enumerate(kw_docs, 1):
                filepath = doc.meta.get("file_path", "Desconocido")
                score = getattr(doc, "score", "N/A")
                logger.debug(f"  {i}. Archivo: {filepath} | Score: {score}")

        logger.info(f"[METRICS] Total documentos (antes de deduplicar): {total_retrieved}")

        # 2. Prompt Builder
        prompt_generado = result.get("prompt_builder", {}).get("prompt", "")
        if prompt_generado:
            words_count = len(prompt_generado.split())
            logger.info(f"[METRICS] Cantidad de palabras (aprox tokens) del prompt enviado al LLM: {words_count}")
            logger.debug(f"[DEBUG-PROMPT] Contenido completo del prompt:\n{prompt_generado}\n--- Fin Prompt ---")
        else:
            logger.debug("[DEBUG-PROMPT] Advertencia: No se pudo capturar el output del prompt_builder en el pipeline run.")

        # 3. Respuesta LLM
        respuesta_llm = result.get("llm", {}).get("replies", [""])[0]
        resp_words_count = len(respuesta_llm.split())
        logger.info(f"[METRICS] Cantidad de palabras (aprox tokens) de la respuesta del LLM: {resp_words_count}")
        logger.debug(f"[DEBUG-LLM] Respuesta completa:\n{respuesta_llm}\n--- Fin Respuesta ---")

        logger.debug(f"[DEBUG-PIPE] Fin ejecución RAG Pipeline.")
        return respuesta_llm

    def _get_document_store(self) -> PgvectorDocumentStore:
        return PgvectorDocumentStore(
            connection_string=Secret.from_token(DB_CONNECTION),
            embedding_dimension=EMBEDDING_DIMENSION,
            table_name=DB_TABLE,
        )

    def _load_local_documents(self) -> list[Document]:
        
        """
            
            
            Recorrer los archivos, buscando una extension y transformarla en otra.
        """
        reader = SimpleDirectoryReader(str(INPUT_DIR))
        llama_documents = reader.load_data()
        
        # --- CONVERSIÓN: LlamaIndex -> Haystack ---
        haystack_docs = []
        for l_doc in llama_documents:
            # Creamos un Document de Haystack usando la data del Document de LlamaIndex
            haystack_doc = Document(
                content=l_doc.text,
                meta=l_doc.metadata
            )
            haystack_docs.append(haystack_doc)
            
        return haystack_docs
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

        new_docs = [doc for doc in docs if (doc.meta.get("file_path"), doc.meta.get("_split_id")) not in existing_keys]
        if not new_docs:
            logger.info("Sin documentos nuevos para indexar.")
            return

        logger.info(f"Embedding de {len(new_docs)} documentos nuevos...")
        doc_embedder = OllamaDocumentEmbedder(
            model=self.valves.embedding_model,
            url=OLLAMA_URL,
            batch_size=32,
        )
        docs_with_embeddings = doc_embedder.run(new_docs)
        self.store.write_documents(docs_with_embeddings["documents"], policy=DuplicatePolicy.OVERWRITE)

    def _build_rag_pipeline(self) -> HaystackPipeline:
        v = self.valves
        pipeline = HaystackPipeline()
        pipeline.add_component("text_embedder", OllamaTextEmbedder(model=v.embedding_model, url=OLLAMA_URL))
        pipeline.add_component("embedding_retriever", PgvectorEmbeddingRetriever(document_store=self.store, top_k=v.retriever_top_k))
        pipeline.add_component("keyword_retriever", PgvectorKeywordRetriever(document_store=self.store, top_k=v.retriever_top_k))
        pipeline.add_component("document_joiner", DocumentJoiner())
        pipeline.add_component("prompt_builder", PromptBuilder(template=PROMPT_TEMPLATE))
        pipeline.add_component("llm", OllamaGenerator(
            model=v.llm_model,
            url=OLLAMA_URL,
            generation_kwargs={
                "num_predict": v.num_predict,
                "temperature": v.temperature,
                "num_ctx":     v.num_ctx,
            },
        ))

        pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
        pipeline.connect("embedding_retriever", "document_joiner")
        pipeline.connect("keyword_retriever", "document_joiner")
        pipeline.connect("document_joiner", "prompt_builder.documents")
        pipeline.connect("prompt_builder", "llm")
        return pipeline