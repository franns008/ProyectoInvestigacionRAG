"""
title: Haystack RAG Pipeline (Groq LLM)
author: Francisco / Luca
version: 1.0
requirements: haystack-ai, pgvector-haystack, ollama-haystack, pypdf, python-docx, markdown-it-py, mdit-plain, openai
"""

import re
from typing import List, Union, Generator, Iterator
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel

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
    PyPDFToDocument, DOCXToDocument, MarkdownToDocument, TextFileToDocument
)
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder, OllamaDocumentEmbedder
from haystack.components.generators.openai import OpenAIGenerator
from haystack.document_stores.types import DuplicatePolicy

# --- Configuración fija de infraestructura ---
LOG_FILE            = Path("/app/pipelines/logs.txt")

DB_CONNECTION       = "postgresql://avdbuser:avdbpass@vdb:5432/pgvdb"
OLLAMA_URL          = "http://ollama:11434"
GROQ_BASE_URL       = "https://api.groq.com/openai/v1"
DB_TABLE            = "local_docs"
EMBEDDING_DIMENSION = 1024
INPUT_DIR           = Path("/app/pipelines/rawdata")

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


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line)


class Pipeline:

    class Valves(BaseModel):
        llm_model:       str   = "llama-3.1-8b-instant"
        embedding_model: str   = "bge-m3"
        retriever_top_k: int   = 3
        temperature:     float = 0.5
        max_tokens:      int   = 1000
        split_length:    int   = 200
        split_overlap:   int   = 20

    def __init__(self):
        self.name              = "Mi Haystack RAG (Groq)"
        self.valves            = self.Valves()
        self.store             = self._get_document_store()
        self.retrieval_pipeline = None
        self.rag_pipeline       = None

    async def on_startup(self):
        try:
            self.retrieval_pipeline = self._build_retrieval_pipeline()
            self.rag_pipeline       = self._build_generation_pipeline()
        except Exception as e:
            print(f"[ERROR] No se pudo construir el pipeline: {e}")
            return
        print("Indexando documentos nuevos si los hay...")
        self._index_new_documents()

    async def on_valves_updated(self):
        print("Valves actualizados — reconstruyendo pipeline...")
        self.retrieval_pipeline = self._build_retrieval_pipeline()
        self.rag_pipeline       = self._build_generation_pipeline()

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        if self.retrieval_pipeline is None or self.rag_pipeline is None:
            self.retrieval_pipeline = self._build_retrieval_pipeline()
            self.rag_pipeline       = self._build_generation_pipeline()

        if user_message.startswith("### Task:"):
            result = self.rag_pipeline.run({
                "prompt_builder": {"question": user_message, "documents": []},
            })
            return result.get("llm", {}).get("replies", [""])[0]

        RAG = "RAG-GROQ"
        log(f"{RAG} ========== INICIO ==========")
        log(f"{RAG} Pregunta: '{user_message}'")
        log(f"{RAG} Modelo: {self.valves.llm_model} | top_k: {self.valves.retriever_top_k}")

        # Paso 1: recuperar documentos
        retrieval_result = self.retrieval_pipeline.run({
            "text_embedder": {"text": user_message},
            "keyword_retriever": {"query": user_message},
        }, include_outputs_from={"embedding_retriever", "keyword_retriever"})

        emb_docs = retrieval_result.get("embedding_retriever", {}).get("documents", [])
        kw_docs  = retrieval_result.get("keyword_retriever",  {}).get("documents", [])

        log(f"{RAG} --- RETRIEVAL ---")
        log(f"{RAG} Embedding docs: {len(emb_docs)} | Keyword docs: {len(kw_docs)}")
        for i, doc in enumerate(emb_docs, 1):
            chars = len(doc.content or "")
            log(f"{RAG}   [EMB {i}] {doc.meta.get('file_path','?')} | score={getattr(doc,'score','N/A'):.4f} | {chars} chars")
        for i, doc in enumerate(kw_docs, 1):
            chars = len(doc.content or "")
            log(f"{RAG}   [KW  {i}] {doc.meta.get('file_path','?')} | score={getattr(doc,'score','N/A')} | {chars} chars")

        # Paso 2: normalizar whitespace antes de enviar al LLM
        joined_docs = retrieval_result.get("document_joiner", {}).get("documents", [])
        log(f"{RAG} --- DOCS TRAS DEDUPLICAR: {len(joined_docs)} ---")
        total_chars_antes = 0
        for i, doc in enumerate(joined_docs, 1):
            chars_orig = len(doc.content or "")
            total_chars_antes += chars_orig
            if doc.content:
                doc.content = re.sub(r'\s+', ' ', doc.content).strip()
            log(f"{RAG}   [{i}] {doc.meta.get('file_path','?')} | {chars_orig} chars → {len(doc.content or '')} chars (tras normalizar)")

        total_chars_despues = sum(len(d.content or "") for d in joined_docs)
        log(f"{RAG} Total chars en contexto: {total_chars_antes} antes / {total_chars_despues} después de normalizar")
        log(f"{RAG} Aprox tokens en contexto: ~{total_chars_despues // 4}")

        # Paso 3: generar respuesta con documentos truncados
        result = self.rag_pipeline.run({
            "prompt_builder": {"question": user_message, "documents": joined_docs},
        }, include_outputs_from={"prompt_builder"})

        prompt_generado = result.get("prompt_builder", {}).get("prompt", "")
        log(f"{RAG} --- PROMPT ENVIADO A GROQ ---")
        log(f"{RAG} Tamaño: {len(prompt_generado)} chars | ~{len(prompt_generado)//4} tokens")
        log(f"{RAG} Contenido:\n{prompt_generado}")
        log(f"{RAG} --- FIN PROMPT ---")

        respuesta_llm = result.get("llm", {}).get("replies", [""])[0]
        log(f"{RAG} --- RESPUESTA LLM ---")
        log(f"{RAG} {len(respuesta_llm)} chars | ~{len(respuesta_llm)//4} tokens")
        log(f"{RAG} {respuesta_llm}")
        log(f"{RAG} ========== FIN ==========")
        return respuesta_llm

    def _get_document_store(self) -> PgvectorDocumentStore:
        return PgvectorDocumentStore(
            connection_string=Secret.from_token(DB_CONNECTION),
            embedding_dimension=EMBEDDING_DIMENSION,
            table_name=DB_TABLE,
        )

    def _load_local_documents(self) -> list[Document]:
        docs: list[Document] = []
        converters = [
            ("*.pdf",  PyPDFToDocument()),
            ("*.docx", DOCXToDocument()),
            ("*.md",   MarkdownToDocument()),
            ("*.txt",  TextFileToDocument()),
        ]
        for pattern, converter in converters:
            files = list(INPUT_DIR.glob(pattern))
            if not files:
                continue
            result = converter.run(sources=files)
            docs.extend(result["documents"])
        return docs

    def _index_new_documents(self) -> None:
        if not INPUT_DIR.exists():
            print(f"Directorio de entrada no encontrado: {INPUT_DIR}")
            return

        docs = self._load_local_documents()
        if not docs:
            print("No se encontraron documentos en rawdata/")
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
            print("Sin documentos nuevos para indexar.")
            return

        print(f"Embedding de {len(new_docs)} documentos nuevos...")
        doc_embedder = OllamaDocumentEmbedder(
            model=self.valves.embedding_model,
            url=OLLAMA_URL,
            batch_size=32,
        )
        docs_with_embeddings = doc_embedder.run(new_docs)
        self.store.write_documents(docs_with_embeddings["documents"], policy=DuplicatePolicy.OVERWRITE)

    def _build_retrieval_pipeline(self) -> HaystackPipeline:
        v = self.valves
        pipeline = HaystackPipeline()
        pipeline.add_component("text_embedder", OllamaTextEmbedder(model=v.embedding_model, url=OLLAMA_URL))
        pipeline.add_component("embedding_retriever", PgvectorEmbeddingRetriever(document_store=self.store, top_k=v.retriever_top_k))
        pipeline.add_component("keyword_retriever", PgvectorKeywordRetriever(document_store=self.store, top_k=v.retriever_top_k))
        pipeline.add_component("document_joiner", DocumentJoiner())

        pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
        pipeline.connect("embedding_retriever", "document_joiner")
        pipeline.connect("keyword_retriever", "document_joiner")
        return pipeline

    def _build_generation_pipeline(self) -> HaystackPipeline:
        v = self.valves
        pipeline = HaystackPipeline()
        pipeline.add_component("prompt_builder", PromptBuilder(template=PROMPT_TEMPLATE))
        pipeline.add_component("llm", OpenAIGenerator(
            api_key=Secret.from_env_var("GROQ_API_KEY"),
            api_base_url=GROQ_BASE_URL,
            model=v.llm_model,
            generation_kwargs={
                "temperature": v.temperature,
                "max_tokens":  v.max_tokens,
            },
        ))

        pipeline.connect("prompt_builder", "llm")
        return pipeline
