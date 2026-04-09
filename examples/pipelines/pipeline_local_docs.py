"""
title: Haystack RAG Pipeline
author: Francisco
version: 1.0
requirements: haystack-ai, pgvector-haystack, ollama-haystack, pypdf, python-docx, markdown-it-py, mdit_plain
"""

from typing import List, Union, Generator, Iterator
import os
from pathlib import Path

from haystack import Pipeline as HaystackPipeline, Document
from haystack.utils import Secret
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.retrievers.pgvector import (
    PgvectorEmbeddingRetriever,
    PgvectorKeywordRetriever,
)
from haystack.components.builders import PromptBuilder
from haystack.components.joiners import DocumentJoiner
from haystack.components.converters import (
    PyPDFToDocument, DOCXToDocument, MarkdownToDocument, TextFileToDocument
)
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder, OllamaDocumentEmbedder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.document_stores.types import DuplicatePolicy

class Pipeline:
    def __init__(self):
        self.name = "Mi Haystack RAG"
        
        self.db_connection = "postgresql://avdbuser:avdbpass@vdb:5432/pgvdb"
        self.ollama_url = "http://ollama:11434"
        self.db_table = "local_docs"
        self.embedding_dimension = 1024
        self.embedding_model = "bge-m3"
        self.llm_model = "phi4"
        self.retriever_top_k = 5
        
        # Ruta corregida para Docker
        self.input_dir = Path("/app/pipelines/rawdata")
        
        self.prompt_template = """
        Given the following information, answer the question.
        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}
        Question: {{question}}
        Answer:
        """
        
        self.store = self._get_document_store()
        self.rag_pipeline = self._build_rag_pipeline()

    async def on_startup(self):
        print("Indexando documentos nuevos si los hay...")
        self._index_new_documents()

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        print(f"Ejecutando RAG para: {user_message}")
        
        result = self.rag_pipeline.run({
            "text_embedder": {"text": user_message},
            "keyword_retriever": {"query": user_message},
            "prompt_builder": {"question": user_message},
        })
        
        answer = result["llm"]["replies"][0]
        return answer

    def _get_document_store(self) -> PgvectorDocumentStore:
        return PgvectorDocumentStore(
            connection_string=Secret.from_token(self.db_connection),
            embedding_dimension=self.embedding_dimension,
            table_name=self.db_table,
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
            files = list(self.input_dir.glob(pattern))
            if not files:
                continue
            result = converter.run(sources=files)
            docs.extend(result["documents"])
        return docs

    def _index_new_documents(self) -> None:
        if not self.input_dir.exists():
            print(f"Directorio de entrada no encontrado: {self.input_dir}")
            return

        docs = self._load_local_documents()
        if not docs:
            return

        existing_keys = {
            doc.meta.get("file_path")
            for doc in self.store.filter_documents()
            if doc.meta.get("file_path") is not None
        }

        new_docs = [doc for doc in docs if doc.meta.get("file_path") not in existing_keys]
        if not new_docs:
            return

        print(f"Embedding de {len(new_docs)} documentos nuevos...")
        doc_embedder = OllamaDocumentEmbedder(
            model=self.embedding_model,
            url=self.ollama_url,
            batch_size=32,
        )
        docs_with_embeddings = doc_embedder.run(new_docs)
        self.store.write_documents(docs_with_embeddings["documents"], policy=DuplicatePolicy.OVERWRITE)

    def _build_rag_pipeline(self) -> HaystackPipeline:
        pipeline = HaystackPipeline()
        pipeline.add_component("text_embedder", OllamaTextEmbedder(model=self.embedding_model, url=self.ollama_url))
        pipeline.add_component("embedding_retriever", PgvectorEmbeddingRetriever(document_store=self.store, top_k=self.retriever_top_k))
        pipeline.add_component("keyword_retriever", PgvectorKeywordRetriever(document_store=self.store, top_k=self.retriever_top_k))
        pipeline.add_component("document_joiner", DocumentJoiner())
        pipeline.add_component("prompt_builder", PromptBuilder(template=self.prompt_template))
        pipeline.add_component("llm", OllamaGenerator(
            model=self.llm_model,
            url=self.ollama_url,
            generation_kwargs={"num_predict": 1000, "temperature": 0.5, "num_ctx": 2048}
        ))

        pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
        pipeline.connect("embedding_retriever", "document_joiner")
        pipeline.connect("keyword_retriever", "document_joiner")
        pipeline.connect("document_joiner", "prompt_builder.documents")
        pipeline.connect("prompt_builder", "llm")
        return pipeline