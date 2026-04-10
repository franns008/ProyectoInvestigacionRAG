"""
title: Haystack RAG Pipeline
author: Francisco
version: 1.1
requirements: haystack-ai, pgvector-haystack, ollama-haystack, pypdf, python-docx, markdown-it-py, mdit-plain
"""

from typing import List, Union, Generator, Iterator
from pathlib import Path
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
        llm_model:       str   = "phi4"
        embedding_model: str   = "bge-m3"
        retriever_top_k: int   = 5
        num_ctx:         int   = 2048
        num_predict:     int   = 1000
        temperature:     float = 0.5
        split_length:    int   = 200
        split_overlap:   int   = 20

    def __init__(self):
        self.name   = "Mi Haystack RAG"
        self.valves = self.Valves()
        self.store  = self._get_document_store()
        self.rag_pipeline = self._build_rag_pipeline()

    async def on_startup(self):
        print("Indexando documentos nuevos si los hay...")
        self._index_new_documents()

    async def on_valves_updated(self):
        print("Valves actualizados — reconstruyendo pipeline...")
        self.rag_pipeline = self._build_rag_pipeline()

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        print(f"Ejecutando RAG para: {user_message}")

        result = self.rag_pipeline.run({
            "text_embedder": {"text": user_message},
            "keyword_retriever": {"query": user_message},
            "prompt_builder": {"question": user_message},
        })

        return result["llm"]["replies"][0]

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