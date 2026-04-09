from haystack import Pipeline, Document
from haystack.utils import Secret
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.retrievers.pgvector import (
    PgvectorEmbeddingRetriever,
    PgvectorKeywordRetriever,
)
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.builders import PromptBuilder
from haystack.components.joiners import DocumentJoiner
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack_integrations.components.generators.ollama import OllamaGenerator

from datasets import load_dataset

# --- Configuración ---
DB_CONNECTION = "postgresql://avdbuser:avdbpass@localhost:5433/pgvdb"
DB_TABLE = "seven_wonders"
EMBEDDING_DIMENSION = 1024
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "qwen2.5:1.5b"
OLLAMA_BASE_URL = "http://localhost:11434"
DATASET_NAME = "bilgeyucel/seven-wonders"
RETRIEVER_TOP_K = 5

PROMPT_TEMPLATE = """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""


def get_document_store() -> PgvectorDocumentStore:
    return PgvectorDocumentStore(
        connection_string=Secret.from_token(DB_CONNECTION),
        embedding_dimension=EMBEDDING_DIMENSION,
        table_name=DB_TABLE,
    )


def index_new_documents(store: PgvectorDocumentStore) -> None:
    dataset = load_dataset(DATASET_NAME, split="train")
    docs: list[Document] = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

    existing_keys = {
        (doc.meta.get("url"), doc.meta.get("_split_id"))
        for doc in store.filter_documents()
        if doc.meta.get("url") is not None and doc.meta.get("_split_id") is not None
    }

    new_docs = [
        doc for doc in docs
        if (doc.meta.get("url"), doc.meta.get("_split_id")) not in existing_keys
    ]

    if not new_docs:
        print("\nNo new documents to embed.")
        return

    print(f"\nEmbedding and writing {len(new_docs)} new documents to the database...")
    doc_embedder = OllamaDocumentEmbedder(
        model=EMBEDDING_MODEL,
        url=OLLAMA_BASE_URL,
        batch_size=32,
        keep_alive=-1,
    )
    docs_with_embeddings = doc_embedder.run(new_docs)
    store.write_documents(docs_with_embeddings["documents"], policy=DuplicatePolicy.OVERWRITE)


def build_rag_pipeline(store: PgvectorDocumentStore) -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component("text_embedder", OllamaTextEmbedder(model=EMBEDDING_MODEL, url=OLLAMA_BASE_URL))
    pipeline.add_component("embedding_retriever", PgvectorEmbeddingRetriever(document_store=store, top_k=RETRIEVER_TOP_K))
    pipeline.add_component("keyword_retriever", PgvectorKeywordRetriever(document_store=store, top_k=RETRIEVER_TOP_K))
    pipeline.add_component("document_joiner", DocumentJoiner())
    pipeline.add_component("prompt_builder", PromptBuilder(template=PROMPT_TEMPLATE))
    pipeline.add_component("llm", OllamaGenerator(
        model=LLM_MODEL,
        url=OLLAMA_BASE_URL,
        generation_kwargs={
            "num_predict": 1000,
            "temperature": 0.5,
            "num_ctx": 2048,
        },
        keep_alive=-1,
        timeout=450,
        streaming_callback=lambda chunk: print(chunk.content, end="", flush=True),
    ))

    pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
    pipeline.connect("embedding_retriever", "document_joiner")
    pipeline.connect("keyword_retriever", "document_joiner")
    pipeline.connect("document_joiner", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")

    return pipeline


def run_interactive_loop(pipeline: Pipeline) -> None:
    while True:
        question = input("\nWhat do you want to ask about the Seven Wonders of the Ancient World? (type 'exit' to quit) ")
        if question.lower() == "exit":
            print("Goodbye!")
            break
        pipeline.run({
            "text_embedder": {"text": question},
            "keyword_retriever": {"query": question},
            "prompt_builder": {"question": question},
        })
        print()


if __name__ == "__main__":
    store = get_document_store()
    index_new_documents(store)
    pipeline = build_rag_pipeline(store)
    run_interactive_loop(pipeline)
