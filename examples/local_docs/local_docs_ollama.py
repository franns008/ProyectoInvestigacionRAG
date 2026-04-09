from pathlib import Path

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
from haystack.components.converters import (
    PyPDFToDocument,
    DOCXToDocument,
    MarkdownToDocument,
    TextFileToDocument,
)
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack_integrations.components.generators.ollama import OllamaGenerator

# --- Configuración ---
DB_CONNECTION = "postgresql://avdbuser:avdbpass@localhost:5433/pgvdb"
DB_TABLE = "local_docs"
EMBEDDING_DIMENSION = 1024
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "qwen2.5:1.5b"
OLLAMA_BASE_URL = "http://localhost:11434"
RETRIEVER_TOP_K = 5
INPUT_DIR = Path(__file__).parent / "input"

PROMPT_TEMPLATE = """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""


def load_local_documents(input_dir: Path) -> list[Document]:
    docs: list[Document] = []

    converters: list[tuple[str, object]] = [
        ("*.pdf",  PyPDFToDocument()),
        ("*.docx", DOCXToDocument()),
        ("*.md",   MarkdownToDocument()),
        ("*.txt",  TextFileToDocument()),
    ]

    for pattern, converter in converters:
        files = list(input_dir.glob(pattern))
        if not files:
            continue
        result = converter.run(sources=files)
        docs.extend(result["documents"])

    return docs


def get_document_store() -> PgvectorDocumentStore:
    return PgvectorDocumentStore(
        connection_string=Secret.from_token(DB_CONNECTION),
        embedding_dimension=EMBEDDING_DIMENSION,
        table_name=DB_TABLE,
    )


def index_new_documents(store: PgvectorDocumentStore) -> None:
    if not INPUT_DIR.exists():
        print(f"\nInput directory not found: {INPUT_DIR}")
        print("Creá la carpeta 'input' junto al script y agregá tus documentos (pdf, docx, md, txt).")
        return

    docs = load_local_documents(INPUT_DIR)

    if not docs:
        print("\nNo se encontraron documentos en el directorio 'input'.")
        return

    existing_keys = {
        doc.meta.get("file_path")
        for doc in store.filter_documents()
        if doc.meta.get("file_path") is not None
    }

    new_docs = [
        doc for doc in docs
        if doc.meta.get("file_path") not in existing_keys
    ]

    if not new_docs:
        print("\nNo hay documentos nuevos para indexar.")
        return

    print(f"\nEmbedding y escritura de {len(new_docs)} documentos nuevos en la base de datos...")
    doc_embedder = OllamaDocumentEmbedder(
        model=EMBEDDING_MODEL,
        url=OLLAMA_BASE_URL,
        batch_size=32,
        keep_alive=-1,
        timeout=450,
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
        question = input("\nHacé una pregunta sobre tus documentos (escribí 'exit' para salir): ")
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
