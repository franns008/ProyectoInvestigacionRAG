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
import ollama

from rich.console import Console

# --- Configuración ---
DB_CONNECTION = "postgresql://avdbuser:avdbpass@localhost:5433/pgvdb"
DB_TABLE = "local_docs"
EMBEDDING_DIMENSION = 1024
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "ministral-3:3b"
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

console = Console()


class StreamState:
    """Coordina el spinner y el streaming del LLM en el loop interactivo."""
    def __init__(self):
        self.status = None
        self.started = False

    def reset(self):
        self.started = False

    def callback(self, chunk):
        if not self.started:
            if self.status:
                self.status.stop()
            console.print("\n[bold green]Respuesta:[/bold green] ", end="")
            self.started = True
        console.print(chunk.content, end="")

stream_state = StreamState()


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


def print_setup_summary(docs: list[Document], new_count: int, llm_model: str) -> None:
    console.rule("[bold cyan]Configuración[/bold cyan]")
    console.print(f"  [cyan]LLM[/cyan]              {llm_model}")
    console.print(f"  [cyan]Embedding model[/cyan]  {EMBEDDING_MODEL}")
    console.print(f"  [cyan]Ollama URL[/cyan]       {OLLAMA_BASE_URL}")
    console.print(f"  [cyan]DB table[/cyan]         {DB_TABLE}")
    console.print(f"  [cyan]Retriever top-k[/cyan]  {RETRIEVER_TOP_K}")
    console.print(f"  [cyan]Input dir[/cyan]        {INPUT_DIR}")

    console.rule("[bold cyan]Documentos[/bold cyan]")
    if docs:
        ext_counts: dict[str, int] = {}
        for doc in docs:
            ext = Path(doc.meta.get("file_path", "")).suffix.lower() or "unknown"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

        for ext, count in sorted(ext_counts.items()):
            console.print(f"  [cyan]{ext}[/cyan]  {count} fragmentos")

        index_status = f"[green]{new_count} nuevos indexados[/green]" if new_count else "[yellow]sin cambios[/yellow]"
        console.print(f"\n  Total: [bold]{len(docs)}[/bold] fragmentos — {index_status}")
    else:
        console.print("  [yellow]No se encontraron documentos en 'input/'[/yellow]")


def index_new_documents(store: PgvectorDocumentStore) -> tuple[list[Document], int]:
    if not INPUT_DIR.exists():
        console.print(f"[red]Directorio no encontrado:[/red] {INPUT_DIR}")
        console.print("Creá la carpeta [bold]input/[/bold] junto al script y agregá tus documentos (pdf, docx, md, txt).")
        return [], 0

    docs = load_local_documents(INPUT_DIR)

    existing_keys = {
        doc.meta.get("file_path")
        for doc in store.filter_documents()
        if doc.meta.get("file_path") is not None
    }

    new_docs = [
        doc for doc in docs
        if doc.meta.get("file_path") not in existing_keys
    ]

    if new_docs:
        console.print(f"\n[cyan]Indexando {len(new_docs)} documentos nuevos...[/cyan]")
        doc_embedder = OllamaDocumentEmbedder(
            model=EMBEDDING_MODEL,
            url=OLLAMA_BASE_URL,
            batch_size=32,
            keep_alive=-1,
            timeout=450,
        )
        docs_with_embeddings = doc_embedder.run(new_docs)
        store.write_documents(docs_with_embeddings["documents"], policy=DuplicatePolicy.OVERWRITE)

    return docs, len(new_docs)


def build_rag_pipeline(store: PgvectorDocumentStore, llm_model: str = LLM_MODEL) -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component("text_embedder", OllamaTextEmbedder(model=EMBEDDING_MODEL, url=OLLAMA_BASE_URL))
    pipeline.add_component("embedding_retriever", PgvectorEmbeddingRetriever(document_store=store, top_k=RETRIEVER_TOP_K))
    pipeline.add_component("keyword_retriever", PgvectorKeywordRetriever(document_store=store, top_k=RETRIEVER_TOP_K))
    pipeline.add_component("document_joiner", DocumentJoiner())
    pipeline.add_component("prompt_builder", PromptBuilder(template=PROMPT_TEMPLATE))
    pipeline.add_component("llm", OllamaGenerator(
        model=llm_model,
        url=OLLAMA_BASE_URL,
        generation_kwargs={
            "num_predict": 1000,
            "temperature": 0.5,
            "num_ctx": 2048,
        },
        keep_alive=-1,
        timeout=450,
        streaming_callback=stream_state.callback,
    ))

    pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
    pipeline.connect("embedding_retriever", "document_joiner")
    pipeline.connect("keyword_retriever", "document_joiner")
    pipeline.connect("document_joiner", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")

    return pipeline


def run_interactive_loop(pipeline: Pipeline) -> None:
    console.print("\n[dim]Escribí [bold]exit[/bold] para salir.[/dim]\n")

    while True:
        try:
            question = console.input("[bold cyan]Pregunta:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not question:
            continue
        if question.lower() == "exit":
            console.print("[dim]Goodbye![/dim]")
            break

        stream_state.reset()
        stream_state.status = None
        pipeline.run({
            "text_embedder": {"text": question},
            "keyword_retriever": {"query": question},
            "prompt_builder": {"question": question},
        })
        console.print("\n")


def select_llm_model() -> str:
    try:
        models = [m.model for m in ollama.list().models]
    except Exception:
        console.print("[red]No se pudo conectar a Ollama. Verificá que esté corriendo.[/red]")
        raise SystemExit(1)

    if not models:
        console.print("[red]No hay modelos instalados en Ollama.[/red]")
        raise SystemExit(1)

    console.rule("[bold cyan]Seleccioná un modelo LLM[/bold cyan]")
    for i, model in enumerate(models, 1):
        console.print(f"  [cyan]{i}[/cyan]  {model}")
    console.print()

    while True:
        raw = console.input("[bold cyan]Modelo:[/bold cyan] ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            return models[int(raw) - 1]
        console.print(f"[yellow]Ingresá un número entre 1 y {len(models)}.[/yellow]")


if __name__ == "__main__":
    llm_model = select_llm_model()
    store = get_document_store()
    docs, new_count = index_new_documents(store)
    print_setup_summary(docs, new_count, llm_model)
    pipeline = build_rag_pipeline(store, llm_model)
    run_interactive_loop(pipeline)
