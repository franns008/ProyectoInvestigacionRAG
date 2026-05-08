from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import os

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

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
from haystack.components.preprocessors import DocumentSplitter
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

# --- Configuración fija ---
DB_CONNECTION       = "postgresql://avdbuser:avdbpass@localhost:5433/pgvdb"
DB_TABLE            = "local_docs"
EMBEDDING_MODEL     = "bge-m3"
EMBEDDING_DIMENSION = 1024
OLLAMA_BASE_URL     = "http://localhost:11434"
INPUT_DIR           = Path(__file__).parent / "rawdata/markdown"
LOG_FILE            = Path(__file__).parent / "log.txt"

# Modelos recomendados; se muestran primero si están instalados
RECOMMENDED_MODELS = [
    "llama3.2:3b",
    "llama3.1:8b",
    "llama3.3:70b",
    "gemma3:4b",
    "gemma3:12b",
    "qwen2.5:7b",
    "mistral:7b",
    "phi4:14b",
]

PROMPT_TEMPLATE = """
Given the following information, answer the question.
YOU SHOULD ANSWER IN THE LANGUAGE OF THE QUESTION, NOT THE LANGUAGE OF THE DOCUMENTS.
If you don't know the answer, say you don't know. Do not try to make up an answer.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}


Question: {{question}}
Answer:
"""

console = Console()

TEMPERATURE = 0.5


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


@dataclass
class Config:
    llm_model:    str
    num_ctx:      int
    num_predict:  int
    top_k:        int
    split_length: int
    split_overlap: int


# --- Setup helpers ---

def pick(title: str, options: list[tuple[str, Any]], default: int) -> Any:
    console.rule(f"[bold cyan]{title}[/bold cyan]")
    for i, (label, _) in enumerate(options, 1):
        marker = "  [dim]← predeterminado[/dim]" if i == default else ""
        console.print(f"  [cyan]{i}[/cyan]  {label}{marker}")
    console.print()
    while True:
        raw = console.input(f"[bold cyan]Opción [{default}]:[/bold cyan] ").strip()
        if raw == "":
            return options[default - 1][1]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][1]
        console.print(f"[yellow]Ingresá un número entre 1 y {len(options)}, o Enter para el predeterminado.[/yellow]")


def get_sorted_models(installed: list[str]) -> list[str]:
    """Devuelve los modelos instalados: primero los recomendados, luego el resto."""
    installed_set = set(installed)
    ordered = [m for m in RECOMMENDED_MODELS if m in installed_set]
    extras  = [m for m in installed if m not in set(RECOMMENDED_MODELS)]
    return ordered + extras


def run_setup() -> Config:
    try:
        installed = [m.model for m in ollama.list().models]
    except Exception:
        console.print("[red]No se pudo conectar a Ollama. Verificá que esté corriendo en localhost:11434.[/red]")
        raise SystemExit(1)

    if not installed:
        console.print("[red]No hay modelos instalados en Ollama.[/red]")
        raise SystemExit(1)

    models = get_sorted_models(installed)

    llm_model = pick(
        "Seleccioná un modelo LLM (Ollama)",
        [(m, m) for m in models],
        default=1,
    )

    num_ctx = pick(
        "Ventana de contexto del LLM (num_ctx)",
        [
            ("Pequeña  (2 048 tokens) — recomendado para modelos ≤3B", 2048),
            ("Media    (4 096 tokens)", 4096),
            ("Grande   (8 192 tokens)", 8192),
            ("Muy grande (16 384 tokens) — solo modelos con capacidad suficiente", 16384),
        ],
        default=2,
    )

    num_predict = pick(
        "Longitud de respuesta (num_predict)",
        [
            ("Corta          (256 tokens)", 256),
            ("Predeterminada (1 000 tokens)", 1000),
            ("Larga          (3 000 tokens)", 3000),
        ],
        default=2,
    )

    top_k = pick(
        "Documentos a recuperar (retriever top-k)",
        [
            ("Pocos    (3)", 3),
            ("Predeterminado (5)", 5),
            ("Muchos   (10)", 10),
        ],
        default=2,
    )

    split_length, split_overlap = pick(
        "Tamaño de chunks",
        [
            ("Chicos        (100 palabras, overlap 10)", (100, 10)),
            ("Predeterminado (200 palabras, overlap 20)", (200, 20)),
            ("Grandes       (400 palabras, overlap 40)", (400, 40)),
        ],
        default=2,
    )

    return Config(
        llm_model=llm_model,
        num_ctx=num_ctx,
        num_predict=num_predict,
        top_k=top_k,
        split_length=split_length,
        split_overlap=split_overlap,
    )


# --- Pipeline ---

class StreamState:
    def __init__(self):
        self.started = False
        self._chunks: list[str] = []

    def reset(self):
        self.started = False
        self._chunks = []

    @property
    def full_response(self) -> str:
        return "".join(self._chunks)

    def callback(self, chunk):
        if not chunk.content:
            return
        self._chunks.append(chunk.content)
        if not self.started:
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


def print_setup_summary(cfg: Config, docs: list[Document], new_count: int) -> None:
    console.rule("[bold cyan]Configuración[/bold cyan]")
    console.print(f"  [cyan]LLM[/cyan]                 {cfg.llm_model} (Ollama)")
    console.print(f"  [cyan]num_ctx[/cyan]              {cfg.num_ctx} tokens")
    console.print(f"  [cyan]num_predict[/cyan]          {cfg.num_predict} tokens")
    console.print(f"  [cyan]temperature[/cyan]          {TEMPERATURE}")
    console.print(f"  [cyan]Retriever top-k[/cyan]      {cfg.top_k}")
    console.print(f"  [cyan]Split length[/cyan]         {cfg.split_length} palabras")
    console.print(f"  [cyan]Split overlap[/cyan]        {cfg.split_overlap} palabras")
    console.print(f"  [cyan]Embedding model[/cyan]      {EMBEDDING_MODEL} (Ollama)")
    console.print(f"  [cyan]Embedding dimension[/cyan]  {EMBEDDING_DIMENSION}")
    console.print(f"  [cyan]Ollama URL[/cyan]           {OLLAMA_BASE_URL}")
    console.print(f"  [cyan]DB table[/cyan]             {DB_TABLE}")
    console.print(f"  [cyan]Input dir[/cyan]            {INPUT_DIR}")

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
        console.print("  [yellow]No se encontraron documentos en 'rawdata/markdown/'[/yellow]")


def index_new_documents(store: PgvectorDocumentStore, cfg: Config) -> tuple[list[Document], int]:
    if not INPUT_DIR.exists():
        console.print(f"[red]Directorio no encontrado:[/red] {INPUT_DIR}")
        console.print("Creá la carpeta [bold]rawdata/markdown/[/bold] junto al script y agregá tus documentos.")
        return [], 0

    docs = load_local_documents(INPUT_DIR)

    splitter = DocumentSplitter(
        split_by="word",
        split_length=cfg.split_length,
        split_overlap=cfg.split_overlap,
    )
    docs = splitter.run(documents=docs)["documents"]

    existing_keys = {
        (doc.meta.get("file_path"), doc.meta.get("_split_id"))
        for doc in store.filter_documents()
        if doc.meta.get("file_path") is not None
    }

    new_docs = [
        doc for doc in docs
        if (doc.meta.get("file_path"), doc.meta.get("_split_id")) not in existing_keys
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


def build_rag_pipeline(store: PgvectorDocumentStore, cfg: Config) -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component("text_embedder", OllamaTextEmbedder(model=EMBEDDING_MODEL, url=OLLAMA_BASE_URL))
    pipeline.add_component("embedding_retriever", PgvectorEmbeddingRetriever(document_store=store, top_k=cfg.top_k))
    pipeline.add_component("keyword_retriever", PgvectorKeywordRetriever(document_store=store, top_k=cfg.top_k))
    pipeline.add_component("document_joiner", DocumentJoiner())
    pipeline.add_component("prompt_builder", PromptBuilder(template=PROMPT_TEMPLATE))
    pipeline.add_component("llm", OllamaGenerator(
        model=cfg.llm_model,
        url=OLLAMA_BASE_URL,
        generation_kwargs={
            "num_predict": cfg.num_predict,
            "temperature": TEMPERATURE,
            "num_ctx":     cfg.num_ctx,
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
        result = pipeline.run(
            {
                "text_embedder": {"text": question},
                "keyword_retriever": {"query": question},
                "prompt_builder": {"question": question},
            },
            include_outputs_from={"prompt_builder"},
        )

        if not stream_state.started:
            console.print("\n[bold green]Respuesta:[/bold green] [dim]Sin respuesta.[/dim]")
        console.print("\n")

        prompt = result.get("prompt_builder", {}).get("prompt", "")
        log("=" * 60)
        log(f"PREGUNTA: {question}")
        log("-" * 40 + " PROMPT")
        log(prompt)
        log("-" * 40 + " RESPUESTA")
        log(stream_state.full_response)
        log("=" * 60 + "\n")


def unload_models(cfg: Config) -> None:
    for model, label in [(cfg.llm_model, "LLM"), (EMBEDDING_MODEL, "Embedding")]:
        try:
            ollama.generate(model=model, prompt="", keep_alive=0)
            console.print(f"  [dim]{label} ({model}) descargado[/dim]")
        except Exception:
            pass


if __name__ == "__main__":
    cfg = run_setup()
    store = get_document_store()
    docs, new_count = index_new_documents(store, cfg)
    print_setup_summary(cfg, docs, new_count)
    pipeline = build_rag_pipeline(store, cfg)
    try:
        run_interactive_loop(pipeline)
    finally:
        console.rule("[bold cyan]Descargando modelos[/bold cyan]")
        unload_models(cfg)
