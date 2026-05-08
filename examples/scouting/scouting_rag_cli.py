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
from haystack.components.generators.openai import OpenAIGenerator
import ollama

from rich.console import Console

# --- Configuración fija ---
DB_CONNECTION       = "postgresql://avdbuser:avdbpass@localhost:5433/pgvdb"
DB_TABLE            = "local_docs"
EMBEDDING_MODEL     = "bge-m3"
EMBEDDING_DIMENSION = 1024
OLLAMA_BASE_URL     = "http://localhost:11434"
GROQ_BASE_URL       = "https://api.groq.com/openai/v1"
INPUT_DIR           = Path(__file__).parent / "rawdata/markdown"
LOG_FILE            = Path(__file__).parent / "log.txt"

GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]

PROMPT_TEMPLATE = """
Given the following information, answer the question.
YOU SHOULD ANSWER IN THE LANGUAGE OF THE QUESTION, NOT THE LANGUAGE OF THE DOCUMENTS.
if you dont know about the question, say you dont know. Do not try to make up an answer.

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
    max_tokens:   int
    top_k:        int
    split_length: int
    split_overlap: int


# --- Setup helpers ---

def pick(title: str, options: list[tuple[str, Any]], default: int) -> Any:
    """Muestra una lista numerada y devuelve el valor elegido. Enter usa el default."""
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


def run_setup() -> Config:
    if not os.environ.get("GROQ_API_KEY"):
        console.print("[red]GROQ_API_KEY no encontrada. Creá un archivo .env en el directorio con esa variable.[/red]")
        raise SystemExit(1)

    llm_model = pick(
        "Seleccioná un modelo Groq",
        [(m, m) for m in GROQ_MODELS],
        default=1,
    )

    max_tokens = pick(
        "Longitud de respuesta (max_tokens)",
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
        max_tokens=max_tokens,
        top_k=top_k,
        split_length=split_length,
        split_overlap=split_overlap,
    )


# --- Pipeline ---

class StreamState:
    """Coordina el streaming del LLM en el loop interactivo."""
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
        # "simple" evita stemming y stopwords específicos de idioma;
        # correcto para docs en español + nombres propios + estadísticas.
        # Si el índice GIN ya existe con 'english', dropearlo y recrearlo:
        #   DROP INDEX haystack_keyword_index;
        # El store lo recrea automáticamente al conectar.
        language="simple",
    )


def print_setup_summary(cfg: Config, docs: list[Document], new_count: int) -> None:
    console.rule("[bold cyan]Configuración[/bold cyan]")
    console.print(f"  [cyan]LLM[/cyan]                 {cfg.llm_model} (Groq)")
    console.print(f"  [cyan]max_tokens[/cyan]           {cfg.max_tokens}")
    console.print(f"  [cyan]temperature[/cyan]          {TEMPERATURE}")
    console.print(f"  [cyan]Retriever top-k[/cyan]      {cfg.top_k}")
    console.print(f"  [cyan]Split length[/cyan]         {cfg.split_length} palabras")
    console.print(f"  [cyan]Split overlap[/cyan]        {cfg.split_overlap} palabras")
    console.print(f"  [cyan]Embedding model[/cyan]      {EMBEDDING_MODEL} (Ollama)")
    console.print(f"  [cyan]Embedding dimension[/cyan]  {EMBEDDING_DIMENSION}")
    console.print(f"  [cyan]Ollama URL[/cyan]           {OLLAMA_BASE_URL}")
    console.print(f"  [cyan]Groq base URL[/cyan]        {GROQ_BASE_URL}")
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
        console.print("  [yellow]No se encontraron documentos en 'input/'[/yellow]")


def index_new_documents(store: PgvectorDocumentStore, cfg: Config) -> tuple[list[Document], int]:
    if not INPUT_DIR.exists():
        console.print(f"[red]Directorio no encontrado:[/red] {INPUT_DIR}")
        console.print("Creá la carpeta [bold]input/[/bold] junto al script y agregá tus documentos (pdf, docx, md, txt).")
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


def build_retrieval_pipeline(store: PgvectorDocumentStore, cfg: Config) -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component("text_embedder", OllamaTextEmbedder(model=EMBEDDING_MODEL, url=OLLAMA_BASE_URL))
    pipeline.add_component("embedding_retriever", PgvectorEmbeddingRetriever(document_store=store, top_k=cfg.top_k))
    pipeline.add_component("keyword_retriever", PgvectorKeywordRetriever(document_store=store, top_k=cfg.top_k))
    pipeline.add_component("document_joiner", DocumentJoiner())

    pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
    pipeline.connect("embedding_retriever", "document_joiner")
    pipeline.connect("keyword_retriever", "document_joiner")

    return pipeline


def build_generation_pipeline(cfg: Config) -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component("prompt_builder", PromptBuilder(template=PROMPT_TEMPLATE))
    pipeline.add_component("llm", OpenAIGenerator(
        api_key=Secret.from_env_var("GROQ_API_KEY"),
        api_base_url=GROQ_BASE_URL,
        model=cfg.llm_model,
        generation_kwargs={
            "temperature": TEMPERATURE,
            "max_tokens":  cfg.max_tokens,
        },
        streaming_callback=stream_state.callback,
    ))

    pipeline.connect("prompt_builder", "llm")

    return pipeline


def estimate_tokens(text: str) -> int:
    """Estimación rápida: ~4 chars por token (válido para español e inglés)."""
    return max(1, len(text) // 4)


def log_retrieval(
    semantic_docs: list[Document],
    keyword_docs: list[Document],
    final_docs: list[Document],
) -> None:
    semantic_ids = {d.id for d in semantic_docs}
    keyword_ids  = {d.id for d in keyword_docs}

    log(f"  Semántico : {len(semantic_docs)} docs")
    log(f"  Keyword   : {len(keyword_docs)} docs")
    log(f"  Final (joiner, dedup): {len(final_docs)} docs")
    log("")

    only_sem = sum(1 for d in final_docs if d.id in semantic_ids and d.id not in keyword_ids)
    only_kw  = sum(1 for d in final_docs if d.id in keyword_ids  and d.id not in semantic_ids)
    both     = sum(1 for d in final_docs if d.id in semantic_ids and d.id in keyword_ids)
    log(f"  Origen: {only_sem} solo-semántico | {only_kw} solo-keyword | {both} en ambos")
    log("")

    file_counts: dict[str, int] = {}
    for d in final_docs:
        fname = Path(d.meta.get("file_path", "desconocido")).name
        file_counts[fname] = file_counts.get(fname, 0) + 1
    log(f"  Archivos fuente ({len(file_counts)} únicos):")
    for fname, count in sorted(file_counts.items()):
        log(f"    {fname}: {count} chunk(s)")
    log("")

    total_tokens = sum(estimate_tokens(d.content or "") for d in final_docs)
    log(f"  Tokens estimados en contexto: ~{total_tokens}")
    log("")

    log("  Detalle de chunks:")
    for i, d in enumerate(final_docs, 1):
        fname    = Path(d.meta.get("file_path", "?")).name
        split_id = d.meta.get("_split_id", "?")
        score    = f"{d.score:.4f}" if d.score is not None else "N/A"
        overlap  = d.meta.get("_split_overlap_unit_count", "?")
        tokens   = estimate_tokens(d.content or "")

        origins = []
        if d.id in semantic_ids:
            origins.append("sem")
        if d.id in keyword_ids:
            origins.append("kw")
        origin_tag = "+".join(origins)

        log(f"    [{i:02d}] {fname}  split={split_id}  overlap={overlap}  score={score}  ~{tokens}tok  [{origin_tag}]")


def print_retrieval_summary(
    semantic_docs: list[Document],
    keyword_docs: list[Document],
    final_docs: list[Document],
) -> None:
    semantic_ids = {d.id for d in semantic_docs}
    keyword_ids  = {d.id for d in keyword_docs}

    only_sem = sum(1 for d in final_docs if d.id in semantic_ids and d.id not in keyword_ids)
    only_kw  = sum(1 for d in final_docs if d.id in keyword_ids  and d.id not in semantic_ids)
    both     = sum(1 for d in final_docs if d.id in semantic_ids and d.id in keyword_ids)

    file_counts: dict[str, int] = {}
    for d in final_docs:
        fname = Path(d.meta.get("file_path", "?")).name
        file_counts[fname] = file_counts.get(fname, 0) + 1

    files_str    = "  ".join(f"[cyan]{n}[/cyan]({c})" for n, c in sorted(file_counts.items()))
    total_tokens = sum(estimate_tokens(d.content or "") for d in final_docs)

    console.rule("[dim]Retrieval[/dim]")
    console.print(
        f"  [dim]{len(final_docs)} chunks[/dim]"
        f"  [dim]·[/dim]"
        f"  [green]{only_sem} sem[/green]"
        f"  [yellow]{only_kw} kw[/yellow]"
        f"  [blue]{both} ambos[/blue]"
        f"  [dim]·[/dim]  {files_str}"
        f"  [dim]·[/dim]  [dim]~{total_tokens} tok[/dim]"
    )

    for i, d in enumerate(final_docs, 1):
        fname    = Path(d.meta.get("file_path", "?")).name
        split_id = d.meta.get("_split_id", "?")
        score    = f"{d.score:.3f}" if d.score is not None else "N/A"
        tokens   = estimate_tokens(d.content or "")
        origins  = []
        if d.id in semantic_ids:
            origins.append("[green]sem[/green]")
        if d.id in keyword_ids:
            origins.append("[yellow]kw[/yellow]")
        console.print(
            f"  [dim]{i:02d}[/dim]  {fname}  "
            f"[dim]split={split_id}  score={score}  ~{tokens}tok[/dim]  "
            + "+".join(origins)
        )
    console.rule()


def log_session_start(cfg: Config) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log("")
    log("╔" + "═" * 58 + "╗")
    log(f"║  NUEVA SESIÓN  —  {ts:<38}║")
    log("╠" + "═" * 58 + "╣")
    log(f"║  LLM            {cfg.llm_model:<41}║")
    log(f"║  max_tokens     {cfg.max_tokens:<41}║")
    log(f"║  top_k          {cfg.top_k:<41}║")
    log(f"║  split_length   {cfg.split_length:<41}║")
    log(f"║  split_overlap  {cfg.split_overlap:<41}║")
    log(f"║  embedding      {EMBEDDING_MODEL:<41}║")
    log("╚" + "═" * 58 + "╝")
    log("")


def run_interactive_loop(
    retrieval_pipeline: Pipeline,
    generation_pipeline: Pipeline,
) -> None:
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

        ret_result = retrieval_pipeline.run(
            {
                "text_embedder": {"text": question},
                "keyword_retriever": {"query": question},
            },
            include_outputs_from={"embedding_retriever", "keyword_retriever"},
        )

        semantic_docs = ret_result.get("embedding_retriever", {}).get("documents", [])
        keyword_docs  = ret_result.get("keyword_retriever",  {}).get("documents", [])
        final_docs    = ret_result.get("document_joiner",    {}).get("documents", [])

        print_retrieval_summary(semantic_docs, keyword_docs, final_docs)

        stream_state.reset()
        gen_result = generation_pipeline.run(
            {
                "prompt_builder": {"documents": final_docs, "question": question},
            },
            include_outputs_from={"prompt_builder"},
        )

        if not stream_state.started:
            console.print("\n[bold green]Respuesta:[/bold green] [dim]Sin respuesta.[/dim]")
        console.print("\n")

        prompt = gen_result.get("prompt_builder", {}).get("prompt", "")

        log("=" * 60)
        log(f"PREGUNTA: {question}")
        log("-" * 40 + " RETRIEVAL")
        log_retrieval(semantic_docs, keyword_docs, final_docs)
        log("-" * 40 + " PROMPT")
        log(prompt)
        log("-" * 40 + " RESPUESTA")
        log(stream_state.full_response)
        log("=" * 60 + "\n")


def unload_embedding_model() -> None:
    try:
        ollama.generate(model=EMBEDDING_MODEL, prompt="", keep_alive=0)
        console.print(f"  [dim]Embedding ({EMBEDDING_MODEL}) descargado[/dim]")
    except Exception:
        pass


if __name__ == "__main__":
    cfg = run_setup()
    log_session_start(cfg)
    store = get_document_store()
    docs, new_count = index_new_documents(store, cfg)
    print_setup_summary(cfg, docs, new_count)
    retrieval_pipeline  = build_retrieval_pipeline(store, cfg)
    generation_pipeline = build_generation_pipeline(cfg)
    try:
        run_interactive_loop(retrieval_pipeline, generation_pipeline)
    finally:
        console.rule("[bold cyan]Descargando modelos[/bold cyan]")
        unload_embedding_model()
