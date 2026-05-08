"""
Herramienta de inspección de estrategias de chunking para RAG.

Uso (dentro del contenedor):
    python /app/pipelines/inspect_chunks.py [--splitter TIPO] [--split_length N] [--split_overlap N]

Splitters disponibles:
    Haystack   : recursive, word, sentence, passage
    LlamaIndex : llama_sentence, llama_semantic, llama_semantic_double
    LangChain  : langchain_recursive

El output se guarda en /app/pipelines/chunks_<splitter>_len<N>_overlap<N>.txt
Para copiarlo a tu máquina:
    docker cp infrastructure-pipelines-1:/app/pipelines/<archivo>.txt .
"""

import argparse
from pathlib import Path

from haystack import Document
from haystack.components.preprocessors import RecursiveDocumentSplitter, DocumentSplitter, DocumentCleaner
from haystack.components.converters import PyPDFToDocument, DOCXToDocument, MarkdownToDocument, TextFileToDocument

INPUT_DIR = Path("/app/pipelines/rawdata")
OLLAMA_URL = "http://ollama:11434"
EMBED_MODEL = "bge-m3"

# ---------------------------------------------------------------------------

def load_documents() -> list[Document]:
    docs = []
    for pattern, converter in [
        ("*.pdf",  PyPDFToDocument()),
        ("*.docx", DOCXToDocument()),
        ("*.md",   MarkdownToDocument()),
        ("*.txt",  TextFileToDocument()),
    ]:
        files = list(INPUT_DIR.glob(pattern))
        if files:
            docs.extend(converter.run(sources=files)["documents"])
    cleaner = DocumentCleaner(remove_extra_whitespaces=True)
    return cleaner.run(documents=docs)["documents"]


def build_splitter(name: str, split_length: int, split_overlap: int):
    if name == "recursive":
        return "haystack", RecursiveDocumentSplitter(
            split_length=split_length,
            split_overlap=split_overlap,
            split_unit="word",
            separators=["\n\n", "sentence", "\n", " "],
        )
    if name in ("word", "sentence", "passage"):
        return "haystack", DocumentSplitter(
            split_by=name,
            split_length=split_length,
            split_overlap=split_overlap,
        )
    if name == "llama_sentence":
        from llama_index.core.node_parser import SentenceSplitter
        return "llama", SentenceSplitter(chunk_size=split_length, chunk_overlap=split_overlap)
    if name == "llama_semantic":
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        from llama_index.embeddings.ollama import OllamaEmbedding
        embed = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_URL)
        return "llama", SemanticSplitterNodeParser.from_defaults(embed_model=embed)
    if name == "llama_semantic_double":
        from llama_index.core.node_parser import SemanticDoubleMergingSplitterNodeParser
        from llama_index.embeddings.ollama import OllamaEmbedding
        embed = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_URL)
        return "llama", SemanticDoubleMergingSplitterNodeParser.from_defaults(embed_model=embed)
    if name == "langchain_recursive":
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        # split_length en palabras → multiplicamos x6 para aproximar caracteres
        return "langchain", RecursiveCharacterTextSplitter(
            chunk_size=split_length * 6,
            chunk_overlap=split_overlap * 6,
        )
    raise ValueError(f"Splitter desconocido: {name}")


def run_splitter(kind: str, splitter, docs: list[Document]) -> list[tuple[str, str]]:
    """Devuelve lista de (file_path, contenido_chunk)."""
    if kind == "haystack":
        chunks = splitter.run(documents=docs)["documents"]
        return [(c.meta.get("file_path", "unknown"), c.content or "") for c in chunks]

    if kind == "llama":
        from llama_index.core import Document as LlamaDoc
        llama_docs = [LlamaDoc(text=d.content, metadata=d.meta) for d in docs]
        try:
            nodes = splitter.get_nodes_from_documents(llama_docs, show_progress=False)
        except Exception as e:
            raise RuntimeError(
                f"Error al splitear con LlamaIndex: {e}\n"
                "Tip: llama_semantic_double puede fallar con bge-m3 por NaN en embeddings de frases cortas."
            ) from e
        return [(n.metadata.get("file_path", "unknown"), n.text) for n in nodes]

    if kind == "langchain":
        result = []
        for doc in docs:
            for chunk in splitter.split_text(doc.content or ""):
                result.append((doc.meta.get("file_path", "unknown"), chunk))
        return result


def write_output(chunks: list[tuple[str, str]], output_path: str, args):
    DIV1 = "=" * 60
    DIV2 = "-" * 60

    by_file: dict[str, list[str]] = {}
    for fpath, content in chunks:
        fname = Path(fpath).name
        by_file.setdefault(fname, []).append(content)

    with open(output_path, "w") as f:
        f.write(f"splitter={args.splitter} | split_length={args.split_length} | split_overlap={args.split_overlap}\n\n")
        for fname, cs in by_file.items():
            wc = [len(c.split()) for c in cs]
            f.write(f"{DIV1}\n")
            f.write(f"ARCHIVO: {fname}  ({len(cs)} chunks | min={min(wc)} avg={sum(wc)//len(wc)} max={max(wc)})\n")
            f.write(f"{DIV1}\n")
            for i, content in enumerate(cs, 1):
                words = len(content.split())
                f.write(f"\n[CHUNK {i}/{len(cs)} - {words} palabras]\n")
                f.write(f"{DIV2}\n")
                f.write(content + "\n")
                f.write(f"{DIV2}\n")
        f.write(f"\nTOTAL: {len(chunks)} chunks\n")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splitter", default="recursive",
        choices=["recursive", "word", "sentence", "passage",
                 "llama_sentence", "llama_semantic", "llama_semantic_double",
                 "langchain_recursive"])
    parser.add_argument("--split_length", type=int, default=200)
    parser.add_argument("--split_overlap", type=int, default=0)
    args = parser.parse_args()

    print(f"Cargando documentos desde {INPUT_DIR}...")
    docs = load_documents()
    print(f"  {len(docs)} documentos cargados.")

    print(f"Construyendo splitter '{args.splitter}'...")
    kind, splitter = build_splitter(args.splitter, args.split_length, args.split_overlap)

    if args.splitter in ("llama_semantic", "llama_semantic_double"):
        print("Spliteando... (generando embeddings con Ollama, puede tardar unos segundos)")
    else:
        print("Spliteando...")
    chunks = run_splitter(kind, splitter, docs)

    output = f"/app/pipelines/splitter_{args.splitter}.txt"
    write_output(chunks, output, args)
    print(f"Guardado en: {output}  ({len(chunks)} chunks totales)")

