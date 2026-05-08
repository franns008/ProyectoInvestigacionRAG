"""
Herramienta de inspección de estrategias de chunking para RAG.

Uso (dentro del contenedor):
    python /app/pipelines/inspect_chunks.py [--splitter TIPO] [--split_length N] [--split_overlap N] [--threshold N]

Splitters disponibles:
    Haystack   : recursive, word, sentence, passage
    LlamaIndex : llama_sentence, llama_semantic, llama_hierarchical
    LangChain  : langchain_recursive
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


def build_splitter(name: str, split_length: int, split_overlap: int, threshold: int):
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
        return "llama", SemanticSplitterNodeParser.from_defaults(
            embed_model=embed,
            breakpoint_percentile_threshold=threshold,
        )
    if name == "llama_hierarchical":
        from llama_index.core.node_parser import HierarchicalNodeParser
        return "llama_hierarchical", HierarchicalNodeParser.from_defaults(
            chunk_sizes=[split_length * 16, split_length * 4, split_length]
        )
    if name == "langchain_recursive":
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return "langchain", RecursiveCharacterTextSplitter(
            chunk_size=split_length * 6,
            chunk_overlap=split_overlap * 6,
        )
    raise ValueError(f"Splitter desconocido: {name}")


def run_splitter(kind: str, splitter, docs: list[Document]) -> list:
    if kind == "haystack":
        chunks = splitter.run(documents=docs)["documents"]
        return [(c.meta.get("file_path", "unknown"), c.content or "") for c in chunks]

    if kind == "llama":
        from llama_index.core import Document as LlamaDoc
        llama_docs = [LlamaDoc(text=d.content, metadata=d.meta) for d in docs]
        try:
            nodes = splitter.get_nodes_from_documents(llama_docs, show_progress=False)
        except Exception as e:
            raise RuntimeError(f"Error al splitear con LlamaIndex: {e}") from e
        return [(n.metadata.get("file_path", "unknown"), n.text) for n in nodes]

    if kind == "llama_hierarchical":
        from llama_index.core import Document as LlamaDoc
        from llama_index.core.node_parser import get_leaf_nodes
        llama_docs = [LlamaDoc(text=d.content, metadata=d.meta) for d in docs]
        all_nodes = splitter.get_nodes_from_documents(llama_docs, show_progress=False)
        result = []
        for node in all_nodes:
            level = 0 if node.parent_node is None else (1 if node.parent_node and any(
                n.node_id == node.parent_node.node_id and n.parent_node is None for n in all_nodes
            ) else 2)
            result.append((node.metadata.get("file_path", "unknown"), node.text, level, node.node_id, node.parent_node.node_id if node.parent_node else None))
        return result

    if kind == "langchain":
        result = []
        for doc in docs:
            for chunk in splitter.split_text(doc.content or ""):
                result.append((doc.meta.get("file_path", "unknown"), chunk))
        return result


def write_hierarchical_output(nodes: list, output_path: str, args):
    DIV1 = "=" * 60
    DIV2 = "-" * 60

    by_file: dict[str, dict[int, list]] = {}
    for fpath, content, level, node_id, parent_id in nodes:
        fname = Path(fpath).name
        by_file.setdefault(fname, {}).setdefault(level, []).append((content, node_id, parent_id))

    chunk_sizes = [args.split_length * 16, args.split_length * 4, args.split_length]
    total = len(nodes)

    with open(output_path, "w") as f:
        f.write(f"splitter=llama_hierarchical | chunk_sizes={chunk_sizes}\n")
        f.write(f"Niveles: 0=raíz ({chunk_sizes[0]} tokens), 1=medio ({chunk_sizes[1]}), 2=hoja ({chunk_sizes[2]})\n\n")
        for fname, levels in by_file.items():
            f.write(f"{DIV1}\nARCHIVO: {fname}\n{DIV1}\n")
            for level in sorted(levels):
                label = ["RAÍZ", "MEDIO", "HOJA"][level]
                chunks_at_level = levels[level]
                wc = [len(c.split()) for c, _, _ in chunks_at_level]
                f.write(f"\n  NIVEL {level} - {label} ({len(chunks_at_level)} chunks | min={min(wc)} avg={sum(wc)//len(wc)} max={max(wc)})\n")
                f.write(f"  {DIV2}\n")
                for i, (content, node_id, parent_id) in enumerate(chunks_at_level, 1):
                    words = len(content.split())
                    parent_str = f"  parent={parent_id[:8]}..." if parent_id else "  (sin padre)"
                    f.write(f"\n  [L{level} CHUNK {i}/{len(chunks_at_level)} - {words} palabras |{parent_str}]\n")
                    f.write(f"  {DIV2}\n")
                    for line in content.splitlines():
                        f.write(f"  {line}\n")
                    f.write(f"  {DIV2}\n")
        f.write(f"\nTOTAL: {total} nodos\n")


def write_output(chunks: list[tuple[str, str]], output_path: str, args):
    DIV1 = "=" * 60
    DIV2 = "-" * 60

    by_file: dict[str, list[str]] = {}
    for fpath, content in chunks:
        fname = Path(fpath).name
        by_file.setdefault(fname, []).append(content)

    if args.splitter == "llama_semantic":
        header = f"splitter={args.splitter} | threshold={args.threshold} (corte por similitud semántica)"
    else:
        header = f"splitter={args.splitter} | split_length={args.split_length} | split_overlap={args.split_overlap}"

    with open(output_path, "w") as f:
        f.write(header + "\n\n")
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
                 "llama_sentence", "llama_semantic",
                 "llama_hierarchical", "langchain_recursive"])
    parser.add_argument("--split_length", type=int, default=200)
    parser.add_argument("--split_overlap", type=int, default=0)
    parser.add_argument("--threshold", type=int, default=95,
        help="Percentil de corte para llama_semantic (1-99). Menor = más chunks. Default: 95")
    args = parser.parse_args()

    print(f"Cargando documentos desde {INPUT_DIR}...")
    docs = load_documents()
    print(f"  {len(docs)} documentos cargados.")

    print(f"Construyendo splitter '{args.splitter}'...")
    kind, splitter = build_splitter(args.splitter, args.split_length, args.split_overlap, args.threshold)

    if args.splitter == "llama_semantic":
        print(f"Spliteando... (threshold={args.threshold}, generando embeddings con Ollama)")
    else:
        print("Spliteando...")
    chunks = run_splitter(kind, splitter, docs)

    output = f"/app/pipelines/splitter_{args.splitter}.txt"
    if kind == "llama_hierarchical":
        write_hierarchical_output(chunks, output, args)
    else:
        write_output(chunks, output, args)
    total = len(chunks)
    print(f"Guardado en: {output}  ({total} {'nodos' if kind == 'llama_hierarchical' else 'chunks'} totales)")
