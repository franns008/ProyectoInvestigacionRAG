"""Registro de estrategias de chunking candidatas para el corpus splittable
(PDF/DOCX/MD/TXT). Los documentos atómicos (CWE, CVE) nunca pasan por acá.

Contexto y resultados empíricos: docs/data_splitting.md. Lo usa
src/pipeline/eval/run_chunking_experiment.py para indexar y comparar cada
estrategia en una tabla pgvector separada, sin tocar la tabla de producción.

Cada estrategia expone `needs_raw_markdown`: True si necesita el texto Markdown
crudo (sintaxis '#' intacta) en vez del texto ya pasado por MarkdownToDocument.
MarkdownToDocument RENDERIZA el markdown a texto plano y borra los headers ('#',
'##', ...) — MarkdownHeaderSplitter, que detecta headers por esa sintaxis, no
encuentra ninguno si se lo alimenta con la salida de MarkdownToDocument (queda
sin partir, ver docs/data_splitting.md §3.8). Verificado corriendo ambos
componentes contra el corpus real.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from haystack import Document
from haystack.components.preprocessors import (
    DocumentSplitter,
    MarkdownHeaderSplitter,
    RecursiveDocumentSplitter,
)


@dataclass
class ChunkingStrategy:
    name: str
    description: str
    build: Callable[[], object]        # -> componente Haystack, ya con .warm_up() si aplica
    needs_raw_markdown: bool = False


def _current_word200() -> DocumentSplitter:
    s = DocumentSplitter(split_by="word", split_length=200, split_overlap=20)
    s.warm_up()
    return s


def _sentence_es() -> DocumentSplitter:
    # split_by="word" + respect_sentence_boundary=True: el splitter NO corta a mitad
    # de oración (extiende el chunk hasta el punto siguiente) — a diferencia de
    # split_by="period", que sí puede combinarse con respect_sentence_boundary pero
    # Haystack lo ignora con un warning ("solo soportado para split_by='word'").
    s = DocumentSplitter(
        split_by="word",
        split_length=220,
        split_overlap=30,
        respect_sentence_boundary=True,
        language="es",
    )
    s.warm_up()
    return s


def _recursive_500() -> RecursiveDocumentSplitter:
    s = RecursiveDocumentSplitter(
        split_length=500,
        split_overlap=75,
        split_unit="word",
        separators=["\n\n", "\n", "sentence", " "],
    )
    s.warm_up()
    return s


def _markdown_header() -> MarkdownHeaderSplitter:
    # secondary_split="word" evita el gap de idioma de "period" (el splitter interno
    # de MarkdownHeaderSplitter no expone `language`, así que "period" tokenizaría
    # oraciones en inglés sobre texto en español).
    return MarkdownHeaderSplitter(
        header_split_levels=[1, 2, 3],
        secondary_split="word",
        split_length=400,
        split_overlap=40,
    )


STRATEGIES: dict[str, ChunkingStrategy] = {
    "current_word200": ChunkingStrategy(
        name="current_word200",
        description="Baseline en producción: DocumentSplitter word/200/overlap20, sin language.",
        build=_current_word200,
    ),
    "sentence_es": ChunkingStrategy(
        name="sentence_es",
        description="DocumentSplitter respetando límites de oración, language=es, ~220 palabras/overlap 30.",
        build=_sentence_es,
    ),
    "recursive_500": ChunkingStrategy(
        name="recursive_500",
        description="RecursiveDocumentSplitter: separadores párrafo→línea→oración→palabra, 500/75.",
        build=_recursive_500,
    ),
    "markdown_header": ChunkingStrategy(
        name="markdown_header",
        description=(
            "MarkdownHeaderSplitter h1-h3 + secondary split word/400/40. Preserva "
            "header/parent_headers en meta. Requiere texto markdown crudo."
        ),
        build=_markdown_header,
        needs_raw_markdown=True,
    ),
}


def prepend_header_context(doc: Document) -> Document:
    """Devuelve una copia del doc con el breadcrumb de headers antepuesto al
    contenido (Mejora 4 de docs/data_splitting.md: el chunk deja de depender de que
    el reader ya sepa en qué sección está). `Document` es un dataclass inmutable
    para Haystack, así que se usa `dataclasses.replace` en vez de mutar `content`.

    Devuelve el mismo doc sin cambios si no tiene meta 'header' (p. ej. no vino de
    MarkdownHeaderSplitter).
    """
    header = (doc.meta.get("header") or "").strip("* \n")
    parents = [str(h).strip("* \n") for h in (doc.meta.get("parent_headers") or []) if h]
    breadcrumb = " > ".join(parents + [header]) if (header or parents) else ""
    if not breadcrumb:
        return doc
    return replace(doc, content=f"[{breadcrumb}]\n{doc.content}")
