"""
Script standalone para la indexación del RAG de Ciberseguridad.
Se encarga de:
1. Parsear PDFs (marker-pdf), XMLs (CWE) y JSONs (CVE).
2. Fragmentar documentos largos (Splitter).
3. Deduplicar contra la base de datos (pgvector).
4. Embeber los documentos nuevos (Ollama).
5. Guardarlos en PostgreSQL.
"""

import logging
import os
from pathlib import Path

from haystack import Document
from haystack.utils import Secret
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.converters import (
    DOCXToDocument, MarkdownToDocument, TextFileToDocument
)
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack.document_stores.types import DuplicatePolicy

# Importamos los conversores que separamos en el otro archivo.
# Import robusto: funciona ejecutado como script (`python indexing/run_indexing.py`,
# donde sys.path incluye indexing/ y no hay paquete padre) y también importado como
# módulo (`from indexing.run_indexing import INPUT_DIR`), que es lo que hace
# src/pipeline/eval/run_chunking_experiment.py para reusar rutas y conversores.
try:
    from .converters import XMLCWEConverter, NVDJsonConverter
except ImportError:  # ejecutado como script: no hay paquete padre
    from converters import XMLCWEConverter, NVDJsonConverter

# --- Configuración ---
DB_CONNECTION       = "postgresql://avdbuser:avdbpass@vdb:5432/pgvdb"
OLLAMA_URL          = "http://ollama:11434"
DB_TABLE            = "ciberseguridad_docs"
EMBEDDING_DIMENSION = 1024
INPUT_DIR           = Path("/app/pipelines/rawdata")
CONVERTED_DIR       = INPUT_DIR / "_converted_md"

# Variables de chunking y embedding (antes vivían en los Valves del pipeline)
SPLIT_LENGTH        = 200
SPLIT_OVERLAP       = 20
EMBEDDING_MODEL     = "bge-m3"
EMBED_BATCH_SIZE    = 32

# --- Logger ---
logger = logging.getLogger("IndexingRAG")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class Indexer:
    def __init__(self):
        self.store = PgvectorDocumentStore(
            connection_string=Secret.from_token(DB_CONNECTION),
            embedding_dimension=EMBEDDING_DIMENSION,
            table_name=DB_TABLE,
            keyword_index_name="ciberseguridad_keyword_index",
        )
        self._marker_converter = None
        self._marker_load_failed = False

    def _get_marker_converter(self):
        """Carga marker-pdf on-demand para ahorrar memoria si no hay PDFs nuevos."""
        if self._marker_converter is not None:
            return self._marker_converter
        if self._marker_load_failed:
            return None
            
        logger.info("Cargando modelos de marker-pdf (lazy)...")
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            self._marker_converter = PdfConverter(artifact_dict=create_model_dict())
            logger.info("marker-pdf cargado correctamente.")
        except Exception as e:
            logger.error(f"No se pudo cargar marker-pdf: {e}")
            self._marker_load_failed = True
        return self._marker_converter

    def _release_marker_converter(self):
        """Libera los modelos de marker-pdf de la memoria RAM/VRAM."""
        if self._marker_converter:
            logger.info("Liberando modelos de marker-pdf de memoria...")
            self._marker_converter = None
            import gc
            gc.collect()

    def _convert_pdf_with_marker(self, pdf_path: Path) -> Path | None:
        CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
        md_path = CONVERTED_DIR / (pdf_path.stem + ".md")

        if md_path.exists():
            logger.info(f"Reutilizando conversión cacheada: {md_path.name}")
            return md_path

        converter = self._get_marker_converter()
        if not converter:
            logger.warning(f"marker-pdf no disponible, salteando: {pdf_path.name}")
            return None

        logger.info(f"Convirtiendo con marker-pdf: {pdf_path.name}")
        try:
            from marker.output import text_from_rendered
            rendered = converter(str(pdf_path))
            markdown_text, _, _ = text_from_rendered(rendered)
            md_path.write_text(markdown_text, encoding="utf-8")
            return md_path
        except Exception as e:
            logger.error(f"Error convirtiendo {pdf_path.name}: {e}")
            return None

    def load_documents(self) -> tuple[list[Document], list[Document]]:
        splittable: list[Document] = []
        atomic: list[Document] = []

        # 1. PDFs
        pdf_files = list(INPUT_DIR.glob("*.pdf"))
        if pdf_files:
            md_paths, source_names = [], []
            for pdf_path in pdf_files:
                md_path = self._convert_pdf_with_marker(pdf_path)
                if md_path:
                    md_paths.append(md_path)
                    source_names.append(pdf_path)

            if md_paths:
                result = MarkdownToDocument().run(sources=md_paths)
                for doc, original_pdf in zip(result["documents"], source_names):
                    doc.meta["file_path"] = str(original_pdf)
                    doc.meta["source"] = original_pdf.name
                splittable.extend(result["documents"])

        # 2. DOCX, MD nativo, TXT
        converters = [
            ("*.docx", DOCXToDocument()),
            ("*.md", MarkdownToDocument()),
            ("*.txt", TextFileToDocument()),
        ]
        for pattern, converter in converters:
            files = [f for f in INPUT_DIR.glob(pattern) if f.parent == INPUT_DIR]
            if files:
                splittable.extend(converter.run(sources=files)["documents"])

        # 3. XML (CWE) y JSON (CVE)
        xml_files = [f for f in INPUT_DIR.glob("*.xml") if f.parent == INPUT_DIR]
        if xml_files:
            atomic.extend(XMLCWEConverter().run(sources=xml_files)["documents"])

        cve_files = sorted(INPUT_DIR.rglob("cves_page_*.json"))
        if cve_files:
            atomic.extend(NVDJsonConverter().run(sources=cve_files)["documents"])

        return splittable, atomic

    def run(self):
        if not INPUT_DIR.exists():
            logger.error(f"Directorio no encontrado: {INPUT_DIR}")
            return

        logger.info("Iniciando escaneo de documentos...")
        splittable_docs, atomic_docs = self.load_documents()

        # Liberamos marker-pdf antes de usar Ollama para no saturar memoria
        self._release_marker_converter()

        if not splittable_docs and not atomic_docs:
            logger.warning("No se encontraron documentos en rawdata/")
            return

        # Chunking solo para los splittables
        if splittable_docs:
            splitter = DocumentSplitter(split_by="word", split_length=SPLIT_LENGTH, split_overlap=SPLIT_OVERLAP)
            splittable_docs = splitter.run(documents=splittable_docs)["documents"]

        # Deduplicación contra la BD
        stored = self.store.filter_documents()
        existing_keys = {
            (d.meta.get("file_path"), d.meta.get("split_id")) 
            for d in stored if d.meta.get("file_path")
        }
        
        new_splittable = [d for d in splittable_docs if (d.meta.get("file_path"), d.meta.get("split_id")) not in existing_keys]
        
        seen_ids = {d.id for d in stored}
        new_atomic = []
        for d in atomic_docs:
            if d.id not in seen_ids:
                seen_ids.add(d.id)
                new_atomic.append(d)

        new_docs = new_splittable + new_atomic
        if not new_docs:
            logger.info("No hay documentos nuevos. Base de datos actualizada.")
            return

        logger.info(f"Comenzando embedding de {len(new_docs)} documentos ({len(new_splittable)} chunks, {len(new_atomic)} atómicos)...")
        
        doc_embedder = OllamaDocumentEmbedder(
            model=EMBEDDING_MODEL,
            url=OLLAMA_URL,
            batch_size=EMBED_BATCH_SIZE,
        )

        embedded_docs = []
        failed_count = 0
        
        # Embebemos en lotes
        for i in range(0, len(new_docs), EMBED_BATCH_SIZE):
            batch = new_docs[i : i + EMBED_BATCH_SIZE]
            try:
                embedded_docs.extend(doc_embedder.run(batch)["documents"])
            except Exception as e:
                failed_count += len(batch)
                logger.error(f"Fallo en lote {i // EMBED_BATCH_SIZE}: {e}")

        if embedded_docs:
            self.store.write_documents(embedded_docs, policy=DuplicatePolicy.OVERWRITE)
            
        logger.info(f"Indexación finalizada. {len(embedded_docs)} agregados, {failed_count} fallidos.")
        logger.info(f"Total en store: {len(self.store.filter_documents())}")


if __name__ == "__main__":
    indexer = Indexer()
    indexer.run()