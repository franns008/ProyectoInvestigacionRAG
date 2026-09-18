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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path

import psycopg

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
EMBEDDING_DIMENSION = 2560  # nativa de qwen3-embedding:4b (sin truncar vía MRL)
INPUT_DIR           = Path("/app/pipelines/rawdata")
CONVERTED_DIR       = INPUT_DIR / "_converted_md"

# Variables de chunking y embedding (antes vivían en los Valves del pipeline)
SPLIT_LENGTH        = 200
SPLIT_OVERLAP       = 20
EMBEDDING_MODEL     = "qwen3-embedding:4b"

# Ajustes de embedding que dependen del hardware. Los defaults son los de siempre (GPU con
# VRAM de sobra); cada máquina los pisa por variable de entorno del container `pipelines`.
# El overlay AMD (infrastructure/docker-compose.amd.yml) fija los valores para placas de 4 GB.
def _env_int(name: str, default: int | None) -> int | None:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default

EMBED_BATCH_SIZE    = _env_int("EMBED_BATCH_SIZE", 64)
# Requests concurrentes a Ollama. Debe ser <= OLLAMA_NUM_PARALLEL o el request de más queda
# encolado del lado del servidor y se come el timeout sin haber empezado.
EMBED_CONCURRENCY   = _env_int("EMBED_CONCURRENCY", 2)
# Por request, no total. 120 s es el default de la librería: alcanza con GPU, no cuando el
# embedding cae a CPU (ahí un lote puede tardar minutos).
EMBED_TIMEOUT_SECONDS = _env_int("EMBED_TIMEOUT_SECONDS", 120)
# Contexto por request de embedding (`options.num_ctx`), no global. Sin definir = el de
# Ollama. Con 4 GB de VRAM, 512 es lo que hace entrar el modelo en GPU (33/37 capas contra
# 0/37 sin fijarlo) y trunca ~0,5% de los CVE (p99 = 396 tokens).
EMBED_NUM_CTX       = _env_int("EMBED_NUM_CTX", None)

# --- Logger ---
logger = logging.getLogger("IndexingRAG")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def _chunk_key(doc: Document) -> tuple[str, str] | None:
    """Clave de un chunk. `split_id` se normaliza a texto porque de la BD vuelve
    como texto y en memoria es un entero: sin esto, ningún chunk deduplicaría."""
    file_path = doc.meta.get("file_path")
    if not file_path:
        return None
    split_id = doc.meta.get("split_id")
    return (str(file_path), "" if split_id is None else str(split_id))


def _is_blank(doc: Document) -> bool:
    """Chunk sin una sola letra: sólo espacios, saltos de línea y tabs.

    marker-pdf rellena las celdas de sus tablas markdown hasta el ancho de la columna, así
    que un índice deja corridas de cientos de espacios seguidos (592 en el estudio de Cloud
    SCI). `split_by="word"` parte por el carácter " " literal, de modo que cada corrida
    aporta N unidades vacías que igual cuentan para `split_length`: una sola corrida llena
    tres chunks enteros sin una letra adentro. El `skip_empty_documents` de Haystack no los
    ve, porque mide `len(txt) > 0` y un chunk de 200 espacios mide 200.

    Se descarta sólo lo TOTALMENTE vacío, no lo corto: hay filas de tabla de 130-180
    caracteres con datos reales (las comparativas de la guía 5G) que un mínimo por longitud
    se llevaría puestas. Los chunks con escombros de tabla ("|", "| Requisitos") quedan para
    cuando se arregle el parsing, que es donde está la causa.
    """
    return not (doc.content or "").strip()


def _in_groups(items: list, size: int):
    it = iter(items)
    while group := list(islice(it, size)):
        yield group


class Indexer:
    def __init__(self, include_cve: bool = False):
        # include_cve=False por defecto: los ~8k CVE de NVD son atómicos y su embedding
        # local en CPU domina el tiempo de indexación (~20 min) sin aportar a las
        # preguntas de CWE/guías. Activar con --include-cve cuando se los quiera indexar.
        self.include_cve = include_cve
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

    def _stored_keys(self) -> tuple[set[str], set[tuple[str, str]]]:
        """Qué hay ya en la base, con un SELECT de sólo las columnas necesarias.

        No usa `store.filter_documents()` a propósito: eso trae los documentos completos
        *con su embedding*, y con cientos de miles de filas son varios GB de RAM nada más
        que para saber qué estaba. Acá se traen el id y dos campos de metadata.
        """
        # Fuerza la creación de la tabla si es una base nueva, antes de consultarla.
        self.store.count_documents()

        ids: set[str] = set()
        keys: set[tuple[str, str]] = set()
        with psycopg.connect(DB_CONNECTION) as conn, conn.cursor() as cursor:
            cursor.execute(
                f"SELECT id, meta->>'file_path', meta->>'split_id' FROM {DB_TABLE}"  # noqa: S608
            )
            for doc_id, file_path, split_id in cursor:
                ids.add(doc_id)
                if file_path:
                    keys.add((file_path, "" if split_id is None else split_id))
        logger.info(f"Ya en la base: {len(ids)} documentos (se saltean).")
        return ids, keys

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

        if self.include_cve:
            cve_files = sorted(INPUT_DIR.rglob("cves_page_*.json"))
            if cve_files:
                atomic.extend(NVDJsonConverter().run(sources=cve_files)["documents"])
        else:
            logger.info("CVE (NVD) OMITIDOS — usar --include-cve para indexarlos.")

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
            chunks = splitter.run(documents=splittable_docs)["documents"]

            # Los chunks en blanco tienen todos el MISMO contenido, o sea el mismo
            # embedding: no se recuperan sueltos sino en bloque, y con ranker_top_k=4 se
            # comen casi todos los lugares del prompt dejando al LLM el nombre del PDF sin
            # texto al lado. Ver _is_blank.
            splittable_docs = [d for d in chunks if not _is_blank(d)]
            blank = len(chunks) - len(splittable_docs)
            if blank:
                logger.info(
                    f"Descartados {blank} chunks en blanco de {len(chunks)} "
                    f"({blank / len(chunks) * 100:.1f}%)."
                )

        # Deduplicación contra la BD. Es también lo que hace la indexación retomable:
        # lo que ya se escribió en una corrida anterior no se vuelve a embeber.
        stored_ids, existing_keys = self._stored_keys()

        new_splittable = [d for d in splittable_docs if _chunk_key(d) not in existing_keys]

        seen_ids = set(stored_ids)
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
            timeout=EMBED_TIMEOUT_SECONDS,
            generation_kwargs={"num_ctx": EMBED_NUM_CTX} if EMBED_NUM_CTX else None,
        )

        batches = [new_docs[i:i + EMBED_BATCH_SIZE] for i in range(0, len(new_docs), EMBED_BATCH_SIZE)]
        logger.info(
            f"Embebiendo {len(batches)} lotes de hasta {EMBED_BATCH_SIZE} docs "
            f"({EMBED_CONCURRENCY} en paralelo, timeout={EMBED_TIMEOUT_SECONDS}s, "
            f"num_ctx={EMBED_NUM_CTX or 'default de Ollama'})..."
        )

        written = 0
        failed_count = 0
        started = time.perf_counter()

        with ThreadPoolExecutor(max_workers=EMBED_CONCURRENCY) as executor:
            # Ventana acotada en vez de encolar los N lotes de una: con cientos de miles
            # de documentos, tener todos los lotes vivos a la vez son varios GB de RAM.
            for done, group in enumerate(_in_groups(batches, EMBED_CONCURRENCY), start=1):
                futures = {executor.submit(doc_embedder.run, batch): batch for batch in group}
                for future in as_completed(futures):
                    try:
                        embedded = future.result()["documents"]
                    except Exception as e:
                        failed_count += len(futures[future])
                        logger.error(f"Fallo en lote: {e}")
                        continue
                    # Se escribe lote a lote, no todo al final: así una corrida larga se
                    # puede cortar y retomar sin perder lo hecho, y la memoria no crece
                    # con el tamaño del corpus.
                    self.store.write_documents(embedded, policy=DuplicatePolicy.OVERWRITE)
                    written += len(embedded)

                processed = done * EMBED_CONCURRENCY
                if processed % 20 < EMBED_CONCURRENCY or processed >= len(batches):
                    elapsed = time.perf_counter() - started
                    rate = written / elapsed if elapsed else 0
                    remaining = len(new_docs) - written - failed_count
                    eta = remaining / rate / 3600 if rate else 0
                    logger.info(
                        f"Progreso: {written}/{len(new_docs)} escritos "
                        f"({rate:.1f} docs/s, quedan ~{eta:.1f} h)"
                    )

        logger.info(f"Indexación finalizada. {written} agregados, {failed_count} fallidos.")
        logger.info(f"Total en store: {self.store.count_documents()}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Indexación del RAG de ciberseguridad.")
    ap.add_argument("--include-cve", action="store_true",
                    help="Indexar también los CVE de NVD (cves_page_*.json). Off por "
                         "defecto: son ~8k documentos atómicos y su embedding en CPU "
                         "domina el tiempo de corrida.")
    args = ap.parse_args()

    Indexer(include_cve=args.include_cve).run()