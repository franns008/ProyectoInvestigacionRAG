"""Conexión a pgvector y a Ollama, compartida por el pipeline, la indexación y el eval."""

from haystack.utils import Secret
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore

DB_CONNECTION       = "postgresql://avdbuser:avdbpass@vdb:5432/pgvdb"
OLLAMA_URL          = "http://ollama:11434"
DB_TABLE            = "ciberseguridad_docs"
KEYWORD_INDEX       = "ciberseguridad_keyword_index"
EMBEDDING_DIMENSION = 2560  # nativa de qwen3-embedding:4b (sin truncar vía MRL)


def get_document_store(
    table_name: str = DB_TABLE,
    keyword_index_name: str = KEYWORD_INDEX,
) -> PgvectorDocumentStore:
    """Los defaults son el store de producción; tabla e índice se overridean para
    experimentos en tablas paralelas (ver docs/data_splitting.md)."""
    return PgvectorDocumentStore(
        connection_string=Secret.from_token(DB_CONNECTION),
        embedding_dimension=EMBEDDING_DIMENSION,
        table_name=table_name,
        keyword_index_name=keyword_index_name,
    )
