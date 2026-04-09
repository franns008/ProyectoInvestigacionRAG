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
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack_integrations.components.generators.ollama import OllamaGenerator

from datasets import load_dataset
import json

token = f"postgresql://avdbuser:avdbpass@localhost:5433/pgvdb"
print(token)
document_store = PgvectorDocumentStore(
    connection_string=Secret.from_token(token),
    embedding_dimension=1024,
    table_name="seven_wonders"
)

dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
docs: list[Document] = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

EMBEDDING_MODEL_NAME = "bge-m3"
MODEL_NAME = "qwen2.5:1.5b"
OLLAMA_BASE_URL = "http://localhost:11434"

filtered_docs = document_store.filter_documents()

# Clave compuesta existente en DB: (url, _split_id)
filtered_keys = {
    (doc.meta.get("url"), doc.meta.get("_split_id"))
    for doc in filtered_docs
    if doc.meta.get("url") is not None and doc.meta.get("_split_id") is not None
}

# Solo splits faltantes
new_docs = [
    doc for doc in docs
    if (doc.meta.get("url"), doc.meta.get("_split_id")) not in filtered_keys
]

if len(new_docs) == 0:
    print("\nNo new documents to embed.")
else:
    print(f"\nEmbedding and writing {len(new_docs)} new documents to the database...")
    print(json.dumps([doc.to_dict() for doc in new_docs], indent=2))
    doc_embedder = OllamaDocumentEmbedder(
        model=EMBEDDING_MODEL_NAME,
        url=OLLAMA_BASE_URL,
        batch_size=32,
        keep_alive=-1
    )

    docs_with_embeddings = doc_embedder.run(new_docs)
    document_store.write_documents(docs_with_embeddings["documents"], policy=DuplicatePolicy.OVERWRITE)

text_embedder = OllamaTextEmbedder(
    model=EMBEDDING_MODEL_NAME,
    url=OLLAMA_BASE_URL
)

# Hybrid retrieval: combina búsqueda semántica (embedding) con búsqueda textual (keyword).
# El embedding retriever es bueno para preguntas conceptuales; el keyword retriever para
# nombres propios y términos exactos. Juntos cubren más casos y mejoran la calidad.
# top_k=5 reduce los documentos enviados al LLM vs. el default de 10,
# achicando el prompt y acelerando la generación en CPU.
embedding_retriever = PgvectorEmbeddingRetriever(document_store=document_store, top_k=5)
keyword_retriever = PgvectorKeywordRetriever(document_store=document_store, top_k=5)
document_joiner = DocumentJoiner()

template = """
        Given the following information, answer the question.

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{question}}
        Answer:
        """

prompt_builder = PromptBuilder(template=template)

generator = OllamaGenerator(
    model=MODEL_NAME,
    url=OLLAMA_BASE_URL,
    generation_kwargs={
        "num_predict": 1000,  # máximo de tokens a generar en la respuesta
        "temperature": 0.5,
        "num_ctx": 2048,      # ventana de contexto del LLM; reducirla baja el uso de RAM
                              # y acelera cada paso de generación en CPU (default: 4096+)
    },
    keep_alive=-1,
    timeout=450,
    streaming_callback=lambda chunk: print(chunk.content, end="", flush=True),  # imprime cada token apenas se genera, sin esperar la respuesta completa
)

basic_rag_pipeline = Pipeline()
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("embedding_retriever", embedding_retriever)
basic_rag_pipeline.add_component("keyword_retriever", keyword_retriever)
basic_rag_pipeline.add_component("document_joiner", document_joiner)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", generator)

basic_rag_pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
basic_rag_pipeline.connect("embedding_retriever", "document_joiner")
basic_rag_pipeline.connect("keyword_retriever", "document_joiner")
basic_rag_pipeline.connect("document_joiner", "prompt_builder.documents")
basic_rag_pipeline.connect("prompt_builder", "llm")

while True:
    question = input("\nWhat do you want to ask about the Seven Wonders of the Ancient World? (type 'exit' to quit) ")
    if question.lower() == "exit":
        print("Goodbye!")
        break
    basic_rag_pipeline.run({
        "text_embedder": {"text": question},
        "keyword_retriever": {"query": question},
        "prompt_builder": {"question": question}
    })
    print()  # salto de línea al terminar el streaming (los tokens no incluyen \n final)
