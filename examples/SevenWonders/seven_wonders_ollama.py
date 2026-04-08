import os



from haystack import Pipeline, Document
from haystack.utils import Secret
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.retrievers.pgvector import (
    PgvectorEmbeddingRetriever,
    PgvectorKeywordRetriever
)
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack_integrations.components.generators.ollama import OllamaGenerator

from datasets import load_dataset
import json

token = f"postgresql://avdbuser:avdbpass@localhost:5433/pgvdb"
print(token)
document_store = PgvectorDocumentStore(
    connection_string= Secret.from_token(token),
    embedding_dimension =1024,
    table_name="seven_wonders"
)

dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
docs : list[Document] = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

EMBEDDING_MODEL_NAME="bge-m3"
MODEL_NAME="phi4"
OLLAMA_BASE_URL="http://localhost:11434"

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

if (len(new_docs) == 0):
    print("\nNo new documents to embed.")
else:
    print(f"\nEmbedding and writing {len(new_docs)} new documents to the database...")
    print(json.dumps([doc.to_dict() for doc in new_docs], indent=2))
    doc_embedder = OllamaDocumentEmbedder(
            model = EMBEDDING_MODEL_NAME,
            url = OLLAMA_BASE_URL,
            batch_size=32,
            keep_alive=-1
        )

    docs_with_embeddings = doc_embedder.run(new_docs)
    document_store.write_documents(docs_with_embeddings["documents"], policy=DuplicatePolicy.OVERWRITE)

text_embedder = OllamaTextEmbedder(
    model = EMBEDDING_MODEL_NAME,
    url = OLLAMA_BASE_URL
)

retriever = PgvectorEmbeddingRetriever(document_store=document_store)
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

# generator = OpenAIGenerator(model="gpt-3.5-turbo")
generator = OllamaGenerator(
    model=MODEL_NAME,
    url=OLLAMA_BASE_URL,
    generation_kwargs={
        "num_predict": 1000,
        "temperature": .5,
    },
    keep_alive=-1,  # <-- mantiene en memoria
    timeout=450
)

basic_rag_pipeline = Pipeline()
# Add components to your pipeline
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", generator)

# Now, connect the components to each other
basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
basic_rag_pipeline.connect("prompt_builder", "llm")
while True:

    question = input("\nWhat do you want to ask about the Seven Wonders of the Ancient World? (type 'exit' to quit) ")
    if question.lower() == "exit":
        print("Goodbye!")
        break
    results = basic_rag_pipeline.run({
        "text_embedder": {"text": question},
        "prompt_builder": {"question": question}
    })

    print(results["llm"]["replies"][0])