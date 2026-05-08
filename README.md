# Infraestructura Básica para RAGs

Repositorio con infraestructura y ejemplos para construir sistemas RAG (Retrieval-Augmented Generation) locales.

## Componentes

- **OpenWebUI** — interfaz web para interactuar con LLMs, similar a ChatGPT pero autohosteada. Permite conectar con Ollama y APIs compatibles con OpenAI.
- **Ollama** — plataforma para correr LLMs localmente sin depender de la nube. Soporta Windows, macOS y Linux.
- **pgvector** — extensión de PostgreSQL para almacenar y buscar vectores de alta dimensión, usada como vector database del RAG.
- **Pipelines (OpenWebUI)** — servidor donde viven los pipelines RAG. Los archivos `.py` en `infrastructure/appdata/pipelines/` se cargan automáticamente.

## Requisitos

- Docker y Docker Compose
- GPU con soporte NVIDIA (opcional pero recomendado para Ollama)
- En Fedora/RHEL: SELinux activo (los volúmenes ya están configurados con `:z`)

## Startup

```bash
git clone <repo>
cd infrastructure
docker compose up -d
```

Luego abrí OpenWebUI en `http://localhost:8180`.

## Documentos para el RAG

Colocá tus archivos (`.pdf`, `.docx`, `.md`, `.txt`) en:

```
infrastructure/appdata/rawdata/
```

Al iniciar el contenedor de pipelines, los documentos se indexan automáticamente en pgvector. Si agregás documentos nuevos, reiniciá el contenedor:

```bash
docker restart infrastructure-pipelines-1
```

Para re-indexar desde cero (por ejemplo al cambiar el splitter):

```bash
docker exec infrastructure-vdb-1 psql -U avdbuser -d pgvdb -c "TRUNCATE TABLE local_docs;"
docker restart infrastructure-pipelines-1
```

## Pipeline RAG

El pipeline principal está en `infrastructure/appdata/pipelines/pipeline_local_docs.py`. Usa:

- **Embeddings**: `bge-m3` via Ollama
- **LLM**: configurable via Valves en OpenWebUI (por defecto `llama3.2:3b`)
- **Retrieval**: híbrido (embeddings + keywords) sobre pgvector
- **Splitter**: `RecursiveDocumentSplitter` de Haystack

Los parámetros (modelo, chunk size, overlap, top-k) se pueden ajustar desde **Admin Panel → Pipelines → Valves** en OpenWebUI sin reiniciar.

## Inspección de Chunking

Para comparar cómo distintos splitters dividen los documentos:

```bash
cd infrastructure

# Splitter por defecto (recursive)
./inspect.sh

# Por oración (Haystack)
./inspect.sh --splitter sentence --split_length 5

# Por tokens respetando oraciones (LlamaIndex)
./inspect.sh --splitter llama_sentence --split_length 256

# Semántico por embeddings (LlamaIndex + bge-m3)
./inspect.sh --splitter llama_semantic

# Semántico con threshold ajustado (menos chunks → más contexto por chunk)
./inspect.sh --splitter llama_semantic --threshold 80

# Jerárquico multinivel (LlamaIndex) — hoja=100, medio=400, raíz=1600
./inspect.sh --splitter llama_hierarchical --split_length 100

# Recursivo por caracteres (LangChain)
./inspect.sh --splitter langchain_recursive --split_length 150
```

Splitters disponibles:

| Nombre | Librería | Descripción |
|---|---|---|
| `recursive` | Haystack | Separa por párrafos → oraciones → saltos de línea → palabras |
| `word` | Haystack | Chunks de N palabras exactas |
| `sentence` | Haystack | Chunks de N oraciones |
| `passage` | Haystack | Chunks de N párrafos |
| `llama_sentence` | LlamaIndex | Tokens respetando límites de oraciones |
| `llama_semantic` | LlamaIndex | Corta donde cambia el tema (usa embeddings). `--threshold` controla la sensibilidad (1-99, default 95). Menor = más cortes. |
| `llama_hierarchical` | LlamaIndex | Genera chunks en 3 niveles (hoja → medio → raíz). `--split_length` define el tamaño de la hoja; los niveles superiores escalan x4 y x16. Útil para small-to-big retrieval. |
| `langchain_recursive` | LangChain | Recursivo por caracteres con separadores configurables |

Los resultados se guardan en `infrastructure/splitter_<nombre>.txt`.

## Creando tu propio RAG

El código de tu RAG puede vivir en cualquier carpeta fuera de `infrastructure` e incluso fuera del repositorio. La carpeta `examples/` tiene ejemplos usando Haystack, pero podés usar LlamaIndex, LangChain o cualquier otra librería. Sigue por el camino de tu arcoiris 🌈.
