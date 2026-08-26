# 🏗️ Arquitectura Desacoplada: RAG de Ciberseguridad

Este documento explica la reestructuración del sistema RAG (Retrieval-Augmented Generation) para ciberseguridad, separando la lógica de inferencia (chat) de la lógica de indexación (procesamiento de datos).

---

## 🚀 1. Visión General: ¿Por qué se dividió la arquitectura?

Anteriormente, el sistema operaba como un monolito: cada vez que el chatbot (Open WebUI) se inicializaba, cargaba en memoria todas las herramientas de extracción (como `marker-pdf` que consume ~3GB de RAM), escaneaba los archivos locales y realizaba los embeddings.

Esto causaba:

1. **Tiempos de arranque lentos** (Cold Starts).
2. **Consumo innecesario de recursos** (RAM y VRAM ocupada permanentemente).
3. **Bloqueos del chat** al procesar nuevos documentos.

**La nueva arquitectura resuelve esto separando las responsabilidades en dos procesos independientes que se comunican únicamente a través de la base de datos vectorial (pgvector).**

---

## 📂 2. Nueva Estructura de Directorios

El código ahora se divide claramente en dos ecosistemas dentro del volumen de Docker:

```text
src/pipeline/
├── pipeline_ciberseguridad.py      # 💬 Inferencia: El motor del Chatbot
├── indexing/                       # ⚙️ Indexación: El motor de Ingesta
│   ├── converters.py               # Lógica para parsear XML (CWE) y JSON (CVE)
│   └── run_indexing.py             # Script standalone para procesar e indexar
├── rawdata/                        # 📄 Carpeta donde se depositan PDFs, XMLs y JSONs
└── ...
```

---

## 💬 3. Inferencia (El Chatbot)

**Archivo principal:** `pipeline_ciberseguridad.py`

Este componente es el encargado de interactuar con el usuario a través de Open WebUI. Su única función es **leer**.

- **Qué hace:** Recibe la pregunta, la transforma en palabras clave/embeddings, busca en pgvector (usando Retriever + Reranker), y envía el contexto al LLM (`qwen2.5` o `llama3.1`).
- **Ventaja:** Al no cargar librerías pesadas de procesamiento de PDFs, inicializa en milisegundos y mantiene la memoria libre para el modelo generativo.

---

## ⚙️ 4. Indexación (El Motor de Datos)

**Archivos principales:** `indexing/run_indexing.py` y `indexing/converters.py`

Este componente es un proceso "Batch" (por lotes) que se ejecuta en segundo plano. Su única función es **escribir**.

- **Qué hace:** Escanea la carpeta `rawdata/`, convierte PDFs a Markdown usando OCR (`marker-pdf`), extrae vulnerabilidades de XML/JSON, divide los textos largos (chunking), calcula los embeddings usando Ollama, y guarda todo en pgvector.
- **Ventaja:** Solo consume memoria RAM/VRAM de forma intensiva mientras se ejecuta. Una vez que finaliza, se apaga y libera los recursos.

---

## 🛠️ 5. Guía de Ejecución: ¿Cómo hacerlo andar?

### A. Encender la Infraestructura (El Chatbot)

El chatbot funciona de manera continua. Para levantar todo el sistema, basta seguir los pasos que se nombraron en los demas documentos.

```bash
# Levantar los contenedores en segundo plano
docker compose up -d
```

*Una vez encendido, podés acceder a Open WebUI (ej. `http://localhost:8180`), seleccionar el modelo "RAG ciberseguridad" y comenzar a chatear. El bot responderá utilizando los datos que ya estén almacenados en la base de datos.*

### B. Ejecutar la Indexación (Actualizar la Base de Datos)

Cuando agregues nuevos archivos (PDF, XML, JSON) a la carpeta `rawdata/`, el chatbot no se enterará automáticamente. Debes ejecutar el script de indexación para procesarlos.

1. Colocá tus nuevos archivos en `data/raw/` (que está mapeado a `rawdata/` en el contenedor).
2. Ejecutá el siguiente comando en tu terminal para lanzar el proceso de indexación dentro del contenedor de pipelines:

```bash
docker exec -it infrastructure-pipelines-1 python /app/pipelines/indexing/run_indexing.py
```

**¿Qué sucederá?**

1. Verás logs en tu consola indicando que se encontraron nuevos documentos.
2. Si hay PDFs, cargará `marker-pdf` en memoria temporalmente.
3. Generará los embeddings y los insertará en pgvector.
4. Finalizará y liberará la memoria.
5. **A partir de ese instante, el chatbot en Open WebUI ya podrá encontrar la nueva información sin necesidad de reiniciar nada.**

---

*Nota: Asegurate de tener configuradas correctamente tus variables de entorno (como `LLM_PROVIDER=ollama`) en el archivo `.env` raíz para que el sistema utilice tu modelo local adecuadamente.*
