# Infraestructura Básica para RAGs
Este es un repositorio con ejemplos de infraestructura básica para RAGs.
Incluye
- **Ollama**: Ollama es una plataforma de código abierto que permite descargar, ejecutar y gestionar modelos de lenguaje grandes (LLM) directamente en el ordenador local, sin depender de servicios en la nube. Fue lanzado en julio de 2023 por Jeffrey Morgan y está diseñado para ejecutarse en sistemas Windows, macOS y Linux, funcionando principalmente a través de la terminal o mediante una API para integraciones.
- **pgvector**: es una extensión de código abierto para PostgreSQL que añade soporte nativo para el almacenamiento, indexación y búsqueda de similitud de vectores de alta dimensión, permitiendo integrar capacidades avanzadas de inteligencia artificial directamente en bases de datos relacionales existentes.

## Puesta en marcha

Paso a paso (CPU o GPU NVIDIA/AMD): [`docs/arquitectura.md`](docs/arquitectura.md) §7.

## Generación local en Ollama

Todo el RAG corre local: los **embeddings** (`qwen3-embedding:4b`) y la **generación**
salen del mismo Ollama del compose. No hace falta ninguna API Key.

El modelo de generación se elige con `LLM_MODEL` en `infrastructure/.env` (vacío = el
default del pipeline, `qwen2.5:3b-instruct`) y tiene que estar `ollama pull`-eado.
Anda en CPU; con GPU va bastante más rápido, y hay un overlay de Compose por familia de
placa (NVIDIA vía CUDA, AMD vía Vulkan). Ver `docs/arquitectura.md` §8.

**Guía completa: [`docs/arquitectura.md`](docs/arquitectura.md).**

## Escaneo de dependencias vulnerables

Además del chat, el repo trae una herramienta que lee un `requirements.txt` y devuelve
sus dependencias vulnerables **priorizadas por explotabilidad real** (CISA KEV → EPSS →
CVSS), con una extensión de VSCode que la muestra.

**No necesita el stack**: ni Docker, ni pgvector, ni Ollama, ni API Key. Es lógica
determinística sobre tres archivos descargados, y corre en ~1 segundo.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python src/ingestion/fetch_all.py --only osv epss kev
.venv/bin/python -m pytest
PYTHONPATH=src/pipeline .venv/bin/python -m deps.cli <tu-requirements.txt> --data data/raw
```

**Puesta en marcha completa: [`docs/escaneo_dependencias_setup.md`](docs/escaneo_dependencias_setup.md)**
· diseño y justificación: [`docs/escaneo_dependencias.md`](docs/escaneo_dependencias.md).
