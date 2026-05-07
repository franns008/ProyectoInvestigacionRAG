import os

os.environ["MARKER_DEVICE"] = "cuda"  # "cpu", "cuda", "mps", etc.
os.environ["INFERENCE_RAM"] = "12"   # GB de RAM para modelos
#Debe estar antes de importar cualquier cosa de marker
# ───────────────────────────────────────────── 
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered


INPUT_FILE = os.path.join(os.getcwd(), "infrastructure", "appdata", "rawdata", "Agente Asistente Personal con Calendario.pdf")
print(f"Archivo de entrada: {INPUT_FILE}")
# ─────────────────────────────────────────────
# 2. CONFIG: diccionario con tus opciones
# ─────────────────────────────────────────────
config = {
    "output_format": "markdown",   # markdown | json | html | chunks
    "force_ocr": False,            # True si el PDF tiene texto raro/escaneado
    "strip_existing_ocr": False,   # True para re-OCR desde cero
    "disable_image_extraction": False,  # True = más rápido, sin imágenes
    "paginate_output": False,      # True = agrega separadores de página
    "use_llm": False,              # True = mayor precisión, necesita API key
    "page_range": None,            # "0-5,10" para páginas específicas
    "debug": False,
}

# ─────────────────────────────────────────────
# 3. CONFIG PARSER: el orquestador central
#    Toma tu dict y genera la configuración
#    final para cada componente
# ─────────────────────────────────────────────
config_parser = ConfigParser(config)

# ─────────────────────────────────────────────
# 4. MODELS: carga todos los modelos de ML
#    ⚠️ Esto descarga ~1-4 GB la primera vez
#    ⚠️ Es costoso — hacerlo UNA SOLA VEZ
# ─────────────────────────────────────────────
artifact_dict = create_model_dict()

# ─────────────────────────────────────────────
# 5. CONVERTER: ensambla el pipeline completo
# ─────────────────────────────────────────────
converter = PdfConverter(
    config=config_parser.generate_config_dict(),  # config final (dict)
    artifact_dict=artifact_dict,                  # modelos de ML
    processor_list=config_parser.get_processors(), # lista de processors activos
    renderer=config_parser.get_renderer(),         # renderer según output_format
    llm_service=config_parser.get_llm_service(),  # None si use_llm=False
)

# ─────────────────────────────────────────────
# 6. CONVERSIÓN: el converter es callable
#    Se llama como función pasándole el path
# ─────────────────────────────────────────────
rendered = converter(INPUT_FILE)

# ─────────────────────────────────────────────
# 7. OUTPUT: extraer el contenido del resultado
# ─────────────────────────────────────────────
text, metadata, images = text_from_rendered(rendered)

# Guardar el markdown completo
with open("salida.md", "w", encoding="utf-8") as f:
    f.write(text)

# Guardar los metadatos en formato JSON
import json
with open("metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)

# Guardar las imágenes extraídas (si las hay)
if images:
    os.makedirs("imagenes_salida", exist_ok=True)
    for img_name, img_bytes in images.items():
        img_path = os.path.join("imagenes_salida", img_name)
        with open(img_path, "wb") as f:
            f.write(img_bytes)

print("¡Conversión finalizada! Se guardaron: 'salida.md', 'metadata.json' y las imágenes en la carpeta 'imagenes_salida/'.")