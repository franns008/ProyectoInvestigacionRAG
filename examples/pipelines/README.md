Para hacer andar el pipeline debemos colocar el archivo "pipeline_local_docs.py" en el directorio /infrastructura/appdata/pipelines.
Una vez puesto alli, debemos colocar los archivos que queremos que maneje el RAG en /rawdata (Directorio que se encuentra en /infrastructura/appdata/pipelines)
Hecho esto, hacer docker compose restart pipelines  si tenemos corriendo el docker compose. Por cada cambio que hagamos en el archivo .py y queramos que se noten los cambios, debemos usar dicho comando.
Cada nueva libreria debe estar especificada en el requirements del archivo pipeline_local_docs.py