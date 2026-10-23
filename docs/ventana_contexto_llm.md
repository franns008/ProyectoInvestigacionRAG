# Ventana de contexto del LLM

Cómo decide el pipeline cuántos tokens de contexto le pide a Ollama, cómo fijarla para el
modelo que corre cada uno y cómo comprobar que es la que se usa de verdad. El código está en
[`src/pipeline/common/llm.py`](../src/pipeline/common/llm.py).

## Por qué importa

La ventana (`num_ctx` en Ollama) es el lugar que comparten el prompt y la respuesta. El
prompt del chat lleva reglas, documentos del corpus, hallazgos adjuntos, historial y la
pregunta. Si no entra, Ollama **recorta el principio sin avisar**, y ahí están las reglas.

Por eso el pipeline **siempre** manda `num_ctx` explícito y nunca deja que Ollama use su
default. El default depende de la versión y la configuración del servidor de Ollama: no hay
forma de saberlo desde el pipeline.

Una ventana más grande no es gratis: el KV cache crece con ella y ocupa VRAM (o RAM, si el
modelo corre en CPU). Con poca VRAM, subir la ventana puede hacer que Ollama mande capas del
modelo a CPU y la generación se vuelva mucho más lenta. Cada uno la ajusta a su hardware.

## Cómo se resuelve

La misma lógica la usan la generación (`build_generator`) y lo que lee la extensión, así que
los dos llegan siempre al mismo número.

| `LLM_NUM_CTX` (en `infrastructure/.env`) | Ventana que se usa |
|---|---|
| sin definir o vacía | el valve `num_ctx` del pipeline (default `8192`) |
| `N` (entero > 0) | `N`, por encima del valve |
| `0` | el `num_ctx` que fija el Modelfile del modelo en uso |

- El modelo en uso es `LLM_MODEL` si está definida y si no, el valve `llm_model`: la misma
  precedencia que en [`arquitectura.md`](arquitectura.md) §4.2.
- Con `0`, el pipeline le pide el Modelfile a Ollama (`POST /api/show`) y manda ese valor
  explícito igual: el resultado es el mismo que dejar que Ollama lo aplique, pero el pipeline
  sabe cuál es.
- Si con `0` el Modelfile no fija `num_ctx`, el pipeline **no arranca** y el error lo dice.
  Lo mismo con un valor que no sea un entero `>= 0`. No hay vuelta silenciosa al default.
- El valor del Modelfile se pide una vez por modelo y queda cacheado. Si se recrea el
  modelo con otro `num_ctx`, hay que reiniciar `pipelines`.
- `pipeline_dependencias` (las explicaciones del escaneo) usa el mismo generador. Sus valves
  no traen `num_ctx`, así que sólo manda una ventana si `LLM_NUM_CTX` está definida. Sus
  prompts son cortos.

Las variables de entorno ganan por la misma razón que `LLM_MODEL`: el servidor guarda los
valves en `valves.json` cada vez que se editan, y un valor guardado ahí no se entera de un
cambio en `.env`.

## Fijar la ventana para tu modelo

Hay dos formas. Las dos se aplican recreando el container (`docker compose ... up -d`, con
los mismos `-f` con los que se levantó el stack), sin rebuild.

**Un número directo.** Alcanza para probar o si se usa un modelo tal como se bajó:

```bash
# infrastructure/.env
LLM_MODEL=<tu modelo>
LLM_NUM_CTX=16384
```

**La ventana dentro del modelo.** Conviene cuando la ventana va atada al modelo, por ejemplo
si se tienen varios modelos preparados para el mismo hardware y se cambia entre ellos: cada
uno lleva la suya y en `.env` sólo se cambia `LLM_MODEL`.

```bash
# Desde infrastructure/. Mismos pesos que el base: no ocupa disco extra.
printf 'FROM <modelo base>\nPARAMETER num_ctx 16384\n' > /tmp/mf
docker compose exec -T ollama sh -c 'cat > /tmp/mf' < /tmp/mf
docker compose exec ollama ollama create <nombre nuevo> -f /tmp/mf
```

```bash
# infrastructure/.env
LLM_MODEL=<nombre nuevo>
LLM_NUM_CTX=0
```

`ollama create` copia del modelo base el template y los parámetros que no se pisan en el
Modelfile. Para ver qué fija un modelo: `docker compose exec ollama ollama show --modelfile
<modelo>`.

## Qué ve la extensión

El servidor de Pipelines sólo expone los valves (`GET /<pipeline>/valves`), así que la
ventana efectiva se publica ahí, como el campo `effective_num_ctx`:

```bash
curl -s -H "Authorization: Bearer <PIPESERVER_KEY>" \
    localhost:9099/pipeline_ciberseguridad/valves
# {..., "num_ctx": 8192, ..., "effective_num_ctx": 32768}
```

- Es un campo calculado y de sólo lectura. No aparece en el schema de edición
  (`/valves/spec`) y el servidor lo ignora al restaurar `valves.json`, así que no se puede
  pisar a mano ni queda congelado.
- `num_ctx` sigue mostrando el valve. Cuando `LLM_NUM_CTX` está definida, la ventana real es
  `effective_num_ctx` y no `num_ctx`.
- El medidor de contexto de la extensión
  ([`extension/src/contextMeter.ts`](../extension/src/contextMeter.ts)) usa
  `effective_num_ctx` como total de la ventana. Sólo si el servidor no lo publica vuelve al
  valve `num_ctx`. Ver [`arquitectura.md`](arquitectura.md) §4.4.

## Cómo verificar

1. **Lo que resolvió el pipeline:** `effective_num_ctx` en el `curl` de arriba.
2. **Lo que recibió Ollama:** después de una consulta, en el log de Ollama la línea
   `starting llama-server` del LLM trae la ventana como `-c N`:

   ```bash
   docker compose logs ollama --since 5m | grep -E "starting llama-server|n_ctx  |offloaded"
   ```

   Ojo con no confundirla con la del embedder, que también aparece y lleva `--embedding`.
3. **Si entra en GPU:** en el mismo log, `offloaded N/M layers to GPU`. Si `N < M`, una
   parte del modelo corre en CPU: bajar la ventana o usar un modelo más chico. Es más
   confiable que `ollama ps`, que con algunos modelos informa una colocación que no es la
   real.
