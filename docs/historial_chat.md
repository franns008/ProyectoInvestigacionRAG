# Historial de mensajes en el chat de ciberseguridad

> **Estado: PROPUESTO — BLOQUEADO.** No hay nada implementado.
>
> **Depende de que primero se cambie el prompt para recibir el contexto del escaneo como
> contexto propio** (hoy la extensión lo manda como primer *mensaje*). Mientras eso no
> esté, este plan no se ejecuta: el bloque de historial y el bloque de contexto comparten
> el mismo prompt y el mismo presupuesto de tokens, y hacerlo en el orden inverso obliga a
> escribir un caso especial para `messages[0]` que después hay que borrar.
>
> Archivos que toca cuando se desbloquee: `src/pipeline/chat/history.py` (nuevo),
> [`pipeline_ciberseguridad.py`](../src/pipeline/pipeline_ciberseguridad.py),
> `tests/test_history.py` (nuevo).
>
> **Restricción de diseño:** no romper el uso desde OpenWebUI, que ya tiene su propio
> historial y además dispara prompts internos por el mismo pipeline. El criterio duro es:
> **con `history_turns = 0`, o ante un prompt interno de OpenWebUI, el prompt rendereado
> tiene que quedar byte a byte igual al de hoy.**

## Qué se quiere

Que el chat de la extensión de VSCode (`ChatPanel`, "Hablar en profundidad") tenga
memoria conversacional: que "¿y me afecta si sólo abro PNG?" se entienda como
continuación de la respuesta anterior, y no como una pregunta suelta.

## Estado actual: no hay historial en ningún cliente

El pipeline **recibe** el historial y casi no lo usa. `pipe()`
([`:422`](../src/pipeline/pipeline_ciberseguridad.py)) tiene el parámetro
`messages: List[dict]` y lo pasa a un único lugar:

- `resolve_vuln_ids(user_message, messages)` ([`:173`](../src/pipeline/pipeline_ciberseguridad.py)),
  que mira los últimos `_HISTORY_TURNS = 4` turnos **sólo** para heredar identificadores
  `CVE-x`/`CWE-x` cuando el turno actual no menciona ninguno. Eso alimenta `id_lookup` y
  `build_ranker_query` ([`:252`](../src/pipeline/pipeline_ciberseguridad.py)).

Todo lo demás es estrictamente single-turn:

- `PROMPT_TEMPLATE` ([`:69`](../src/pipeline/pipeline_ciberseguridad.py)) interpola
  `documents` y `{{question}}`. **El LLM nunca ve los turnos anteriores**, ni desde
  OpenWebUI ni desde la extensión.
- `build_embedding_query` / `build_keyword_query` se arman sólo con `user_message`.
- `pipe()` pasa `{"prompt_builder": {"question": user_message}}`
  ([`:458`](../src/pipeline/pipeline_ciberseguridad.py)) y nada más.

La extensión, en cambio, **ya manda el historial completo**: `ChatPanel` acumula
`this.messages` en el cliente ([`chatPanel.ts:37`](../extension/src/chatPanel.ts), `:67`)
y lo envía entero en cada `POST /v1/chat/completions` (`:86`). O sea que **el transporte
ya existe**; lo que falta es que el pipeline consuma `messages`.

## Restricciones del server de Pipelines (verificadas en el container, 2026-09-17)

Tres cosas que condicionan el diseño y que conviene no volver a averiguar:

| Hecho | Dónde se verificó | Consecuencia |
|---|---|---|
| `user_message` es el **último** mensaje con `role == "user"` | `/app/utils/pipelines/main.py:28` (`get_last_user_message`) | El turno actual viene **duplicado** dentro de `messages`. El corte del historial se hace por el **índice** de ese último `user`, no comparando strings: no depende de que el duplicado esté exactamente al final. |
| `OpenAIChatMessage.content: str \| List` | `/app/schemas.py:4-8` | El normalizador tiene que soportar `content` multimodal (lista de partes con `{"type": "text", ...}`), igual que ya hace `_message_text` ([`:163`](../src/pipeline/pipeline_ciberseguridad.py)). |
| `src/pipeline/` está bind-mounteado a `/app/pipelines` | [`docker-compose.yml:111-116`](../infrastructure/docker-compose.yml) (volumen `pipelines`, device `../src/pipeline/`) | Un directorio nuevo `chat/` aparece en el container **sin rebuild de la imagen**: alcanza con reiniciar el servicio `pipelines`. |

## La decisión central: stateless, el historial lo manda el cliente

Se descarta guardar sesiones server-side indexadas por `chat_id`. Razones:

- La extensión **ya** manda el historial, y OpenWebUI **ya** manda el historial. El
  estado existe dos veces del lado del cliente; una tercera copia en el pipeline es una
  fuente de verdad más que se puede desincronizar.
- El server recarga los pipelines al editar valves y al reiniciar: el estado en memoria se
  pierde justo cuando el usuario está en medio de una conversación.
- Habría que escribir evicción y expiración para un problema que el cliente ya resolvió.

Consecuencia buena: **un solo camino de código para los dos clientes**. No hay que
detectar si quien pregunta es la extensión o OpenWebUI.

## Por qué hay que esperar al cambio de contexto

Hoy el primer elemento de `messages` que manda la extensión es el JSON del hallazgo
(`contextFor(finding)` en [`chatPanel.ts`](../extension/src/chatPanel.ts)), no una
pregunta. Si se implementa el historial antes de moverlo:

- Ese JSON entra al prompt como un turno de usuario, recortado por el tope por mensaje, y
  a partir del tercer intercambio **se cae por la ventana de `history_turns`** — o sea que
  el modelo pierde el paquete y la versión instalada justo cuando la charla se pone larga.
- La alternativa es un caso especial que detecte `messages[0]`, lo parsee y lo rendee
  aparte, fuera de la ventana. Eso es trabajo que se tira a la basura en cuanto el
  contexto viaje por su propio canal.

Cuando el contexto salga de `messages`, el historial queda **puramente conversacional**:
descartar el turno duplicado, descartar meta-prompts, recortar. Sin casos especiales.
`history.py` no necesita ni una línea distinta por ese cambio.

## Diseño: `src/pipeline/chat/history.py`

Módulo nuevo, **lógica pura**: no importa Haystack ni torch, así que se testea con
`pytest` sin levantar nada — misma disciplina que [`deps/`](../src/pipeline/deps/) y
[`explain/`](../src/pipeline/explain/). Se importa desde los tests como `chat.history`
gracias al `sys.path.insert` que ya hace [`tests/conftest.py:14`](../tests/conftest.py).

Superficie pública:

| Nombre | Qué es |
|---|---|
| `HISTORY_TURNS = 4` | mensajes conservados (≈2 intercambios). Reemplaza `_HISTORY_TURNS` ([`:159`](../src/pipeline/pipeline_ciberseguridad.py)), que pasa a importarse de acá para no tener la constante duplicada. |
| `HISTORY_MAX_CHARS = 2000` | presupuesto total del bloque de historial. |
| `HISTORY_MSG_MAX_CHARS = 600` | tope por mensaje. Con `max_tokens = 512`, una respuesta del asistente son ~2000 chars: sin este tope, dos respuestas desplazan del prompt a los 4 docs reranqueados. |
| `META_PROMPT_MARKERS` | migra tal cual desde [`:158`](../src/pipeline/pipeline_ciberseguridad.py). |
| `message_text(msg)` | migra `_message_text` ([`:163`](../src/pipeline/pipeline_ciberseguridad.py)). |
| `is_meta_prompt(text)` | el chequeo de marcadores, hoy inline dentro de `resolve_vuln_ids`. |
| `build_history(messages, current_user_message, max_turns, max_chars, msg_max_chars)` | devuelve `list[{"role", "content"}]` en orden cronológico. |

Reglas de `build_history`, en orden:

1. Cortar en el índice del último `role == "user"`; el historial es **lo anterior a ese
   índice**. Si no hay ningún `user`, todo es historial.
2. Descartar entradas que no sean dict, y todo rol que no sea `user`/`assistant`. Los
   `system` que manda OpenWebUI se ignoran: las instrucciones las pone `PROMPT_TEMPLATE`, y
   meter otras en el mismo prompt es pedirle al modelo que obedezca dos jefes.
3. Descartar vacíos, descartar meta-prompts y descartar cualquier mensaje cuyo texto sea
   igual al turno actual (defensa por si el cliente reenvía).
4. Colapsar whitespace: un mensaje, una línea. El prompt queda compacto y el log grepeable,
   que es la convención del archivo (ver `_one_line`).
5. Truncar cada mensaje a `msg_max_chars` conservando el arranque, con marca de recorte.
6. Quedarse con los últimos `max_turns` y después ir tirando **el más viejo** hasta entrar
   en `max_chars`. Si sobrevive uno solo y todavía no entra, truncarlo a `max_chars` en vez
   de devolver vacío.

`max_turns <= 0` o `max_chars <= 0` devuelven `[]`. Ese es el escape hatch.

## Cambios en `pipeline_ciberseguridad.py`

1. `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))`, con el comentario que
   ya está en [`pipeline_dependencias.py:53-62`](../src/pipeline/pipeline_dependencias.py).
   **Es el único riesgo real del cambio**: hoy este archivo no importa módulos hermanos, y
   el server carga cada pipeline con `spec_from_file_location`, que no agrega el directorio
   al `sys.path`. Si el import falla, el server **mueve el archivo a `pipelines/failed/`** y
   el modelo desaparece con un `404 Pipeline pipeline_ciberseguridad not found`.
2. Borrar `_message_text`, `_META_PROMPT_MARKERS` y `_HISTORY_TURNS`; pasan a venir de
   `chat.history`. `resolve_vuln_ids` usa los importados.
3. `PROMPT_TEMPLATE`: una regla nueva y el bloque de conversación, los dos guardados por
   `{% if history %}`. La regla tiene que ser explícita en que el historial **no es fuente
   de hechos**: scores, severidades y versiones salen sólo de `Context`.
4. `Valves` ([`:389`](../src/pipeline/pipeline_ciberseguridad.py)): `history_turns: int =
   HISTORY_TURNS` y `history_max_chars: int = HISTORY_MAX_CHARS`. Con `history_turns = 0`
   se vuelve al comportamiento de hoy desde la UI, sin redeploy.
5. `pipe()`: construir el historial **sólo si `kind == "user"`** —`_classify_request`
   ([`:215`](../src/pipeline/pipeline_ciberseguridad.py)) ya distingue los prompts internos
   de OpenWebUI— y pasarlo como `{"prompt_builder": {"question": user_message, "history":
   history}}`.
6. Log: una línea `[HISTORIAL] turnos=N | chars=M` con el `rid` que ya arma `_RequestLog`,
   para poder auditarlo por grep como el resto del archivo.

`history` queda **opcional** en el template: `PromptBuilder` detecta las variables del
template y no las exige, y un `Undefined` de Jinja es falsy, así que el `{% if history %}`
no se rendea. Los scripts de [`eval/`](../src/pipeline/eval/), que pasan sólo `question`,
siguen corriendo sin cambios y produciendo el mismo prompt.

## Cómo deriva en el prompt

Conversación en el chat de la extensión, sobre un hallazgo de `pillow 5.2.0`
(CVE-2023-4863, CWE-787):

```
[contexto]  (lo inyecta ChatPanel; con el cambio pendiente, deja de ser un mensaje)
usuario     ¿qué tan grave es esto para mi proyecto?
asistente   Es un desbordamiento de escritura (CWE-787) en el decodificador WebP...
            CVE-2023-4863 tiene CVSS 8.8. Se arregla en pillow 10.0.1.
usuario     ¿y me afecta si sólo abro PNG?
```

Lo que llega al `:9099` en el tercer turno:

```json
{ "model": "pipeline_ciberseguridad", "stream": false, "messages": [
  {"role":"user","content":"{\"vulnerability\":\"CVE-2023-4863\",\"package\":\"pillow\", ...}"},
  {"role":"user","content":"¿qué tan grave es esto para mi proyecto?"},
  {"role":"assistant","content":"Es un desbordamiento de escritura (CWE-787)... CVSS 8.8."},
  {"role":"user","content":"¿y me afecta si sólo abro PNG?"}
]}
```

El último elemento es el turno actual, o sea `user_message` duplicado: el normalizador lo
descarta por índice.

**Prompt de hoy** (el historial se pierde entero):

```
You are a cybersecurity assistant. Answer strictly from the provided context.
...(reglas, sin cambios)...

Context:
- [CVE-2023-4863] severity=HIGH CVSS=8.8 (related: CWE-787) affects: libwebp, chrome
  Heap buffer overflow in libwebp in Google Chrome prior to 116.0.5845.187...
- CWE-787: Out-of-bounds Write. The product writes data past the end...
- [CVE-2022-22817] severity=CRITICAL CVSS=9.8 (related: CWE-94) affects: pillow
  PIL.ImageMath.eval in Pillow before 9.0.0 allows evaluation of arbitrary expressions...

Question: ¿y me afecta si sólo abro PNG?
Answer:
```

El modelo no sabe qué es "esto", ni que el paquete es `pillow 5.2.0`, ni que se venía
hablando del decodificador WebP. Lo único que sobrevive del historial son los
identificadores: `resolve_vuln_ids` devuelve `["CVE-2023-4863", "CWE-787"]` con
`origen=historial`, y por eso `id_lookup` trae los docs correctos. **El retrieval acierta;
la redacción queda ciega.**

**Prompt con la feature** (lo marcado es lo nuevo; el bloque de contexto lo aporta el
cambio pendiente, no este plan):

```
You are a cybersecurity assistant. Answer strictly from the provided context.
...(reglas de siempre)...
- The finding under discussion and the conversation below resolve references         ← nuevo
  like "this vulnerability" or "that". They are NOT a source of facts: every
  claim about scores, severities or affected versions still has to come from
  the Context documents.

Finding under discussion:                                        ← lo aporta el cambio de contexto
  package: pillow 5.2.0 (fixed in 10.0.1)
  vulnerability: CVE-2023-4863   weaknesses: CWE-787
  summary: Heap buffer overflow in libwebp

Conversation so far:                                                                 ← nuevo
  user: ¿qué tan grave es esto para mi proyecto?
  assistant: Es un desbordamiento de escritura (CWE-787) en el decodificador
    WebP... CVSS 8.8. Se arregla en pillow 10.0.1.

Context:
- [CVE-2023-4863] severity=HIGH CVSS=8.8 (related: CWE-787) affects: libwebp, chrome
  ...
Question: ¿y me afecta si sólo abro PNG?
Answer:
```

**Desde OpenWebUI**, turno normal: exactamente igual, porque manda el mismo array. No hay
contexto de escaneo, así que ese bloque no se rendea y sólo aparece `Conversation so far`.

**Prompt interno de OpenWebUI** (título, follow-ups, tags), que es donde está el riesgo de
romper algo:

```
Question: ### Task:
Generate a concise, 3-5 word title...
<chat_history>
USER: ¿qué tan grave es esto para mi proyecto?
ASSISTANT: Es un desbordamiento de escritura...
</chat_history>
Answer:
```

`_classify_request` devuelve `meta:title`, así que `pipe()` no pasa `history` y el prompt
queda **byte a byte como hoy**: el `<chat_history>` que OpenWebUI ya trae embebido no se
duplica.

## Lo que NO se toca, y por qué importa

**`resolve_vuln_ids` sigue leyendo `messages` crudo, no el historial normalizado.** Es
tentador unificarlo para que "lo que el modelo ve" y "de dónde salen los identificadores"
sean lo mismo, pero **rompería el primer turno de la extensión mientras el contexto siga
siendo un mensaje**: hoy los IDs se heredan de ese JSON, que trae el `cve` cerca del
principio pero puede traer los CWE más adentro de `details`, y el truncado por
`HISTORY_MSG_MAX_CHARS` los perdería. Los dos caminos quedan separados a propósito.

**El retrieval no cambia.** `build_embedding_query`, `build_keyword_query` y
`build_ranker_query` se siguen armando sólo con `user_message`. La memoria se gana en la
generación, que es donde está el agujero; el retrieval ya se apoya en los IDs heredados.

## Pasos a seguir

1. **(Bloqueante, no es de este plan)** Mover el contexto del escaneo de `messages` al
   prompt como contexto propio.
2. Escribir `src/pipeline/chat/history.py` + `chat/__init__.py` con la API de arriba.
3. Escribir `tests/test_history.py` y correr `pytest` — sin stack, sin Docker.
4. **Revisar la semántica del normalizador contra los tests antes de tocar el pipeline.**
   Los pasos 2-3 tienen riesgo cero sobre lo que está corriendo; el 5 no.
5. Aplicar los 6 cambios de `pipeline_ciberseguridad.py`.
6. Verificar (abajo).
7. Actualizar este doc a **IMPLEMENTADO** con lo que se haya aprendido.

## Tests (`tests/test_history.py`)

- Turno actual duplicado al final.
- Turno actual duplicado **no** al final (hay un `system` después): el corte por índice lo
  tiene que resolver igual.
- `content` multimodal como lista de partes.
- Meta-prompt de OpenWebUI con `<chat_history>`: filtrado.
- Mensaje `system`: ignorado.
- Recorte por `max_turns`: queda lo más reciente.
- Recorte por `max_chars`: cae el más viejo.
- Un único mensaje gigante: se trunca, no devuelve vacío.
- `max_turns = 0`, `messages = []`, `messages = None`.
- Entrada con basura: elementos no-dict, `content: None`.

## Verificación

1. `pytest` — los tests nuevos y los tres suites que ya existen, sin stack.
2. `docker compose restart pipelines` y revisar `docker logs`: confirmar que el archivo
   **no** terminó en `pipelines/failed/` y que el modelo sigue listado en `/v1/models`. Es
   el chequeo que cubre el riesgo del `sys.path`.
3. Un chat de tres turnos desde la extensión y
   `grep '\[HISTORIAL\]' src/pipeline/logCiberseguridad.txt`. Comparar el
   `[TOKENS] prompt_chars` de antes y después para ver cuánto creció el prompt.
4. Un turno en OpenWebUI: confirmar que un `meta:title` sigue sin historial.
5. Hoy el prompt rendereado **no se loguea** (`_log_token_usage`
   [`:481`](../src/pipeline/pipeline_ciberseguridad.py) sólo mide `len`), así que para
   auditar el bloque a ojo hace falta una línea `[PROMPT]` temporal. Decidir en la review
   si queda detrás de un valve o se va.

## Riesgos

| Riesgo | Mitigación |
|---|---|
| El import del módulo hermano falla y el server manda el pipeline a `failed/` (404) | El `sys.path.insert` es idéntico al que ya funciona en `pipeline_dependencias.py`. Verificación 2, inmediatamente después del deploy. |
| El historial desplaza docs del prompt y empeora las respuestas | Topes por mensaje y totales; `history_turns = 0` revierte desde los valves sin redeploy. |
| El modelo toma un dato del historial como si fuera del corpus (un CVSS que él mismo dijo antes) | La regla explícita en el template. La extensión además ya marca los IDs citados que no venían del escaneo (`renderCitedIds` en [`chatPanel.ts`](../extension/src/chatPanel.ts)), así que esto es auditable a ojo. |
| Doble historial en los prompts internos de OpenWebUI | `kind == "user"` como guarda; verificación 4. |

## Fuera de alcance (ideas para después)

- **Historial en las queries de retrieval** (`history_in_query`): concatenar los últimos
  turnos de usuario a la query de embedding y de ranker. Barato, ayuda con follow-ups
  cortos, mete ruido. Medir en el harness antes de prenderlo.
- **Condensación con el LLM** ("reescribí la pregunta como pregunta autocontenida"): mejor
  recall en los follow-ups, pero es **una llamada extra a Ollama local en CPU por turno**, y
  la latencia ya es el cuello de botella (ver [`arquitectura.md`](arquitectura.md)).
- **Sesiones server-side** por `chat_id`: descartado arriba, queda anotado por si el
  contexto cambia.
