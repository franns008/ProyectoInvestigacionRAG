# Chat de la extensión: próximos pasos

> **Propuesto, no implementado.** El chat actual (hallazgos adjuntos, historial, medidor de
> contexto) está descrito en [`arquitectura.md`](arquitectura.md) §4.4, junto con el
> pendiente de que el pipeline informe sus límites y el uso real de tokens, del que dependen
> estos pasos.

1. **Que el LLM cargue las vulnerabilidades a demanda.** Hoy cada pregunta lleva todos los
   hallazgos adjuntos con su advisory, pregunte o no por ellos, y el tope fijo de 6 es lo
   que evita que se coman el lugar de los documentos. La idea es mandar al prompt sólo un
   índice liviano de los adjuntos, con identificador, paquete y scores, y que el advisory
   completo se cargue cuando hace falta:
   - la opción simple, determinística, es elegir los hallazgos que la pregunta nombra (por
     ID o paquete), o todos si no nombra ninguno;
   - la otra opción es tool calling: el modelo pide `get_finding(id)`. Depende de que el
     modelo local lo soporte bien, y en modelos chicos no es confiable.

   Así el tope de adjuntos deja de estar atado al tamaño del prompt.
2. **Que el contexto se arme según el modelo.** En lugar de `num_ctx` / `max_tokens` fijos:
   - leer de Ollama (`/api/show` → `<arch>.context_length`, p. ej. 32768 en
     `qwen2.5:3b-instruct`) la ventana del modelo activo (`LLM_MODEL` o valve `llm_model`);
   - usar `min(context_length, valve num_ctx_max)`: en CPU la ventana cuesta RAM y latencia;
   - derivar `max_tokens` de esa ventana y recortarlo por request a lo que sobre tras el prompt
     (`generation_kwargs` en `llm.run`);
   - repartir el resto con prioridad: pregunta y reglas > hallazgos > documentos
     (`ranker_top_k` hasta 2) > historial viejo. Un adjunto nunca se descarta en silencio.

   La lógica iría pura en `src/pipeline/chat/budget.py`, con tests de pytest. El resultado,
   límites y `max_attachments`, lo publica la consulta de info del pendiente de §4.4 en `arquitectura.md`, y la
   extensión reemplaza `MAX_ATTACHMENTS` y las constantes copiadas de `contextMeter.ts`.
