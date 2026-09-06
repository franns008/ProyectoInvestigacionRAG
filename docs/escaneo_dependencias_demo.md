# Guion de la demo — escaneo de dependencias

Qué mostrar y en qué orden. Para hacerlo andar, ver
[`escaneo_dependencias_setup.md`](escaneo_dependencias_setup.md); para el porqué de cada
decisión, [`escaneo_dependencias.md`](escaneo_dependencias.md).

Dura **3 a 5 minutos**. No necesita Docker, ni pgvector, ni LLM, ni red.

## Antes de empezar

```bash
git checkout feature/requirements
./scripts/fetch_deps_data.sh          # EPSS y KEV cambian a diario: rebajalos hoy
.venv/bin/python -m pytest            # 54 en <1 s: confirma que todo está sano
```

Después, F5 en la carpeta `extension/` y dejá la ventana de prueba con
`requirements_demo.txt` abierto, **sin escanear todavía**.

Una sola pantalla partida: VSCode a la izquierda (~60%), el panel de resultados a la
derecha. No dos monitores — la relación entre la acción y su resultado tiene que verse de
un vistazo.

> **Rebajá los datos el mismo día.** EPSS se actualiza a diario y el catálogo de CISA
> también. Los números de abajo son los del 28/08/2026 y van a variar un poco; el que no
> debería moverse es el 1 de KEV.

## El guion

### 1. El problema (30 s)

Mostrá `requirements_demo.txt` abierto. Cinco librerías que todos conocen, fijadas en
versiones de 2018-2019.

> "Este es un `requirements.txt` cualquiera. Cinco dependencias, versiones de hace unos
> años. La pregunta del desarrollador es simple: ¿esto es seguro?"

### 2. El escaneo (15 s)

Clic en **🛡 Escanear dependencias**. Tarda ~1 segundo.

> "Un segundo. Sin conectarse a ningún servicio, sin mandar el código a ningún lado."

Ese paréntesis importa: es el argumento de confidencialidad, y conviene decirlo mientras
sucede en vez de defenderlo después.

### 3. El número que no sirve (30 s)

Señalá el total.

> "128 vulnerabilidades. Esto es lo que te da cualquier escáner, y es inútil: nadie
> arregla 128 cosas."

### 4. El embudo — **este es el momento** (60 s)

Bajá por la tabla, línea por línea.

```
CVEs que afectan tus versiones                128
  filtrando por CVSS >= 7.0                    66   (52%)
  filtrando por EPSS >= 10%                    11   (9%)
  presentes en CISA KEV (explotadas hoy)        1   (1%)
```

> "Si filtro por severidad alta —que es lo que hace la mayoría de las herramientas— me
> quedan 66 de 128. **La mitad. No filtró nada.**
>
> Pero 'grave' y 'te está pasando' son preguntas distintas. Si ordeno por probabilidad
> real de explotación quedan 11. Y si me quedo con las que CISA confirma que se están
> explotando **hoy**: una."

Dejá un silencio acá. Es el argumento central de todo el trabajo.

### 5. La respuesta accionable (45 s)

Señalá la primera tarjeta.

> "Empezá por esta: **Pillow 5.2.0, actualizá a 10.0.1**. Es un desbordamiento de memoria
> en la librería que procesa imágenes WebP. Probabilidad de explotación: 100%. Está en el
> catálogo de vulnerabilidades explotadas de CISA.
>
> **Y las otras 127 no tienen exploit conocido circulando.**"

Esa última frase es la que más vale. Decir que 127 no urgen es tan útil como señalar la
que sí — es la diferencia entre un escáner y un consultor.

### 6. Dónde entra el RAG (45 s)

Sé explícito sobre lo que hay y lo que falta. La honestidad acá suma más que estirar.

> "Esto que ven es determinístico: no hay inteligencia artificial, son cruces de datos.
> Y es a propósito, porque tiene que ser exacto.
>
> Lo que sigue es que el sistema **explique** este hallazgo: qué es un desbordamiento de
> memoria, por qué importa, qué clase de debilidad es —acá dice CWE-787— con cada
> afirmación citada contra el catálogo de MITRE que ya tenemos indexado. Eso es lo que un
> ranking no puede hacer y un RAG sí.
>
> Y al revés: sin esta priorización, el modelo escribiría 128 párrafos que nadie lee. Se
> necesitan mutuamente."

### 7. Por qué las CVE y no el código (30 s, opcional)

Si hay tiempo, o si preguntan por qué no analizan código fuente:

> "Un modelo de lenguaje reconoce una inyección SQL sin ayuda: eso está en sus datos de
> entrenamiento hace décadas. Pero **CVE-2023-4863 se publicó después** de que estos
> modelos se entrenaran, y no vive en la lógica del código: vive en la versión de una
> librería. Ahí el conocimiento externo no es un adorno, es la única forma de saberlo."

## Preguntas previsibles

| Pregunta | Respuesta |
|---|---|
| *"¿Esto no lo hace Dependabot?"* | Dependabot lista. Esto prioriza por explotabilidad real y va a explicar con fuentes citadas. Y corre local, sin mandar el manifiesto a un tercero. |
| *"¿No alcanza con darle búsqueda web al modelo?"* | Corpus auditable y acotado a fuentes autoritativas, todo local por confidencialidad, y una cita verificable en cada afirmación. |
| *"Si el embudo es determinístico, ¿para qué el RAG?"* | El ranking dice qué arreglar primero; no dice por qué ni lo puede probar. Sin priorización el RAG escribiría 128 párrafos que nadie lee; sin RAG queda una tabla ordenada, que es lo que `pip-audit` ya da. Se necesitan mutuamente. |
| *"Un modelo ya sabe qué es un buffer overflow"* | Cierto — los CWE tienen décadas y están en el entrenamiento. Por eso la explicación no se apoya ahí, sino en la **cita verificable**, en las mitigaciones concretas del catálogo de MITRE y en el detalle del advisory (que el fallo esté en `libwebp` embebida en Pillow es post-cutoff). Detalle en [escaneo_dependencias.md](escaneo_dependencias.md#el-punto-débil-y-cómo-se-responde). |
| *"¿Y si le preguntan algo que no sabe?"* | Está diseñado para abstenerse, y está medido (columna `correct_rejection` del eval). Conviene **anunciarlo antes** de que pase: así ese caso confirma el diseño en vez de parecer una falla. |
| *"¿Por qué no analizan el código fuente?"* | Ver el punto 7. |
| *"¿Cubre otros lenguajes?"* | OSV cubre npm, Go, crates.io y más. El alcance de esta etapa es Python; extenderlo es cambiar el ecosistema del lookup. |
| *"¿Sabe si mi código llega a usar la parte vulnerable?"* | No. El análisis de alcanzabilidad es bastante más difícil y es lo que venden las herramientas comerciales. Está fuera de alcance — **conviene nombrarlo antes de que lo pregunten**. |

## Lo que NO hay que hacer

- **No mostrar internals en la extensión.** Nada de scores de retriever, etapas del
  pipeline ni métricas de evaluación. Eso va en el panel de evidencia, que es la otra
  pantalla. Si se filtran a la herramienta, el conjunto se lee como proyecto de facultad.
- **No abrir el `requirements.txt` del repo.** No tiene pines `==` y da cero hallazgos
  —correctamente— pero en vivo parece que está roto.
- **No prometer la explicación como si existiera.** Hoy el texto de cada tarjeta es el
  resumen que el propio advisory trae escrito. Decir "esto lo redacta el modelo" sería
  falso y es innecesario: el embudo ya se sostiene solo.
- **No escanear un archivo sin ninguna vulnerabilidad de KEV.** Sin esa línea en 1, el
  remate desaparece y la demo pierde su argumento.

## Si algo falla en vivo

| Síntoma | Qué hacer |
|---|---|
| El botón no aparece | El archivo tiene que llamarse `requirements*.txt`. Alternativa: `Ctrl+Shift+P` → "Escanear". |
| "No encontré el dump de OSV" | Falta `./scripts/fetch_deps_data.sh`. |
| Cero hallazgos | Estás sobre un manifiesto sin pines `==`. Abrí `requirements_demo.txt`. |
| La extensión no responde | Plan B: la misma salida en la terminal, que no depende de VSCode. Vale la pena tenerla ya escrita en una pestaña: `PYTHONPATH=src/pipeline .venv/bin/python -m deps.cli requirements_demo.txt --data data/raw --top 3` |
