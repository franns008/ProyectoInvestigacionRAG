#!/usr/bin/env bash
# Diagnóstico del pipeline de explicaciones (docs/explicacion_hallazgos.md).
#
# Responde por qué la extensión no muestra explicaciones. El síntoma habitual es un
# 404 "Pipeline pipeline_dependencias not found": el servidor de Pipelines devuelve eso
# cuando el modelo NO está registrado, que puede pasar por cuatro motivos distintos.
# Este script los separa.
#
#   ./scripts/check_explicaciones.sh
#
# Puerto y clave se pueden pasar por entorno (defaults = los de docker-compose):
#   PIPELINES_PORT=9099 PIPESERVER_KEY=0p3n-w3bui ./scripts/check_explicaciones.sh

set -uo pipefail

PORT="${PIPELINES_PORT:-9099}"
KEY="${PIPESERVER_KEY:-0p3n-w3bui}"
MODEL="pipeline_dependencias"
BASE="http://localhost:${PORT}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_FILE="${REPO}/src/pipeline/${MODEL}.py"

ok()   { printf '  \033[32mOK\033[0m   %s\n' "$1"; }
bad()  { printf '  \033[31mMAL\033[0m  %s\n' "$1"; }
info() { printf '       %s\n' "$1"; }
paso() { printf '\n\033[1m%s\033[0m\n' "$1"; }

fallo=0

# ── 1. El archivo está donde tiene que estar ────────────────────────────────────
paso "1. El archivo del pipeline"
if [ -f "$PIPELINE_FILE" ]; then
    ok "src/pipeline/${MODEL}.py existe"
else
    bad "NO está src/pipeline/${MODEL}.py"
    if [ -f "${REPO}/src/pipeline/failed/${MODEL}.py" ]; then
        info "Está en src/pipeline/failed/: el servidor lo movió ahí porque falló al importar."
        info "Restauralo y volvé a correr esto:"
        info "    git checkout src/pipeline/ && rm -f src/pipeline/failed/${MODEL}.py"
    else
        info "Falta del repo. ¿Hiciste git pull en la rama feature/idQuery?"
    fi
    exit 1
fi

if grep -q "_PIPELINES_DIR" "$PIPELINE_FILE"; then
    ok "tiene la salvaguarda de sys.path (sin ella el servidor lo descarta)"
else
    bad "le falta la salvaguarda de sys.path — estás en una versión vieja del archivo"
    info "Actualizá: git pull"
    exit 1
fi

# ── 2. El container ve ESTE repo ───────────────────────────────────────────────
paso "2. El container de pipelines"
if ! docker compose -f "${REPO}/infrastructure/docker-compose.yml" ps pipelines 2>/dev/null | grep -q "Up"; then
    bad "el container 'pipelines' no está corriendo"
    info "Levantalo:  cd infrastructure && docker compose up -d"
    exit 1
fi
ok "el container está arriba"

CID="$(docker compose -f "${REPO}/infrastructure/docker-compose.yml" ps -q pipelines 2>/dev/null)"
if docker exec "$CID" test -f "/app/pipelines/${MODEL}.py" 2>/dev/null; then
    ok "el container ve el archivo en /app/pipelines/"
else
    bad "el container NO ve /app/pipelines/${MODEL}.py"
    info "Su volumen apunta a otro directorio. Contenido real de /app/pipelines:"
    docker exec "$CID" ls /app/pipelines 2>/dev/null | sed 's/^/       /'
    info "El compose bindea ../src/pipeline: corré docker compose DESDE infrastructure/ de ESTE repo."
    exit 1
fi

# ── 3. ¿Está registrado el modelo? ─────────────────────────────────────────────
paso "3. El modelo registrado en el servidor"
modelos() { curl -s -m 10 -H "Authorization: Bearer ${KEY}" "${BASE}/v1/models"; }

RESP="$(modelos)"
if [ -z "$RESP" ]; then
    bad "el servidor no responde en ${BASE}"
    info "¿Es el puerto correcto? PIPELINES_PORT=<puerto> ./scripts/check_explicaciones.sh"
    exit 1
fi

tiene_modelo() { printf '%s' "$1" | grep -q "\"${MODEL}\""; }

if tiene_modelo "$RESP"; then
    ok "${MODEL} está registrado"
else
    bad "${MODEL} NO está registrado (por eso el 404)"
    info "Modelos que sí conoce:"
    printf '%s' "$RESP" | python3 -c 'import json,sys; [print("        -", m["id"]) for m in json.load(sys.stdin).get("data",[])]' 2>/dev/null \
        || info "        (no pude leer la lista)"

    info ""
    info "Los pipelines se cargan SOLO al arrancar. Recargando..."
    curl -s -m 60 -X POST -H "Authorization: Bearer ${KEY}" "${BASE}/v1/pipelines/reload" >/dev/null
    RESP="$(modelos)"

    if tiene_modelo "$RESP"; then
        ok "ahora sí quedó registrado (era eso: el servidor no lo había cargado)"
    else
        bad "sigue sin registrarse: el módulo falla al importar"
        if [ -f "${REPO}/src/pipeline/failed/${MODEL}.py" ]; then
            info "Confirmado: el servidor lo movió de nuevo a src/pipeline/failed/"
        fi
        info "El motivo está en el log del servidor:"
        docker compose -f "${REPO}/infrastructure/docker-compose.yml" logs --tail 200 pipelines 2>/dev/null \
            | grep -A 6 "Error loading module" | tail -20 | sed 's/^/       /'
        exit 1
    fi
fi

# ── 4. Una explicación de verdad ───────────────────────────────────────────────
paso "4. Una explicación real, punta a punta"
PAYLOAD='{"model":"'"${MODEL}"'","stream":false,"messages":[{"role":"user","content":"{\"findings\":[{\"package\":\"pillow\",\"installed_version\":\"5.2.0\",\"cve\":\"CVE-2023-4863\",\"osv_ids\":[\"GHSA-j7hp-h8jx-5ppr\"],\"cwe_ids\":[\"CWE-787\"],\"summary\":\"libwebp: OOB write in BuildHuffmanTable\",\"details\":\"Heap buffer overflow in libwebp allows a remote attacker to perform an out of bounds memory write via a crafted HTML page.\"}]}"}]}'

# Se separa el cuerpo del codigo HTTP y del codigo de salida de curl: sin eso, un
# timeout (el LLM tardando) y un error del servidor se ven exactamente igual -un cuerpo
# vacio- y se busca el problema en el lugar equivocado.
RAW="$(curl -s -m 180 -w '\n%{http_code}' -X POST "${BASE}/v1/chat/completions" \
        -H "Authorization: Bearer ${KEY}" -H "Content-Type: application/json" \
        -d "$PAYLOAD")"
CURL_RC=$?
HTTP_CODE="$(printf '%s' "$RAW" | tail -n1)"
OUT="$(printf '%s' "$RAW" | sed '$d')"

logs() {
    docker compose -f "${REPO}/infrastructure/docker-compose.yml" logs --tail "${1:-40}" pipelines 2>/dev/null
}

if [ $CURL_RC -eq 28 ]; then
    bad "el servidor no respondio en 180 s: la generacion se esta colgando"
    info "El lookup es instantaneo; lo que tarda es el LLM. Casos tipicos:"
    info "  - LLM_PROVIDER=groq sin GROQ_API_KEY valida -> 5 reintentos de 60 s cada uno"
    info "  - LLM_PROVIDER=ollama con un modelo que todavia se esta descargando"
    info ""
    info "Que alcanzo a hacer el pipeline:"
    logs 60 | grep -E "DepsExplain|explicando|Error|Traceback" | tail -10 | sed 's/^/       /'
    info ""
    info "Proveedor configurado (esto NO muestra la key, solo si esta definida):"
    info "    grep LLM_PROVIDER infrastructure/.env"
    info "    grep -c '^GROQ_API_KEY=.\\+' infrastructure/.env   # 1 = tiene valor, 0 = vacia"
    exit 1
elif [ $CURL_RC -ne 0 ]; then
    bad "curl fallo (codigo $CURL_RC) contra ${BASE}"
    exit 1
fi

if [ -z "$OUT" ]; then
    bad "el servidor respondio HTTP ${HTTP_CODE} con el cuerpo vacio"
    info "Ultimas lineas del log del servidor:"
    logs 40 | tail -15 | sed 's/^/       /'
    exit 1
fi

info "HTTP ${HTTP_CODE}"

# El cuerpo se pasa por ARCHIVO, no por pipe: `python3 - <<EOF` toma el PROGRAMA de
# stdin, así que un `printf ... | python3 - <<EOF` deja a sys.stdin.read() devolviendo
# vacío y la respuesta se pierde en silencio.
BODY_FILE="$(mktemp)"
trap 'rm -f "$BODY_FILE"' EXIT
printf '%s' "$OUT" > "$BODY_FILE"

python3 - "$BODY_FILE" <<'PYEOF'
import json, sys

OK, MAL, AVISO = "\033[32mOK\033[0m", "\033[31mMAL\033[0m", "\033[33mSIN EXPLICACION\033[0m"
raw = open(sys.argv[1], encoding="utf-8").read()

try:
    body = json.loads(raw)
except json.JSONDecodeError:
    print(f"  {MAL}  el servidor devolvio algo que no es JSON:")
    print("      ", raw[:400] if raw.strip() else "(cuerpo vacio)")
    sys.exit(1)

if "detail" in body:
    print(f"  {MAL}  el servidor rechazo la llamada:", body["detail"])
    sys.exit(1)

content = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
try:
    payload = json.loads(content)
except json.JSONDecodeError:
    print(f"  {MAL}  el pipeline no devolvio JSON:")
    print("      ", content[:400] if content.strip() else "(contenido vacio)")
    sys.exit(1)

if payload.get("error"):
    print(f"  {MAL} ", payload["error"])
    sys.exit(1)

explicaciones = payload.get("explanations", [])
if not explicaciones:
    print(f"  {AVISO}  el pipeline respondio bien pero no explico nada.")
    print("       O el LLM no contesto (revisa LLM_PROVIDER y el modelo en ollama),")
    print("       o el modelo respondio que el contexto no alcanzaba.")
    sys.exit(1)

for e in explicaciones:
    print(f"  {OK}   {e['identifier']}")
    print(f"       {e['explanation']}")
    print(f"       fuentes usadas: {', '.join(e.get('kinds', [])) or '-'}")
    print(f"       citas: {', '.join(e.get('citations', [])) or '-'}")
PYEOF
estado=$?

printf '\n'
if [ $estado -eq 0 ]; then
    printf '\033[32mTodo bien.\033[0m Si la extension igual no muestra nada, es del lado de VSCode:\n'
    printf '  cd extension && npm install && npm run compile   (out/ NO viaja por git)\n'
    printf '  y recarga la ventana (Developer: Reload Window).\n'
fi
exit $estado
