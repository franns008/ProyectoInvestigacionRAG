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

OUT="$(curl -s -m 120 -X POST "${BASE}/v1/chat/completions" \
        -H "Authorization: Bearer ${KEY}" -H "Content-Type: application/json" \
        -d "$PAYLOAD")"

printf '%s' "$OUT" | python3 - <<'PY'
import json, sys
raw = sys.stdin.read()
try:
    body = json.loads(raw)
except json.JSONDecodeError:
    print("  \033[31mMAL\033[0m  respuesta ilegible del servidor:"); print("      ", raw[:400]); sys.exit(1)

if "detail" in body:
    print("  \033[31mMAL\033[0m  el servidor rechazó la llamada:", body["detail"]); sys.exit(1)

content = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
try:
    payload = json.loads(content)
except json.JSONDecodeError:
    print("  \033[31mMAL\033[0m  el pipeline no devolvió JSON:"); print("      ", content[:400]); sys.exit(1)

if payload.get("error"):
    print("  \033[31mMAL\033[0m ", payload["error"]); sys.exit(1)

explicaciones = payload.get("explanations", [])
if not explicaciones:
    print("  \033[33mSIN EXPLICACIÓN\033[0m  el pipeline respondió bien pero no explicó nada.")
    print("       El LLM no contestó (revisá LLM_PROVIDER y GROQ_API_KEY en infrastructure/.env)")
    print("       o el modelo dijo que el contexto no alcanzaba.")
    sys.exit(1)

for e in explicaciones:
    print(f"  \033[32mOK\033[0m   {e['identifier']}")
    print(f"       {e['explanation']}")
    print(f"       fuentes usadas: {', '.join(e.get('kinds', [])) or '—'}")
    print(f"       citas: {', '.join(e.get('citations', [])) or '—'}")
PY
estado=$?

printf '\n'
if [ $estado -eq 0 ]; then
    printf '\033[32mTodo bien.\033[0m Si la extensión igual no muestra nada, es del lado de VSCode:\n'
    printf '  cd extension && npm install && npm run compile   (out/ NO viaja por git)\n'
    printf '  y recargá la ventana (Developer: Reload Window).\n'
fi
exit $estado
