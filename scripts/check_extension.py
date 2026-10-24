"""Prueba contra el stack lo que la extensión de VSCode necesita, en el orden en que lo usa.

    .venv/bin/python scripts/check_extension.py [--manifest requirements.txt]

Usar el mismo intérprete que `cibersec.pythonPath`: el escaneo necesita `packaging`. Puerto
y clave salen de PIPELINES_PORT y PIPESERVER_KEY (defaults del compose). Corta en el primer
paso que falla.
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src/pipeline"))

from common.text import finding_identifier  # noqa: E402
CHAT = "pipeline_ciberseguridad"
EXPLAIN = "pipeline_dependencias"
COMPOSE = "cd infrastructure && docker compose"


def ok(text: str) -> None:
    print(f"  \033[32mOK\033[0m   {text}")


def fail(text: str, *hints: str) -> None:
    print(f"  \033[31mMAL\033[0m  {text}")
    for hint in hints:
        print(f"       {hint}")
    sys.exit(1)


def step(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def call(base: str, key: str, path: str, body: dict | None = None, timeout: float = 10) -> dict:
    request = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        fail(f"{path} respondió {error.code}: {error.read().decode(errors='replace')[:300]}")
    except (urllib.error.URLError, TimeoutError) as error:
        fail(f"{path} no respondió ({getattr(error, 'reason', error)})", f"¿Está arriba el stack? {COMPOSE} ps")


def completion(base: str, key: str, model: str, messages: list[dict], timeout: float) -> str:
    body = {"model": model, "stream": False, "messages": messages}
    response = call(base, key, "/v1/chat/completions", body, timeout)
    return response["choices"][0]["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=REPO / "tests/test_requirements/requirements_3vuln.txt")
    parser.add_argument("--timeout", type=float, default=300, help="segundos por llamada al LLM")
    args = parser.parse_args()
    base = f"http://localhost:{os.getenv('PIPELINES_PORT', '9099')}"
    key = os.getenv("PIPESERVER_KEY", "0p3n-w3bui")

    step("1. Servidor y modelos")
    models = {m["id"] for m in call(base, key, "/v1/models").get("data", [])}
    missing = {CHAT, EXPLAIN} - models
    if missing:
        fail(
            f"no están registrados: {', '.join(sorted(missing))}",
            "Los pipelines se cargan al arrancar; el motivo está en el log:",
            f"{COMPOSE} logs pipelines | grep -A 6 'Error loading module'",
        )
    ok(f"{base} responde con {CHAT} y {EXPLAIN}")

    step("2. Valves del medidor de contexto")
    valves = call(base, key, f"/{CHAT}/valves")
    if not valves.get("effective_num_ctx"):
        fail("los valves no traen effective_num_ctx: el medidor usaría los defaults de la extensión")
    ok(f"llm_model={valves.get('llm_model')} effective_num_ctx={valves['effective_num_ctx']} "
       f"max_tokens={valves.get('max_tokens')}")

    step("3. Escaneo local")
    scan = subprocess.run(
        [sys.executable, "-m", "deps.cli", str(args.manifest), "--data", str(REPO / "data/raw"), "--json"],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src/pipeline")},
    )
    if scan.returncode != 0:
        fail("deps.cli falló", *scan.stderr.strip().splitlines()[-3:])
    findings = json.loads(scan.stdout)["findings"]
    if not findings:
        fail(f"{args.manifest.name} no dio hallazgos", "Probá con un manifiesto con versiones vulnerables.")
    ok(f"{len(findings)} hallazgos en {args.manifest.name}")

    step("4. Explicaciones")
    sent = findings[:3]
    content = completion(base, key, EXPLAIN, [{"role": "user", "content": json.dumps({"findings": sent})}], args.timeout)
    result = json.loads(content)
    if result.get("error"):
        fail(result["error"])
    explanations = result.get("explanations") or []
    if not explanations:
        fail("el pipeline no explicó ningún hallazgo", f"Mirá el log: {COMPOSE} logs pipelines | grep DepsExplain")
    sent_ids = {finding_identifier(f) for f in sent}
    for explanation in explanations:
        if explanation["identifier"] not in sent_ids:
            fail(f"volvió una explicación para {explanation['identifier']}, que no se mandó")
        ok(explanation["identifier"])
        print(f"       {explanation['explanation']}")
        print(f"       citas: {', '.join(explanation.get('citations', [])) or '-'}")

    step("5. Chat con un hallazgo adjunto")
    messages = [
        {"role": "system", "content": json.dumps({"findings": findings[:1]})},
        {"role": "user", "content": "¿Qué tan grave es y cómo lo mitigo?"},
    ]
    answer = completion(base, key, CHAT, messages, args.timeout).strip()
    if not answer:
        fail("el chat respondió vacío", f"Mirá el log: {COMPOSE} exec pipelines tail -20 /app/pipelines/logCiberseguridad.txt")
    ok(answer[:300] + ("…" if len(answer) > 300 else ""))

    print("\n\033[32mTodo bien.\033[0m")


if __name__ == "__main__":
    main()
