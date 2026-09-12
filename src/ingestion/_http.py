"""Helpers de descarga compartidos por los fetchers de `src/ingestion/`.

Las descargas son *condicionales*: se manda `If-Modified-Since` y si el servidor
responde 304 no se baja nada. La marca de tiempo no vive en ningún archivo de estado
aparte — es el `mtime` del archivo ya descargado, al que se le fija el `Last-Modified`
que mandó el servidor. Es lo que hacen `curl -z` y `wget -N`: el estado no se puede
desincronizar del archivo que describe.

Tres protecciones, todas con motivo: reintentos con backoff, escritura atómica (una
descarga cortada no debe dejar un archivo a medias que el escáner intente leer) y
validación de tamaño mínimo, porque más de una de estas fuentes responde 200 con el
cuerpo truncado en vez de fallar.
"""

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import formatdate, parsedate_to_datetime
from pathlib import Path

import requests

MAX_RETRIES = 3
TIMEOUT_SECONDS = 30
# 304 es una respuesta *buena*: significa "no cambió". Si no estuviera acá, una descarga
# condicional se reintentaría tres veces y terminaría en un error sin sentido.
OK_STATUSES = (200, 304)

REPO_ROOT = Path(__file__).resolve().parents[2]
# docker-compose montea este directorio como el volumen `rawdata` -> /app/pipelines/rawdata,
# que es el INPUT_DIR del indexer. Es también el --data que espera deps/cli.py.
RAW_DIR = REPO_ROOT / "data" / "raw"

NUEVO = "nuevo"
ACTUALIZADO = "actualizado"
SIN_CAMBIOS = "sin cambios"


@dataclass
class Result:
    """Lo que devuelve cada fetcher, para que el orquestador pueda resumir el estado."""
    status: str
    detail: str


def human_size(num_bytes: int) -> str:
    if num_bytes >= 1_000_000:
        return f"{num_bytes / 1_000_000:.1f} MB"
    return f"{num_bytes / 1_000:.0f} kB"


def fecha_utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d")


def get_with_retries(url: str, source: str, **kwargs) -> requests.Response:
    """GET con reintentos y backoff exponencial. `kwargs` van tal cual a requests.get."""
    for attempt in range(1, MAX_RETRIES + 1):
        response = requests.get(url, timeout=TIMEOUT_SECONDS, **kwargs)
        if response.status_code in OK_STATUSES:
            return response
        if attempt == MAX_RETRIES:
            # Levanta para >= 400; para un 2xx/3xx raro hay que ser explícito, o el
            # error termina siendo imposible de diagnosticar.
            response.raise_for_status()
            raise RuntimeError(f"{url} respondió {response.status_code}, que no se esperaba")
        wait = 2 ** attempt
        print(f"[{source}] status {response.status_code}, reintentando en {wait}s "
              f"(intento {attempt}/{MAX_RETRIES})")
        time.sleep(wait)
    raise RuntimeError("unreachable")


def conditional_get(url: str, source: str, reference: Path, force: bool = False) -> requests.Response:
    """GET condicional contra el `mtime` de `reference`. Un 304 significa 'no cambió'."""
    headers = {}
    if reference.exists() and not force:
        headers["If-Modified-Since"] = formatdate(reference.stat().st_mtime, usegmt=True)
    print(f"[{source}] consultando {url}")
    return get_with_retries(url, source, headers=headers)


def checked_payload(response: requests.Response, min_bytes: int) -> bytes:
    payload = response.content
    if len(payload) < min_bytes:
        raise RuntimeError(
            f"descarga sospechosamente chica ({len(payload)} bytes, se esperaban "
            f">{min_bytes}): probablemente venga truncada"
        )
    return payload


def write_atomic(dest: Path, payload: bytes, response: requests.Response | None = None) -> None:
    """Escribe a un temporal y recién ahí renombra, para no dejar archivos a medias.

    Con `response`, le copia al archivo el `Last-Modified` del servidor: ese mtime es lo
    que la próxima corrida manda como `If-Modified-Since`.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(dest)

    header = response.headers.get("Last-Modified") if response is not None else None
    if header:
        try:
            stamp = parsedate_to_datetime(header).timestamp()
        except (TypeError, ValueError):
            return  # sin fecha utilizable: queda el mtime de ahora, que también sirve
        os.utime(dest, (stamp, stamp))


def download(url: str, dest: Path, min_bytes: int, source: str, force: bool = False) -> Result:
    """Baja `url` a `dest` sólo si cambió en el servidor."""
    existia = dest.exists()
    response = conditional_get(url, source, dest, force=force)

    if response.status_code == 304:
        stat = dest.stat()
        detail = f"{human_size(stat.st_size)}, del {fecha_utc(stat.st_mtime)}"
        print(f"[{source}] sin cambios ({detail})")
        return Result(SIN_CAMBIOS, detail)

    payload = checked_payload(response, min_bytes)
    write_atomic(dest, payload, response)

    status = ACTUALIZADO if existia else NUEVO
    detail = f"{human_size(len(payload))} -> {dest.relative_to(REPO_ROOT)}"
    print(f"[{source}] {status}: {detail}")
    return Result(status, detail)
