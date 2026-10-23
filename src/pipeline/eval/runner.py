"""Piezas comunes a los runners del eval: dataset, log de la corrida, run_id y metadata de git.

No se llama `common` a propósito: los runners corren como script, con `eval/` primero en
`sys.path`, y taparía al paquete `src/pipeline/common`.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

import csv_store


def load_dataset(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("dataset.yaml debe ser una lista de preguntas")
    return data


class Tee:
    """Duplica lo escrito a stdout hacia un archivo, para persistir el log de la corrida."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)

    def flush(self):
        for st in self._streams:
            st.flush()


def make_run_id(runs_csv: Path) -> str:
    """Timestamp UTC compacto, único dentro de runs.csv (sufija `-2`, `-3`… si colisiona).

    El orden lexicográfico sigue siendo cronológico: `report.py` depende de eso.
    """
    base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    usados = set(csv_store.load_runs(runs_csv)["run_id"].dropna())
    if base not in usados:
        return base
    n = 2
    while f"{base}-{n}" in usados:
        n += 1
    return f"{base}-{n}"


def git_metadata() -> dict:
    """Metadata que inyectan `scripts/eval*.sh` como variables de entorno (el container no
    tiene el `.git`). Sin las variables, queda vacía."""
    return {
        "git_commit": os.getenv("GIT_COMMIT", ""),
        "git_branch": os.getenv("GIT_BRANCH", ""),
        "git_dirty":  os.getenv("GIT_DIRTY", ""),
    }
