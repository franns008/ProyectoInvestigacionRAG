"""Persistencia de resultados del eval en CSV.

Único lugar del harness que sabe leer/escribir las tablas de resultados. Nadie
más hace `csv.writer` ni `pd.read_csv` a mano: si hace falta una columna nueva,
se agrega acá y la migración de esquema se encarga del histórico.

Dos tablas acumulativas (ver docs/eval/refactor_resultados_csv.md):

  results/runs.csv       1 fila por corrida     (versionada en git)
  results/questions.csv  1 fila por corrida × pregunta  (gitignored)

Escribir no depende de pandas (usa el módulo `csv` de la stdlib); pandas sólo
se importa, de forma lazy, en las funciones de lectura.

Notas asumidas:

- Si un texto contiene la secuencia literal `\\n` (backslash + n tipeado a mano),
  al des-escaparlo es indistinguible de un salto de línea real. Aceptado: el
  texto es para lectura humana, no para round-trip exacto.
- **Sin locking**: se asume un solo runner a la vez (el harness se usa a mano;
  `run_eval.py` y `run_eval_llm.py` no se corren en simultáneo).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # sólo para type hints; pandas no se importa en runtime al escribir
    import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Esquemas
#
# Orden de las columnas: claves → config/dimensiones → métricas → texto.
# Una columna nueva se inserta en SU grupo, nunca "al final de todo" (salvo que
# sea texto). El orden es el del archivo en disco, así que respetarlo mantiene
# los CSV legibles a ojo.
# ─────────────────────────────────────────────────────────────────────────────

RUNS_COLUMNS = [
    # claves e identidad
    "run_id", "timestamp_utc", "suite", "epoch", "label",
    "git_commit", "git_branch", "git_dirty",
    # config
    "llm_provider", "llm_model", "judge_model", "temperature",
    "embedding_model", "ranker_model", "retriever_top_k", "ranker_top_k",
    "chunking",                       # estrategia de chunking (hoy "production")
    "n_docs_store", "dataset_n", "n_errors", "duration_s",
    # Tier 1 (calculadas sobre la salida del ranker → sufijo _eff, "efectivo")
    "n_gt", "recall_eff", "hit_eff", "mrr_eff",
    # Tier 1b — retrieval a nivel de FUENTE (agnóstico al chunking; ver metrics.py)
    "n_source", "source_recall", "source_hit", "source_mrr",
    # Tier 2
    "n_sas", "sas_mean", "sas_std",
    # Tier 3 (suite=judge)
    "n_judge", "faithfulness", "context_relevance",
]

QUESTIONS_COLUMNS = [
    # claves
    "run_id", "question_id", "category", "status", "error",
    # Tier 1
    "n_expected", "expected_ids", "retrieved_ids", "joined_ids", "rank_first_hit",
    "recall_eff", "hit_eff", "rr_eff", "via_embedding", "via_keyword",
    # Tier 1b (fuente)
    "expected_sources", "retrieved_sources",
    "source_recall", "source_hit", "source_rr",
    # abstención (negativas; ver check_correct_rejection en run_eval.py)
    "correct_rejection",
    # Tier 2 / Tier 3
    "sas", "faithfulness", "context_relevance",
    # texto (siempre al final)
    "question", "answer", "reference_answer",
]

# Renombres históricos de columnas: la migración de esquema los consulta antes
# de descartar datos. {nombre_viejo: nombre_nuevo}. Arranca vacío.
RENAMES: dict[str, str] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Serialización
# ─────────────────────────────────────────────────────────────────────────────

def _ser(value: Any) -> str:
    """Convierte un valor Python a su representación en el CSV.

    El objetivo es que **cada fila del CSV sea una línea física**, para que
    `grep`/`wc -l` sigan funcionando sobre el archivo. Por eso los saltos de
    línea del texto se escapan a un `\\n` literal (backslash + n).

    El quoting de comas y comillas lo maneja `csv.writer` (QUOTE_MINIMAL).
    """
    if value is None:
        return ""
    if isinstance(value, bool):          # antes que int: bool es subclase de int
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return "|".join(str(v) for v in value)
    if isinstance(value, float):
        return str(round(value, 4))
    if isinstance(value, int):
        return str(value)
    text = str(value)
    return text.replace("\r\n", "\n").replace("\n", "\\n")


def unescape_text(s: str) -> str:
    """Inversa de `_ser` para las columnas de texto.

    La usa `report.py` cada vez que muestra texto completo (pregunta, respuesta,
    referencia). Nadie des-escapa a mano.
    """
    if s is None:
        return ""
    return str(s).replace("\\n", "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Escritura
# ─────────────────────────────────────────────────────────────────────────────

def append_row(path: Path, row: dict, columns: list[str]) -> None:
    """Agrega una fila a `path`, migrando el esquema si hiciera falta.

    Si el header del archivo no coincide con `columns`, el archivo se reescribe
    entero con el esquema nuevo (las columnas que no existían quedan vacías en
    las filas viejas) usando un rename atómico, así un corte a mitad de escritura
    nunca deja el CSV corrupto.

    Las claves de `row` que no estén en `columns` son un error (atrapa typos);
    las de `columns` que falten en `row` quedan vacías.
    """
    path = Path(path)

    desconocidas = set(row) - set(columns)
    if desconocidas:
        raise ValueError(
            f"columnas desconocidas para {path.name}: {sorted(desconocidas)}. "
            f"¿Typo, o falta agregarlas al esquema en csv_store.py?"
        )

    serializada = {col: _ser(row.get(col)) for col in columns}

    # Archivo nuevo: header + fila y listo.
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            writer.writerow([serializada[c] for c in columns])
        return

    with path.open("r", newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh), [])

    # Caso normal: el esquema no cambió, append directo.
    if header == columns:
        with path.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([serializada[c] for c in columns])
        return

    _migrar_y_append(path, serializada, columns, header)


def _migrar_y_append(
    path: Path, serializada: dict[str, str], columns: list[str], header: list[str]
) -> None:
    """Reescribe `path` con el esquema `columns` y le agrega la fila nueva."""
    with path.open("r", newline="", encoding="utf-8") as fh:
        filas_viejas = list(csv.DictReader(fh))

    eliminadas = [c for c in header if c not in columns and c not in RENAMES]
    for col in eliminadas:
        print(
            f"⚠ csv_store: columna '{col}' eliminada del esquema; "
            f"datos descartados en {path}"
        )

    migradas = []
    for vieja in filas_viejas:
        nueva = {}
        for col_vieja, valor in vieja.items():
            destino = RENAMES.get(col_vieja, col_vieja)
            if destino in columns:
                nueva[destino] = valor
        migradas.append(nueva)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for fila in migradas:
            writer.writerow([fila.get(c, "") for c in columns])
        writer.writerow([serializada[c] for c in columns])

    os.replace(tmp, path)  # atómico: nunca deja el archivo a medio escribir


# ─────────────────────────────────────────────────────────────────────────────
# Lectura (para report.py)
#
# dtypes explícitos: los vacíos quedan NaN y los flags 1/0 se leen como float
# —intencional— para poder hacer `.mean()` directo sobre ellos.
# ─────────────────────────────────────────────────────────────────────────────

_RUNS_STR = {
    "run_id", "timestamp_utc", "suite", "epoch", "label",
    "git_commit", "git_branch", "llm_provider", "llm_model", "judge_model",
    "embedding_model", "ranker_model", "chunking",
}

_QUESTIONS_STR = {
    "run_id", "question_id", "category", "status", "error",
    "expected_ids", "retrieved_ids", "joined_ids",
    "expected_sources", "retrieved_sources",
    "question", "answer", "reference_answer",
}

DTYPES_RUNS = {c: ("string" if c in _RUNS_STR else "float64") for c in RUNS_COLUMNS}
DTYPES_QUESTIONS = {
    c: ("string" if c in _QUESTIONS_STR else "float64") for c in QUESTIONS_COLUMNS
}


def _load(path: Path, columns: list[str], dtypes: dict[str, str]) -> "pd.DataFrame":
    import pandas as pd

    path = Path(path)
    if not path.exists():
        # DataFrame vacío pero con el esquema puesto: evita FileNotFoundError y
        # los `KeyError` aguas abajo en report.py.
        return pd.DataFrame({c: pd.Series(dtype=dtypes[c]) for c in columns})
    return pd.read_csv(path, dtype=dtypes, keep_default_na=True)


def load_runs(path: Path) -> "pd.DataFrame":
    """Lee `runs.csv`. `timestamp_utc` queda como string: quien necesite fecha
    hace `pd.to_datetime` aparte."""
    return _load(path, RUNS_COLUMNS, DTYPES_RUNS)


def load_questions(path: Path) -> "pd.DataFrame":
    """Lee `questions.csv`."""
    return _load(path, QUESTIONS_COLUMNS, DTYPES_QUESTIONS)


# ─────────────────────────────────────────────────────────────────────────────
# Metadata del harness (eval_meta.yaml)
# ─────────────────────────────────────────────────────────────────────────────

def effective_llm_model(valves) -> str:
    """El modelo de generación que REALMENTE se usó, no el default de los valves.

    `valves.llm_model` es el default de Groq y queda igual aunque se corra con
    `LLM_PROVIDER=ollama`: sin esto, una corrida local se registra como si hubiera
    sido por Groq y queda incomparable para siempre. La resolución replica la de
    `build_generator()` en el pipeline.

    La usan los dos runners (`run_eval.py` y `run_eval_llm.py`) para que registren
    lo mismo.
    """
    import pipeline_ciberseguridad as rag  # lazy: csv_store se importa solo en tests

    modelo_env = os.getenv("LLM_MODEL")
    if rag._llm_provider() == "ollama":
        return modelo_env or rag.DEFAULT_OLLAMA_LLM
    return modelo_env or valves.llm_model


def load_meta(path: Path) -> dict:
    """Lee `eval_meta.yaml`. Sin defaults silenciosos: si falta el archivo o una
    clave obligatoria, error claro."""
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no existe {path}. Es el archivo que declara `baseline_run_id` y "
            f"`epoch`; ver docs/eval/refactor_resultados_csv.md (Paso 2)."
        )

    meta = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for clave in ("baseline_run_id", "epoch"):
        if clave not in meta:
            raise KeyError(f"falta la clave '{clave}' en {path}")
    return meta
