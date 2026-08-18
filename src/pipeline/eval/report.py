"""Análisis y reporte del eval sobre las tablas CSV.

Sin estado propio: lee `results/runs.csv`, `results/questions.csv` y
`eval_meta.yaml` vía `csv_store.load_*` / `load_meta`. **Nunca escribe en las
tablas**; sus únicos outputs son stdout y (en el modo `html`) archivos bajo
`results/report/`.

Corre siempre dentro del container; en el host sólo si hay pandas + pyyaml (no
es un requisito del proyecto).

Diseño: docs/eval/refactor_resultados_csv.md (Paso 5).

Principios (§5.0 del plan):

1. Sólo lee crudos; los agregados se recalculan acá, no se persisten.
2. Degradación en cascada, nunca crash: sin baseline → resumen solo; sin filas
   del baseline en `questions.csv` → delta global con aviso.
3. El color nunca es el único canal: `▲▼=` y texto siempre acompañan.
4. Exit 0 siempre, salvo `--strict` (regresiones) o error de uso (exit 2).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:            # permite `python report.py` desde cualquier cwd
    sys.path.insert(0, str(EVAL_DIR))

import csv_store  # noqa: E402

if TYPE_CHECKING:
    import pandas as pd


# Caída de SAS (0..1) a partir de la cual se considera regresión de generación.
# Se recalibra en H5 de mejoras_harness.md; por ahora es una constante de este módulo.
SAS_REGRESSION_THRESHOLD = 0.05

DEFAULT_RESULTS = EVAL_DIR / "results"
DEFAULT_META = EVAL_DIR / "eval_meta.yaml"

# Columnas que describen CÓMO se corrió el eval, para el diff de config.
#
# Es un subconjunto del grupo "config" de RUNS_COLUMNS: quedan afuera `dataset_n`
# y `n_errors` (se avisan aparte, en el bloque de advertencias) y `duration_s`
# (es un resultado, no una decisión: diffearlo produciría ruido en cada corrida).
CONFIG_COLUMNS = [
    "llm_provider", "llm_model", "judge_model", "temperature",
    "embedding_model", "ranker_model", "retriever_top_k", "ranker_top_k",
    "chunking", "n_docs_store",
]

# Métricas globales del bloque GLOBAL, en orden de lectura. `source_recall` puede
# no existir en corridas viejas: se muestra `–` y no rompe.
GLOBAL_METRICS = ["recall_eff", "hit_eff", "mrr_eff", "source_recall", "sas_mean"]

ANCHO = 64


class ReportError(Exception):
    """Error de uso (run_id inexistente, sin corridas). `main()` lo traduce a exit 2."""


# ─────────────────────────────────────────────────────────────────────────────
# Presentación: color y formato de números
# ─────────────────────────────────────────────────────────────────────────────

_ANSI = {
    "good": "\033[32m", "bad": "\033[31m", "warn": "\033[33m",
    "dim": "\033[2m", "bold": "\033[1m",
}
_RESET = "\033[0m"

_no_color = False   # lo prende --no-color; ver paint()


def _stdout_es_tty() -> bool:
    """`getattr` y no `sys.stdout.isatty()` a secas: durante una corrida stdout es
    el `_Tee` de run_eval.py, que no implementa isatty. Sin esto, el log tee-ado
    quedaría lleno de códigos ANSI."""
    isatty = getattr(sys.stdout, "isatty", None)
    return bool(isatty and isatty())


def paint(s: str, role: str) -> str:
    """Devuelve `s` con el color del rol, o tal cual si el color está desactivado.

    El color SIEMPRE es redundante: refuerza lo que `▲▼=` o el texto ya dicen.
    """
    if _no_color or os.getenv("NO_COLOR") or not _stdout_es_tty():
        return s
    code = _ANSI.get(role)
    return f"{code}{s}{_RESET}" if code else s


def _hdr(titulo: str) -> str:
    """Encabezado de bloque `── TÍTULO ─────…` a ANCHO columnas."""
    izq = f"── {titulo} "
    return izq + "─" * max(0, ANCHO - len(izq))


def _f(v) -> float | None:
    """Valor numérico o None (cubre None, NaN, string vacío y no-numéricos)."""
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x   # NaN != NaN


def num(v, decimales: int = 3) -> str:
    x = _f(v)
    return "–" if x is None else f"{x:.{decimales}f}"


def _celda(v) -> str:
    """Valor de config para mostrar: 15.0 → '15', vacío → '–', texto tal cual.

    Los faltantes llegan de tres formas distintas según el dtype de la columna:
    `None`, `float('nan')` y el `pd.NA` de las columnas string. Las tres tienen
    que verse igual: `str(pd.NA)` es `'<NA>'` y filtrarlo a mano es un olvido
    seguro, así que se normalizan acá y en un solo lugar.
    """
    x = _f(v)
    if x is not None:
        return str(int(x)) if x == int(x) else f"{x:g}"
    if v is None:
        return "–"
    s = str(v).strip()
    return s if s and s.lower() not in {"nan", "<na>", "none", "nat"} else "–"


def _arrow(delta: float) -> str:
    if delta > 0:
        return "▲"
    if delta < 0:
        return "▼"
    return "="


def fmt_delta(cur, base, decimales: int = 3) -> str:
    """`0.841  ▲0.012` — valor actual y su delta contra el baseline.

    Casos degradados (semántica heredada del report viejo): sin ninguno de los
    dos → `–`; sin baseline → `(nuevo)`; sin actual → se muestra el baseline.
    """
    c, b = _f(cur), _f(base)
    if c is None and b is None:
        return "–"
    if b is None:
        return f"{c:.{decimales}f} " + paint("(nuevo)", "dim")
    if c is None:
        return "– " + paint(f"(baseline {b:.{decimales}f})", "dim")
    d = c - b
    rol = "good" if d > 0 else "bad" if d < 0 else "dim"
    return f"{c:.{decimales}f}  " + paint(f"{_arrow(d)}{abs(d):.{decimales}f}", rol)


# ─────────────────────────────────────────────────────────────────────────────
# Datos
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Ctx:
    """Todo lo que hace falta leer del disco, leído una sola vez."""
    runs: "pd.DataFrame"
    questions: "pd.DataFrame"
    meta: dict
    results_dir: Path
    meta_path: Path


def load_context(results_dir: Path | None = None, meta_path: Path | None = None) -> Ctx:
    results_dir = Path(results_dir or DEFAULT_RESULTS)
    meta_path = Path(meta_path or DEFAULT_META)
    return Ctx(
        runs=csv_store.load_runs(results_dir / "runs.csv"),
        questions=csv_store.load_questions(results_dir / "questions.csv"),
        meta=csv_store.load_meta(meta_path),
        results_dir=results_dir,
        meta_path=meta_path,
    )


def latest_run_id(runs: "pd.DataFrame") -> str | None:
    """La última corrida = máximo `run_id` con `suite == 'retrieval'`.

    El run_id es un timestamp UTC, así que el orden lexicográfico ES el
    cronológico y no hace falta parsear fechas. Las corridas `judge` quedan
    afuera: comparar judge contra retrieval no significa nada (llega con H7).
    """
    r = runs[runs["suite"] == "retrieval"]
    return None if r.empty else str(r["run_id"].max())


def _run_row(runs: "pd.DataFrame", run_id: str) -> "pd.Series":
    fila = runs[runs["run_id"] == run_id]
    if fila.empty:
        ultimas = ", ".join(runs["run_id"].dropna().sort_values().tail(5)) or "(ninguna)"
        raise ReportError(f"run_id '{run_id}' no existe; corridas disponibles: {ultimas}")
    return fila.iloc[0]


def resolve_pair(ctx: Ctx, run: str | None = None, baseline: str | None = None):
    """(fila actual, fila baseline | None). Sin `run` → la última corrida; sin
    `baseline` → el `baseline_run_id` del meta (que puede ser null)."""
    run_id = run or latest_run_id(ctx.runs)
    if run_id is None:
        raise ReportError(
            f"no hay ninguna corrida en {ctx.results_dir / 'runs.csv'}; "
            f"corré el eval con scripts/eval.sh"
        )
    row_cur = _run_row(ctx.runs, run_id)

    base_id = baseline if baseline is not None else ctx.meta.get("baseline_run_id")
    if base_id is None or (isinstance(base_id, float) and base_id != base_id):
        return row_cur, None
    base_id = str(base_id).strip()
    if not base_id or base_id.lower() in {"null", "none", "nan"}:
        return row_cur, None
    return row_cur, _run_row(ctx.runs, base_id)


# ─────────────────────────────────────────────────────────────────────────────
# Análisis — puras: DataFrame in → DataFrame out, sin prints.
# Separadas de la capa de presentación para que el modo html las reuse tal cual.
# ─────────────────────────────────────────────────────────────────────────────

def config_diff(row_base: "pd.Series", row_cur: "pd.Series") -> list[tuple]:
    """[(columna, valor_base, valor_cur)] sólo de las columnas que difieren."""
    out = []
    for col in CONFIG_COLUMNS:
        vb, vc = row_base.get(col), row_cur.get(col)
        fb, fc = _f(vb), _f(vc)
        if fb is not None or fc is not None:
            if fb == fc:                      # 15 y 15.0 son la misma config
                continue
        elif _celda(vb) == _celda(vc):
            continue
        out.append((col, _celda(vb), _celda(vc)))
    return out


def category_table(q: "pd.DataFrame", run_base: str, run_cur: str) -> "pd.DataFrame":
    """Recall y SAS promedio por categoría, un lado por corrida, con sus deltas.

    Ordenada por Δrecall ascendente: lo que más empeoró queda arriba.
    """
    def lado(rid: str) -> "pd.DataFrame":
        d = q[q["run_id"] == rid]
        return d.groupby("category").agg(
            n=("question_id", "size"),
            recall=("recall_eff", "mean"),
            sas=("sas", "mean"),
        )

    t = lado(run_base).join(lado(run_cur), how="outer", lsuffix="_base", rsuffix="_cur")
    # `n` es el de la corrida ACTUAL (el sujeto de la tabla), no el máximo de los
    # dos lados: si las corridas tienen distinto nº de preguntas, el máximo hace
    # creer que el promedio actual se calculó sobre más preguntas de las que hubo.
    t["n"] = t["n_cur"].fillna(t["n_base"])
    t["d_recall"] = t["recall_cur"] - t["recall_base"]
    t["d_sas"] = t["sas_cur"] - t["sas_base"]
    t = t.sort_values("d_recall", ascending=True, na_position="last").reset_index()
    return t[["category", "n", "recall_base", "recall_cur", "d_recall",
              "sas_base", "sas_cur", "d_sas"]]


def question_pivot(q: "pd.DataFrame", metric: str, run_ids: list[str]) -> "pd.DataFrame":
    """Matriz pregunta × corrida para una métrica. La base de todo lo por-pregunta."""
    import pandas as pd

    sub = q[q["run_id"].isin(run_ids)]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="question_id", columns="run_id", values=metric,
                           aggfunc="first", dropna=False)


def _paired(q: "pd.DataFrame", metric: str, run_base: str, run_cur: str) -> "pd.DataFrame":
    """question_id | base | cur | delta — una fila por pregunta de cualquiera de las dos."""
    import numpy as np
    import pandas as pd

    p = question_pivot(q, metric, [run_base, run_cur])
    if p.empty:
        return pd.DataFrame(columns=["question_id", "base", "cur", "delta"])
    for rid in (run_base, run_cur):
        if rid not in p.columns:              # corrida sin filas para esa métrica
            p[rid] = np.nan
    d = p[[run_base, run_cur]].copy()
    d.columns = ["base", "cur"]
    d = d.reset_index()
    d["delta"] = d["cur"] - d["base"]
    return d


def regressions(q: "pd.DataFrame", run_base: str, run_cur: str) -> dict[str, "pd.DataFrame"]:
    """{'retrieval': df, 'sas': df} con las preguntas que empeoraron.

    Retrieval: cualquier caída de `recall_eff` (perder una pregunta es grave, no
    hay umbral). Generación: caída de SAS mayor al umbral, porque el SAS tiene
    ruido propio entre corridas.
    """
    ret = _paired(q, "recall_eff", run_base, run_cur)
    sas = _paired(q, "sas", run_base, run_cur)
    return {
        "retrieval": ret[ret["delta"] < 0].sort_values("delta"),
        "sas": sas[sas["delta"] < -SAS_REGRESSION_THRESHOLD].sort_values("delta"),
    }


def improvements(q: "pd.DataFrame", run_base: str, run_cur: str) -> dict[str, "pd.DataFrame"]:
    """Simétrico de `regressions`: las subas también validan un cambio."""
    ret = _paired(q, "recall_eff", run_base, run_cur)
    sas = _paired(q, "sas", run_base, run_cur)
    return {
        "retrieval": ret[ret["delta"] > 0].sort_values("delta", ascending=False),
        "sas": sas[sas["delta"] > SAS_REGRESSION_THRESHOLD].sort_values("delta", ascending=False),
    }


def question_membership(q: "pd.DataFrame", run_base: str, run_cur: str) -> tuple[list, list]:
    """(nuevas, desaparecidas) — preguntas que están en una corrida y no en la otra."""
    b = set(q.loc[q["run_id"] == run_base, "question_id"].dropna())
    c = set(q.loc[q["run_id"] == run_cur, "question_id"].dropna())
    return sorted(c - b), sorted(b - c)


# ─────────────────────────────────────────────────────────────────────────────
# Modo `compare`
# ─────────────────────────────────────────────────────────────────────────────

def _tabla(df: "pd.DataFrame", indent: str = "  ") -> str:
    """DataFrame → texto. `to_string` y no `to_markdown`: to_markdown depende de
    tabulate, que no está en la imagen."""
    texto = df.to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}")
    return "\n".join(indent + linea for linea in texto.splitlines())


def _ident(row: "pd.Series") -> str:
    label = str(row.get("label") or "").strip()
    label = f"'{label}'" if label and label.lower() != "nan" else "sin label"
    return f"{row['run_id']} ({label}, epoch {_celda(row.get('epoch'))})"


def _print_resumen_solo(row: "pd.Series") -> None:
    """Sin baseline no hay delta: se muestran las métricas de la corrida a secas."""
    print("\n  Global:")
    for m in GLOBAL_METRICS:
        print(f"    {m:<14} {num(row.get(m))}")


def _print_advertencias(row_base: "pd.Series", row_cur: "pd.Series") -> None:
    """Todo lo que hace que el delta de abajo mienta. Va ANTES de los números."""
    avisos = []
    if _celda(row_base.get("epoch")) != _celda(row_cur.get("epoch")):
        avisos.append(
            f"topologías diferentes ({_celda(row_base.get('epoch'))} vs "
            f"{_celda(row_cur.get('epoch'))}); el delta NO es comparable. Re-baseá."
        )
    nb, nc = _f(row_base.get("dataset_n")), _f(row_cur.get("dataset_n"))
    if nb is not None and nc is not None and nb != nc:
        avisos.append(
            f"distinto nº de preguntas ({_celda(nb)} vs {_celda(nc)}); "
            f"promedios no comparables."
        )
    for etiqueta, row in (("baseline", row_base), ("actual", row_cur)):
        n_err = _f(row.get("n_errors"))
        if n_err:
            avisos.append(f"la corrida {etiqueta} tuvo {_celda(n_err)} pregunta(s) con error.")
    for a in avisos:
        print("  " + paint(f"⚠ {a}", "warn"))
    if avisos:
        print()


def _print_config_diff(row_base: "pd.Series", row_cur: "pd.Series") -> None:
    print(_hdr("CONFIG"))
    diff = config_diff(row_base, row_cur)
    if not diff:
        print("  config idéntica")
        return
    ancho = max(len(c) for c, _, _ in diff)
    for col, vb, vc in diff:
        print(f"  {col:<{ancho}} : {vb} → {vc}")


def _print_global(row_base: "pd.Series", row_cur: "pd.Series") -> None:
    print("\n" + _hdr("GLOBAL"))
    for m in GLOBAL_METRICS:
        print(f"  {m:<14} {fmt_delta(row_cur.get(m), row_base.get(m))}")


def _print_categorias(q: "pd.DataFrame", base_id: str, cur_id: str) -> None:
    print("\n" + _hdr("POR CATEGORÍA"))
    t = category_table(q, base_id, cur_id)
    if t.empty:
        print("  sin detalle por categoría")
        return
    display = t.rename(columns={
        "recall_base": "recall_b", "recall_cur": "recall_c", "d_recall": "Δrecall",
        "sas_base": "sas_b", "sas_cur": "sas_c", "d_sas": "Δsas",
    })
    display["n"] = display["n"].map(_celda)
    print(_tabla(display))


def _print_lista_preguntas(df: "pd.DataFrame", titulo: str, rol: str,
                           decimales: int = 2) -> None:
    print("\n  " + paint(titulo, rol))
    ancho = max(len(str(qid)) for qid in df["question_id"])
    for _, fila in df.iterrows():
        print(f"    {str(fila['question_id']):<{ancho}}  "
              f"{num(fila['base'], decimales)} → {num(fila['cur'], decimales)}")


def compare(run_id: str | None = None, baseline_run_id: str | None = None,
            results_dir: Path | None = None, meta_path: Path | None = None) -> int:
    """Imprime el reporte de la corrida contra su baseline. Devuelve el nº de
    regresiones por pregunta (retrieval + SAS).

    Única implementación del modo: la llaman `main()` y `run_eval.py`.
    """
    ctx = load_context(results_dir, meta_path)
    row_cur, row_base = resolve_pair(ctx, run_id, baseline_run_id)
    cur_id = str(row_cur["run_id"])

    print("\n" + _hdr("COMPARACIÓN"))
    print(f"  run       {_ident(row_cur)}")

    if row_base is None:
        print("  " + paint("⚠ sin baseline configurado (eval_meta.yaml)", "warn"))
        _print_resumen_solo(row_cur)
        print("\n  " + paint("Fijalo con: scripts/eval.sh --set-baseline", "dim"))
        return 0

    base_id = str(row_base["run_id"])
    print(f"  baseline  {_ident(row_base)}\n")

    _print_advertencias(row_base, row_cur)
    _print_config_diff(row_base, row_cur)
    _print_global(row_base, row_cur)

    # Degradación: `questions.csv` es gitignored, así que el detalle del baseline
    # puede no existir en esta máquina. El delta global (runs.csv) sirve igual.
    q = ctx.questions
    if q.empty or base_id not in set(q["run_id"].dropna()):
        print("\n  " + paint(
            "⚠ sin detalle por pregunta del baseline en esta máquina; "
            "sólo delta global (runs.csv)", "warn"))
        print("\n  " + paint("Detalle HTML: python report.py html", "dim"))
        return 0

    _print_categorias(q, base_id, cur_id)

    regs = regressions(q, base_id, cur_id)
    mejoras = improvements(q, base_id, cur_id)
    n_reg = len(regs["retrieval"]) + len(regs["sas"])

    print("\n" + _hdr("POR PREGUNTA"))
    if n_reg == 0:
        print("  Sin regresiones por pregunta. 🎉")
    else:
        if not regs["retrieval"].empty:
            _print_lista_preguntas(regs["retrieval"],
                                   "⚠ Regresiones de RETRIEVAL (recall bajó):", "bad")
        if not regs["sas"].empty:
            _print_lista_preguntas(
                regs["sas"],
                f"⚠ Regresiones de GENERACIÓN (SAS bajó > {SAS_REGRESSION_THRESHOLD}):",
                "bad", decimales=3)

    if not mejoras["retrieval"].empty or not mejoras["sas"].empty:
        total = len(mejoras["retrieval"]) + len(mejoras["sas"])
        detalle = []
        if not mejoras["retrieval"].empty:
            detalle.append(", ".join(str(x) for x in mejoras["retrieval"]["question_id"]))
        if not mejoras["sas"].empty:
            detalle.append("SAS: " + ", ".join(str(x) for x in mejoras["sas"]["question_id"]))
        print("\n  " + paint(f"▲ Mejoras ({total}): " + " · ".join(detalle), "dim"))

    nuevas, desaparecidas = question_membership(q, base_id, cur_id)
    if nuevas:
        print("  " + paint(f"· preguntas nuevas vs baseline: {', '.join(nuevas)}", "dim"))
    if desaparecidas:
        print("  " + paint(f"· preguntas del baseline ausentes: {', '.join(desaparecidas)}", "dim"))

    print("\n  " + paint("Detalle HTML: python report.py html", "dim"))
    return n_reg


# ─────────────────────────────────────────────────────────────────────────────
# Modo `runs` — historial
# ─────────────────────────────────────────────────────────────────────────────

_BLOQUES = "▁▂▃▄▅▆▇█"


def spark(vals) -> str:
    """Sparkline unicode con **escala fija 0–1**, no min-max relativo.

    Es a propósito: con min-max, una métrica que se movió entre 0.80 y 0.82 se
    vería igual de dramática que una que fue de 0.10 a 0.95, y las dos líneas no
    se podrían comparar entre sí. Faltante → `·`.
    """
    salida = []
    for v in vals:
        x = _f(v)
        salida.append("·" if x is None else _BLOQUES[max(0, min(7, round(x * 7)))])
    return "".join(salida)


def _ultimo_valor(vals) -> str:
    for v in reversed(list(vals)):
        if _f(v) is not None:
            return num(v)
    return "–"


def corridas_completas(ctx: Ctx) -> "pd.DataFrame":
    """Corridas de la epoch vigente con el dataset entero, en orden cronológico.

    Las parciales (`--limit N`) quedan afuera: un promedio sobre 3 preguntas al
    lado de uno sobre 39 no es un punto de la misma serie, y meterlo en la
    sparkline haría ver saltos que no son cambios del sistema.

    **"Completa" = `dataset_n` MÁXIMO de la epoch, no la moda.** El plan (§5.4,
    gráfico 1) dice moda; se cambió a propósito y está documentado en la bitácora
    (hallazgo 13). Motivo: durante el desarrollo uno corre muchas veces con
    `--limit 3` y pocas con el dataset entero, así que la moda termina siendo el
    smoke test y la evolución deja afuera justo las corridas que importan — que es
    exactamente lo que pasaba acá: 5 corridas de 3 preguntas contra 1 de 39.

    Contra conocida: si algún día se achica el dataset a propósito, el máximo
    queda apuntando al tamaño viejo hasta que se re-basee. Es aceptable porque
    achicar el dataset es un cambio deliberado y raro, mientras que correr con
    `--limit` es cosa de todos los días.
    """
    r = ctx.runs[(ctx.runs["suite"] == "retrieval")
                 & (ctx.runs["epoch"] == ctx.meta.get("epoch"))].sort_values("run_id")
    if r.empty or r["dataset_n"].isna().all():
        return r
    return r[r["dataset_n"] == r["dataset_n"].max()]


def cmd_runs(args) -> int:
    ctx = load_context(_opt(args, "results", DEFAULT_RESULTS),
                       _opt(args, "meta", DEFAULT_META))
    todas = _opt(args, "all", False)
    base_id = str(ctx.meta.get("baseline_run_id") or "")

    r = ctx.runs if todas else ctx.runs[ctx.runs["suite"] == "retrieval"]
    r = r.sort_values("run_id")
    if not todas:
        r = r.tail(15)

    print("\n" + _hdr("CORRIDAS" if todas else "ÚLTIMAS CORRIDAS"))
    if r.empty:
        print("  todavía no hay corridas; corré el eval con scripts/eval.sh")
        return 0

    cols = ["run_id", "suite", "epoch", "label", "dataset_n",
            "recall_eff", "mrr_eff", "sas_mean", "n_errors"]
    d = r[cols].copy()
    # El `*` va pegado al run_id y no en una columna aparte: así la marca no se
    # pierde de vista cuando la tabla es ancha.
    d["run_id"] = d["run_id"].map(lambda x: ("* " if str(x) == base_id else "  ") + str(x))
    for c in ("dataset_n", "n_errors"):
        d[c] = d[c].map(_celda)
    d["label"] = d["label"].map(lambda v: _celda(v))
    if not todas:
        d = d.drop(columns=["suite"])
    print(_tabla(d))
    if base_id:
        print("\n  " + paint(f"* baseline ({base_id})", "dim"))
    else:
        print("\n  " + paint("sin baseline configurado (eval_meta.yaml)", "warn"))

    comp = corridas_completas(ctx)
    if len(comp) >= 2:
        print("\n" + _hdr(f"EVOLUCIÓN (epoch {_celda(ctx.meta.get('epoch'))})"))
        for m in ("recall_eff", "mrr_eff", "sas_mean"):
            print(f"  {m:<12}  {spark(comp[m])}  {_ultimo_valor(comp[m])}")
        print("\n  " + paint(
            f"{len(comp)} corridas completas (dataset_n={_celda(comp['dataset_n'].iloc[-1])}); "
            f"escala fija 0–1", "dim"))
    else:
        # §5.4 pide omitir la evolución con < 2 corridas completas, pero omitirla
        # en silencio se lee como "no pasó nada". El aviso dice además cuál es el
        # criterio, que es lo que uno se pregunta al no ver el bloque.
        n = _celda(comp["dataset_n"].max()) if not comp.empty else "?"
        print("\n  " + paint(
            f"aún no hay historia suficiente: {len(comp)} corrida(s) completa(s) "
            f"(dataset_n={n}) en la epoch vigente; hacen falta 2", "dim"))
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Modo `question` — drill-down de una pregunta
# ─────────────────────────────────────────────────────────────────────────────

def via_series(d: "pd.DataFrame"):
    """De qué retriever vino el documento: emb / kw / emb+kw / -.

    Los flags se leen como float (así `.mean()` funciona directo sobre ellos),
    por eso la comparación es contra 1.0 y no contra True.
    """
    import numpy as np

    emb = d["via_embedding"] == 1.0
    kw = d["via_keyword"] == 1.0
    return np.select([emb & kw, emb, kw], ["emb+kw", "emb", "kw"], default="-")


def cmd_question(args) -> int:
    import difflib

    ctx = load_context(_opt(args, "results", DEFAULT_RESULTS),
                       _opt(args, "meta", DEFAULT_META))
    qid = args.question_id
    q = ctx.questions

    ids = sorted(set(q["question_id"].dropna().astype(str))) if not q.empty else []
    if qid not in ids:
        sugerencias = difflib.get_close_matches(qid, ids, n=5)
        detalle = (f"; ¿quisiste decir {', '.join(sugerencias)}?" if sugerencias
                   else f"; hay {len(ids)} preguntas en questions.csv")
        raise ReportError(f"la pregunta '{qid}' no existe{detalle}")

    d = q[q["question_id"].astype(str) == qid].sort_values("run_id")
    ultima = d.iloc[-1]

    print("\n" + _hdr(f"PREGUNTA {qid}"))
    print(f"  categoría     {_celda(ultima.get('category'))}")
    print(f"  expected_ids  {_celda(ultima.get('expected_ids'))}")
    print(f"  corridas      {len(d)}")
    print("\n  " + csv_store.unescape_text(str(ultima.get("question") or "")).strip())

    print("\n" + _hdr("EVOLUCIÓN"))
    t = d.merge(ctx.runs[["run_id", "epoch", "label"]], on="run_id", how="left")
    t["via"] = via_series(t)
    display = t[["run_id", "epoch", "label", "recall_eff", "rank_first_hit",
                 "via", "sas", "faithfulness"]].copy()
    display["rank_first_hit"] = display["rank_first_hit"].map(_celda)
    display["label"] = display["label"].map(_celda)
    display["epoch"] = display["epoch"].map(_celda)
    print(_tabla(display))

    # La respuesta del baseline sólo si difiere: repetir el mismo texto dos veces
    # no aporta y empuja lo importante fuera de la pantalla.
    base_id = str(ctx.meta.get("baseline_run_id") or "")
    resp_actual = csv_store.unescape_text(str(ultima.get("answer") or "")).strip()
    fila_base = d[d["run_id"] == base_id]
    resp_base = (csv_store.unescape_text(str(fila_base.iloc[0].get("answer") or "")).strip()
                 if not fila_base.empty else "")

    if resp_actual:
        print("\n" + _hdr(f"RESPUESTA — {ultima['run_id']}"))
        print(resp_actual)
    if resp_base and resp_base != resp_actual:
        print("\n" + _hdr(f"RESPUESTA — baseline {base_id}"))
        print(paint(resp_base, "dim"))
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Entrada
# ─────────────────────────────────────────────────────────────────────────────

def _parser() -> argparse.ArgumentParser:
    # default=SUPPRESS en el padre: si no, el subparser pisa con su propio default
    # lo que el parser principal ya había leído (`report.py --no-color compare`).
    padre = argparse.ArgumentParser(add_help=False)
    padre.add_argument("--results", type=Path, default=argparse.SUPPRESS,
                       help="directorio de resultados (default: eval/results)")
    padre.add_argument("--meta", type=Path, default=argparse.SUPPRESS,
                       help="eval_meta.yaml (default: eval/eval_meta.yaml)")
    padre.add_argument("--no-color", action="store_true", default=argparse.SUPPRESS,
                       help="sin códigos ANSI (también respeta NO_COLOR y el pipe)")
    padre.add_argument("--strict", action="store_true", default=argparse.SUPPRESS,
                       help="exit 1 si hubo regresiones (para un hook o CI)")

    # Flags de `compare` en un padre aparte: sin subcomando el default ES compare,
    # así que `report.py --run X` tiene que andar igual que `report.py compare --run X`.
    padre_cmp = argparse.ArgumentParser(add_help=False)
    padre_cmp.add_argument("--run", default=argparse.SUPPRESS,
                           help="run_id a reportar (default: la última corrida)")
    padre_cmp.add_argument("--baseline", default=argparse.SUPPRESS,
                           help="run_id del baseline (default: el de eval_meta.yaml)")

    ap = argparse.ArgumentParser(
        description="Reporte del eval sobre runs.csv / questions.csv",
        parents=[padre, padre_cmp])
    # OJO: `cmd` y nada más. `set_defaults` MUTA el `.default` de los objetos
    # action, y `parents=` comparte esas mismas instancias con el subparser: usarlo
    # acá para --results/--meta/etc. les borraría el SUPPRESS y volvería el bug de
    # arriba. Los defaults reales se resuelven después de parsear, en `_opt`.
    ap.set_defaults(cmd="compare")

    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("compare", parents=[padre, padre_cmp],
                   help="una corrida contra el baseline (default)")

    p_runs = sub.add_parser("runs", parents=[padre], help="historial de corridas")
    p_runs.add_argument("--all", action="store_true", default=argparse.SUPPRESS,
                        help="todas las corridas, incluidas las de suite=judge")

    p_q = sub.add_parser("question", parents=[padre],
                         help="evolución de una pregunta en todas las corridas")
    p_q.add_argument("question_id", help="id de la pregunta (ver dataset.yaml)")
    return ap


def _opt(args: argparse.Namespace, nombre: str, default):
    """Valor de un flag global, venga de antes o de después del subcomando.

    Con `default=SUPPRESS` el atributo sólo existe si el usuario lo escribió, así
    que el subparser no puede pisar lo que ya había parseado el parser principal.
    """
    return getattr(args, nombre, default)


def main() -> None:
    global _no_color
    args = _parser().parse_args()
    _no_color = _opt(args, "no_color", False)

    try:
        if args.cmd == "runs":
            cmd_runs(args)
            raise SystemExit(0)
        if args.cmd == "question":
            cmd_question(args)
            raise SystemExit(0)
        n_reg = compare(run_id=_opt(args, "run", None),
                        baseline_run_id=_opt(args, "baseline", None),
                        results_dir=_opt(args, "results", DEFAULT_RESULTS),
                        meta_path=_opt(args, "meta", DEFAULT_META))
    except ReportError as e:
        print(f"\n✗ {e}", file=sys.stderr)
        raise SystemExit(2)

    # Sin --strict el reporte informa, no bloquea.
    raise SystemExit(1 if (_opt(args, "strict", False) and n_reg) else 0)


if __name__ == "__main__":
    main()
