#!/usr/bin/env python3
"""
Analiza métricas de los archivos splitter_*.txt generados por inspect.sh.
Uso: python analyze_chunks.py
"""

import re
from pathlib import Path
from statistics import mean, stdev

SPLITTER_DIR = Path(__file__).parent
SEP = "-" * 60


def parse_chunks(path: Path) -> list[str]:
    text = path.read_text()
    # Extrae el contenido entre líneas divisorias de chunks
    return [
        chunk.strip()
        for chunk in re.findall(r"-{60}\n(.*?)-{60}", text, re.DOTALL)
        if chunk.strip()
    ]


def metrics(chunks: list[str]) -> dict:
    words = [len(c.split()) for c in chunks]
    return {
        "chunks":    len(words),
        "min":       min(words),
        "max":       max(words),
        "avg":       round(mean(words), 1),
        "std":       round(stdev(words), 1) if len(words) > 1 else 0.0,
        "cortos_%":  round(100 * sum(1 for w in words if w < 30) / len(words), 1),
        "largos_%":  round(100 * sum(1 for w in words if w > 300) / len(words), 1),
    }


def main():
    files = sorted(SPLITTER_DIR.glob("splitter_*.txt"))
    if not files:
        print("No hay archivos splitter_*.txt en este directorio.")
        print("Corré ./inspect.sh primero.")
        return

    results = []
    for f in files:
        chunks = parse_chunks(f)
        if not chunks:
            continue
        m = metrics(chunks)
        m["splitter"] = f.stem.replace("splitter_", "")
        results.append(m)

    if not results:
        print("No se pudieron extraer chunks de ningún archivo.")
        return

    # Tabla de resultados
    cols = ["splitter", "chunks", "min", "max", "avg", "std", "cortos_%", "largos_%"]
    headers = {
        "splitter":  "splitter",
        "chunks":    "n",
        "min":       "min",
        "max":       "max",
        "avg":       "avg",
        "std":       "std",
        "cortos_%":  "<30p %",
        "largos_%":  ">300p %",
    }
    widths = {c: max(len(headers[c]), max(len(str(r[c])) for r in results)) for c in cols}

    def row(r):
        return "  ".join(str(r[c]).ljust(widths[c]) for c in cols)

    print("\nMétricas de chunks (palabras)\n")
    print("  ".join(headers[c].ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in results:
        print(row(r))

    print()
    print("  <30p  = % de chunks con menos de 30 palabras  (posible ruido)")
    print("  >300p = % de chunks con más de 300 palabras   (posible exceso de contexto)")
    print()

    # Destacar el splitter con menor std (más uniforme)
    best_std = min(results, key=lambda r: r["std"])
    print(f"  Más uniforme (menor std): {best_std['splitter']}  (std={best_std['std']})")


if __name__ == "__main__":
    main()
