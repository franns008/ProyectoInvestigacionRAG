"""Orquestador de los fetchers de la Capa 1: baja todos los datos crudos de una.

Cada fuente vive en su propio `fetch_<fuente>.py` y se puede correr suelta; esto sólo
las encadena y reporta. Todas exponen el mismo contrato: `fetch(force: bool) -> Result`.

Las descargas son condicionales: si el servidor dice que el archivo no cambió, no se
baja nada. Así que correrlo seguido es barato y mantiene los datos frescos — que es lo
que hace falta con EPSS y KEV, que se actualizan a diario.

NVD queda fuera del set por defecto porque su carga inicial son ~250k CVEs y domina el
tiempo por órdenes de magnitud; las otras cuatro son ~57 MB en segundos. Es el mismo
criterio que usa run_indexing.py con --include-cve.

Uso:
    python src/ingestion/fetch_all.py                  # osv, epss, kev, cwe
    python src/ingestion/fetch_all.py --include-nvd    # + NVD (lento la primera vez)
    python src/ingestion/fetch_all.py --only cwe osv
    python src/ingestion/fetch_all.py --force          # ignora checkpoints y rebaja todo
"""

import argparse
import importlib
import sys

from _http import RAW_DIR, REPO_ROOT, SIN_CAMBIOS, Result

# El orden importa poco, pero conviene dejar OSV primero (es el único obligatorio para
# el escaneo de dependencias) y NVD último (es el lento).
SOURCES = ["osv", "epss", "kev", "cwe", "nvd"]
# NVD es la única excepción, y se declara una sola vez para que sumar una fuente nueva
# a SOURCES la incluya en los defaults sin tener que acordarse de otra lista.
OPT_IN = "nvd"
DEFAULT_SOURCES = [name for name in SOURCES if name != OPT_IN]

FALLO = "FALLO"


def main() -> int:
    parser = argparse.ArgumentParser(description="Baja los datos crudos de la Capa 1 a data/raw/")
    parser.add_argument("--only", nargs="+", choices=SOURCES,
                        help="correr sólo estas fuentes")
    parser.add_argument("--include-nvd", action="store_true",
                        help=f"sumar {OPT_IN.upper()}, que queda afuera por defecto (~250k CVEs)")
    parser.add_argument("--force", action="store_true",
                        help="ignora checkpoints y rebaja aunque el servidor diga que no cambió")
    args = parser.parse_args()

    selected = args.only or DEFAULT_SOURCES + ([OPT_IN] if args.include_nvd else [])

    results: dict[str, Result] = {}
    for name in selected:
        print(f"\n=== {name} ===")
        try:
            # Import perezoso: así una dependencia que sólo usa una fuente (NVD y su
            # python-dotenv) no rompe la corrida de las demás.
            module = importlib.import_module(f"fetch_{name}")
            results[name] = module.fetch(force=args.force)
        except Exception as error:
            # Una fuente caída no debería impedir bajar las demás.
            print(f"[{name}] ERROR: {error}", file=sys.stderr)
            results[name] = Result(FALLO, str(error))

    print("\n=== resumen ===")
    for name, result in results.items():
        print(f"{result.status:<12} {name:<5} {result.detail}")

    cambiaron = [n for n, r in results.items() if r.status not in (SIN_CAMBIOS, FALLO)]
    fallaron = [n for n, r in results.items() if r.status == FALLO]

    if cambiaron:
        print(f"\n{len(cambiaron)} de {len(results)} con novedades ({', '.join(cambiaron)}) "
              f"-> {RAW_DIR.relative_to(REPO_ROOT)}")
    elif not fallaron:
        print("\nTodo al día: ninguna fuente cambió desde la última corrida.")

    if fallaron:
        print(f"\nFallaron {len(fallaron)} de {len(results)}: {', '.join(fallaron)}", file=sys.stderr)
        return 1

    # Sólo si la selección fue la de por defecto: con --only el usuario ya eligió.
    if not args.only and OPT_IN not in selected:
        print(f"({OPT_IN.upper()} no se bajó — sumalo con --include-nvd)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
