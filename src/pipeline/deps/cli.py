"""
CLI del escaneo de dependencias: requirements.txt -> hallazgos priorizados.

Corre sin stack, sin LLM y sin base de datos. Sirve para tres cosas:
verificar el resolver contra el dump real de OSV, alimentar la extensión de VSCode
(`--json`), y tener algo que mostrar sin levantar infraestructura.

    python -m deps.cli requirements.txt --data data/raw

Espera en `--data` los archivos que bajan los fetchers de la Fase 1:
    osv/all.zip                        (obligatorio)
    epss/epss_scores-current.csv.gz    (opcional)
    kev/known_exploited_vulnerabilities.json (opcional)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from .prioritize import funnel, load_epss, load_kev, prioritize
from .resolver import build_index, load_osv_zip, parse_requirements, scan


def _first_existing(base: Path, *candidates: str) -> Path | None:
    for candidate in candidates:
        path = base / candidate
        if path.exists():
            return path
    return None


def _plural(n: int, singular: str, plural: str) -> str:
    return f"{n} {singular if n == 1 else plural}"


def _render(vulnerabilities, requirements, stats, elapsed_ms) -> str:
    lines: list[str] = []
    # Sale de stats y no de `vulnerabilities`, que puede venir recortada por --top.
    afectados = stats["paquetes"]
    escaneadas = sum(1 for r in requirements if r.is_scannable)

    lines.append("")
    if not stats["total"]:
        lines.append(
            "Sin vulnerabilidades conocidas en "
            f"{_plural(escaneadas, 'dependencia fijada', 'dependencias fijadas')}."
        )
    else:
        lines.append(
            f"{_plural(afectados, 'dependencia', 'dependencias')} con vulnerabilidades "
            f"conocidas — {_plural(stats['total'], 'CVE', 'CVEs')} en total."
        )

    if stats["total"]:
        lines.append("")
        lines.append(f"  {'CVEs que afectan tus versiones':<44}{stats['total']:>5}")
        lines.append(
            f"  {'  filtrando por CVSS >= 7.0':<44}{stats['cvss_alto']:>5}"
            f"   ({stats['cvss_alto'] / stats['total']:.0%})"
        )
        lines.append(
            f"  {'  filtrando por EPSS >= 10%':<44}{stats['epss_alto']:>5}"
            f"   ({stats['epss_alto'] / stats['total']:.0%})"
        )
        lines.append(
            f"  {'  presentes en CISA KEV (explotadas hoy)':<44}{stats['kev']:>5}"
            f"   ({stats['kev'] / stats['total']:.0%})"
        )
        # Lo que el embudo NO puede ordenar, dicho explícitamente.
        if stats["sin_cve"] or stats["sin_cvss"]:
            lines.append(
                f"  ({stats['sin_cve']} sin alias a CVE — quedan fuera del cruce con EPSS/KEV; "
                f"{stats['sin_cvss']} sin CVSS calculable)"
            )

    for vulnerability in vulnerabilities:
        marca = "[KEV]" if vulnerability.kev else "     "
        arreglo = f" -> {vulnerability.fixed_version}" if vulnerability.fixed_version else " (sin arreglo publicado)"
        score = f"CVSS {vulnerability.cvss_score}" if vulnerability.cvss_score is not None else "CVSS s/d"
        lines.append("")
        lines.append(
            f"{marca} {vulnerability.package} {vulnerability.installed_version}{arreglo}"
        )
        lines.append(
            f"       {vulnerability.identifier} · {score} · EPSS {vulnerability.epss:.1%}"
            + (f" · {', '.join(vulnerability.cwe_ids)}" if vulnerability.cwe_ids else "")
        )
        if vulnerability.summary:
            lines.append(f"       {vulnerability.summary[:100]}")

    # Se listan también los `-r otro.txt` (esconden dependencias que no se miraron);
    # las banderas puras de pip (--index-url y compañía) son ruido y no.
    omitidas = [r for r in requirements if not r.is_scannable and not r.raw_name.startswith("--")]
    if omitidas:
        lines.append("")
        lines.append(f"No escaneadas ({len(omitidas)}):")
        for requirement in omitidas:
            lines.append(f"  línea {requirement.line_number}: {requirement.raw_name} — {requirement.skip_reason}")

    lines.append("")
    lines.append(
        f"({_plural(escaneadas, 'dependencia fijada', 'dependencias fijadas')}, {elapsed_ms:.0f} ms)"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Escaneo de dependencias contra OSV.")
    parser.add_argument("requirements", type=Path, help="ruta al requirements.txt")
    parser.add_argument("--data", type=Path, default=Path("data/raw"), help="directorio con osv/, epss/ y kev/")
    parser.add_argument("--json", action="store_true", help="salida JSON (la que consume la extensión)")
    parser.add_argument("--top", type=int, default=0, help="mostrar sólo los N primeros hallazgos")
    args = parser.parse_args(argv)

    osv_zip = _first_existing(args.data, "osv/all.zip", "osv/PyPI/all.zip", "all.zip")
    if osv_zip is None:
        parser.error(f"no encontré el dump de OSV bajo {args.data}/osv/. Corré primero el fetcher (Fase 1).")

    epss_path = _first_existing(args.data, "epss/epss_scores-current.csv.gz", "epss.csv.gz")
    kev_path = _first_existing(args.data, "kev/known_exploited_vulnerabilities.json", "kev.json")

    started = time.perf_counter()
    index = build_index(load_osv_zip(osv_zip))
    requirements = parse_requirements(args.requirements.read_text(encoding="utf-8"))
    vulnerabilities = prioritize(
        scan(requirements, index),
        load_epss(epss_path) if epss_path else {},
        load_kev(kev_path) if kev_path else set(),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    stats = funnel(vulnerabilities)
    if args.top:
        vulnerabilities = vulnerabilities[: args.top]

    if args.json:
        print(
            json.dumps(
                {
                    "manifest": str(args.requirements),
                    "funnel": stats,
                    "skipped": [
                        {"line": r.line_number, "raw": r.raw_name, "reason": r.skip_reason}
                        for r in requirements
                        if not r.is_scannable
                    ],
                    "findings": [dataclasses.asdict(v) for v in vulnerabilities],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(_render(vulnerabilities, requirements, stats, elapsed_ms))

    return 0


if __name__ == "__main__":
    sys.exit(main())
