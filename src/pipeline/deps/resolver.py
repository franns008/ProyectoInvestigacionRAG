"""
Resolución determinística: qué advisories de OSV afectan a ESTAS versiones exactas.

Es la mitad determinística del escaneo descrito en docs/escaneo_dependencias.md.
No usa LLM, ni embeddings, ni base de datos: dado un requirements.txt y el dump de
OSV para PyPI, la respuesta es siempre la misma. El ORDEN en que hay que atacar los
hallazgos lo decide prioritize.py.

Tres decisiones que el esquema de OSV obliga a tomar (las tres medidas sobre el dump
real, ver docs/escaneo_dependencias.md):

- **Un mismo CVE llega por varios advisories.** GHSA y PYSEC publican el mismo fallo
  por separado: sin colapsarlos, un requirements.txt de 10 dependencias devuelve 362
  hallazgos en vez de 144. scan() los deduplica por CVE y UNE sus `cwe_ids`, lo que
  además cierra un hueco real: los 7.346 advisories PYSEC no traen `cwe_ids` en
  ningún caso y el GHSA hermano se los presta (cobertura de CWE: 61% -> 98%).

- **La versión de arreglo es el menor `fixed` estrictamente mayor que la instalada.**
  Tomar el mínimo global de los `fixed` propone downgrades: para django 2.2.0
  devolvería 2.1.9, que es anterior a lo que ya está instalado.

- **Los rangos `GIT` se ignoran.** Referencian commits, no versiones de PyPI. El dump
  trae 1.581 y ninguno es resoluble contra un pin de requirements.txt.

- **Los advisories retirados (`withdrawn`) no cuentan.** Son 492 (3,7% del dump) y no
  son inocuos: GHSA-56pw-mpj4-fxww es un duplicado de CVE-2023-4863 publicado sin
  aliases, así que la deduplicación por CVE no lo agarra y aparecería como un hallazgo
  extra con el texto "Duplicate Advisory" a la vista.
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from packaging.requirements import InvalidRequirement, Requirement as PkgRequirement
from packaging.version import InvalidVersion, Version

logger = logging.getLogger("DepsScan")

ECOSYSTEM = "PyPI"

# Tipos de rango resolubles contra una versión de PyPI. GIT queda afuera a propósito.
_RESOLVABLE_RANGE_TYPES = {"ECOSYSTEM", "SEMVER"}

# Los advisories de paquetes maliciosos (typosquatting) usan este prefijo y son otra
# clase de hallazgo: no tienen rangos de versión que resolver. Ver el cierre de
# docs/escaneo_dependencias.md.
_MALICIOUS_PREFIX = "MAL"

# Líneas de requirements.txt que no son una dependencia (flags de pip, includes, etc.).
_OPTION_LINE = re.compile(r"^\s*-")
_COMMENT = re.compile(r"(?:^|\s)#.*$")


def normalize_name(name: str) -> str:
    """Normalización PEP 503: `Foo.Bar_baz` y `foo-bar-baz` son el mismo paquete."""
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


# ======================================================================
# Parseo del manifiesto
# ======================================================================
@dataclass(frozen=True)
class Requirement:
    """Una línea de requirements.txt ya interpretada.

    `version` sólo se completa cuando hay un pin exacto (`==`), que es lo único que
    permite resolver rangos sin ambigüedad. Las demás líneas se devuelven con
    `skip_reason` en vez de descartarse en silencio: el escaneo tiene que poder
    decir *qué no miró*.
    """

    raw_name: str
    name: str
    version: str | None = None
    line_number: int = 0
    marker: str | None = None
    skip_reason: str | None = None

    @property
    def is_scannable(self) -> bool:
        return self.version is not None and self.skip_reason is None


def parse_requirements(text: str) -> list[Requirement]:
    """requirements.txt -> lista de Requirement, en orden de aparición.

    Devuelve TODAS las líneas de dependencia, incluidas las que no se pueden escanear
    (sin pin exacto, includes `-r`, instalaciones editables). El llamador filtra con
    `is_scannable`. Ver "Limitaciones honestas" en docs/escaneo_dependencias.md: el
    soporte arranca en `==` a propósito.
    """
    out: list[Requirement] = []

    # Las barras invertidas continúan la línea lógica; se unen antes de parsear para
    # que el número de línea reportado siga siendo el del inicio de la dependencia.
    logical: list[tuple[int, str]] = []
    buffer, start = "", 0
    for number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.rstrip()
        if not buffer:
            start = number
        if stripped.endswith("\\"):
            buffer += stripped[:-1]
            continue
        logical.append((start, buffer + stripped))
        buffer = ""
    if buffer:
        logical.append((start, buffer))

    for number, raw in logical:
        # En requirements.txt un '#' abre comentario al principio de línea o
        # precedido por espacio (no dentro de un especificador).
        line = _COMMENT.sub("", raw).strip()
        if not line:
            continue

        if _OPTION_LINE.match(line):
            out.append(
                Requirement(
                    raw_name=line,
                    name="",
                    line_number=number,
                    skip_reason="opción de pip o include (-r/-c/-e): no es una dependencia fijada",
                )
            )
            continue

        try:
            parsed = PkgRequirement(line)
        except InvalidRequirement as exc:
            out.append(
                Requirement(
                    raw_name=line,
                    name="",
                    line_number=number,
                    skip_reason=f"línea no parseable como dependencia ({exc})",
                )
            )
            continue

        if parsed.url:
            out.append(
                Requirement(
                    raw_name=parsed.name,
                    name=normalize_name(parsed.name),
                    line_number=number,
                    skip_reason="instalada desde URL: no hay versión de PyPI que resolver",
                )
            )
            continue

        pins = [s.version for s in parsed.specifier if s.operator == "=="]
        # `==1.2.*` no fija una versión: se descarta como pin.
        pins = [p for p in pins if "*" not in p]
        marker = str(parsed.marker) if parsed.marker else None

        if len(pins) != 1:
            specifier = str(parsed.specifier) or "sin especificador"
            out.append(
                Requirement(
                    raw_name=parsed.name,
                    name=normalize_name(parsed.name),
                    line_number=number,
                    marker=marker,
                    skip_reason=f"sin pin exacto ({specifier}): el resolver sólo soporta '=='",
                )
            )
            continue

        out.append(
            Requirement(
                raw_name=parsed.name,
                name=normalize_name(parsed.name),
                version=pins[0],
                line_number=number,
                marker=marker,
            )
        )

    return out


# ======================================================================
# Índice de advisories
# ======================================================================
def load_osv_zip(path: str | Path, include_malicious: bool = False) -> Iterator[dict]:
    """Itera los advisories del `all.zip` de OSV para PyPI, uno por JSON.

    Los registros `MAL` (paquetes maliciosos) se saltean por defecto: no traen rangos
    de versión y son una clase de hallazgo distinta.
    """
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".json"):
                continue
            try:
                advisory = json.loads(archive.read(name))
            except json.JSONDecodeError:
                logger.warning("Advisory ilegible en el zip de OSV: %s", name)
                continue
            if not include_malicious and str(advisory.get("id", "")).startswith(_MALICIOUS_PREFIX):
                continue
            yield advisory


def build_index(advisories: Iterable[dict], include_withdrawn: bool = False) -> dict[str, list[dict]]:
    """Advisories -> índice `nombre PyPI normalizado -> advisories que lo mencionan`.

    Es el lookup que reemplaza al retrieval: la clave es el nombre del paquete, que es
    exactamente lo que uno tiene al mirar un requirements.txt. Un advisory que afecta
    a varios paquetes se indexa bajo cada uno.

    Los advisories con `withdrawn` quedan afuera por defecto: OSV los marca así cuando
    se retractan o se publican duplicados, y contarlos infla los hallazgos.
    """
    index: dict[str, list[dict]] = {}
    for advisory in advisories:
        if not include_withdrawn and advisory.get("withdrawn"):
            continue
        for affected in advisory.get("affected", []):
            package = affected.get("package", {})
            if package.get("ecosystem") != ECOSYSTEM:
                continue
            name = package.get("name")
            if not name:
                continue
            bucket = index.setdefault(normalize_name(name), [])
            # Un advisory puede repetir el mismo paquete en varias entradas `affected`.
            if not bucket or bucket[-1] is not advisory:
                bucket.append(advisory)
    return index


# ======================================================================
# Lógica de rangos del esquema OSV
# ======================================================================
def _parse(version: str) -> Version | None:
    try:
        return Version(version)
    except (InvalidVersion, TypeError):
        return None


def _affected_entries(advisory: dict, package: str) -> Iterator[dict]:
    for affected in advisory.get("affected", []):
        pkg = affected.get("package", {})
        if pkg.get("ecosystem") != ECOSYSTEM:
            continue
        if normalize_name(pkg.get("name", "")) == package:
            yield affected


def _event_sort_key(event: dict) -> tuple[Version, int]:
    """Ordena los eventos de un rango por versión.

    OSV no garantiza que vengan ordenados y el algoritmo depende de recorrerlos en
    orden. `introduced: "0"` es el mínimo absoluto, y ante la misma versión un
    `introduced` va antes que el `fixed` que lo cierra.
    """
    for kind, rank in (("introduced", 0), ("fixed", 1), ("last_affected", 1), ("limit", 1)):
        if kind in event:
            raw = event[kind]
            if raw == "0":
                return (Version("0"), rank)
            parsed = _parse(raw)
            # Un evento con versión ilegible se manda al final para no alterar el
            # estado de los que sí se entienden.
            return (parsed or Version("99999"), rank)
    return (Version("0"), 0)


def _range_affects(range_: dict, version: Version) -> bool:
    """Aplica el algoritmo de eventos de OSV a un rango.

    Se recorre en orden de versión alternando el estado: `introduced` abre el intervalo
    de vulnerabilidad, `fixed` / `last_affected` / `limit` lo cierran.
    """
    if range_.get("type") not in _RESOLVABLE_RANGE_TYPES:
        return False

    affected = False
    for event in sorted(range_.get("events", []), key=_event_sort_key):
        if "introduced" in event:
            introduced = event["introduced"]
            if introduced == "0":
                affected = True
            else:
                parsed = _parse(introduced)
                if parsed is not None and version >= parsed:
                    affected = True
        elif "fixed" in event:
            parsed = _parse(event["fixed"])
            if parsed is not None and version >= parsed:
                affected = False
        elif "last_affected" in event:
            parsed = _parse(event["last_affected"])
            if parsed is not None and version > parsed:
                affected = False
        elif "limit" in event:
            parsed = _parse(event["limit"])
            if parsed is not None and version >= parsed:
                affected = False
    return affected


def advisory_affects(advisory: dict, package: str, version: str) -> bool:
    """¿Este advisory afecta a `package` en esta versión exacta?

    `package` se espera ya normalizado (PEP 503). Se consulta tanto la enumeración
    explícita `versions[]` como los rangos: OSV usa ambas y no son excluyentes.
    """
    parsed_version = _parse(version)
    if parsed_version is None:
        return False

    for affected in _affected_entries(advisory, package):
        for listed in affected.get("versions", []) or []:
            # Comparación PEP 440, no de strings: "1.0" y "1.0.0" son la misma versión.
            if _parse(listed) == parsed_version:
                return True
        for range_ in affected.get("ranges", []) or []:
            if _range_affects(range_, parsed_version):
                return True
    return False


def fixed_version(advisory: dict, package: str, current: str) -> str | None:
    """Menor versión `fixed` estrictamente mayor que la instalada.

    El filtro `> current` es lo que evita proponer un downgrade: los advisories viejos
    de un paquete traen `fixed` anteriores a lo que el usuario ya tiene instalado.
    Devuelve None si el advisory no publica arreglo aplicable (p. ej. sólo trae
    `last_affected`).
    """
    installed = _parse(current)
    if installed is None:
        return None

    candidates: list[Version] = []
    for affected in _affected_entries(advisory, package):
        for range_ in affected.get("ranges", []) or []:
            if range_.get("type") not in _RESOLVABLE_RANGE_TYPES:
                continue
            for event in range_.get("events", []):
                if "fixed" not in event:
                    continue
                parsed = _parse(event["fixed"])
                if parsed is not None and parsed > installed:
                    candidates.append(parsed)
    return str(min(candidates)) if candidates else None


# ======================================================================
# Hallazgos
# ======================================================================
@dataclass
class Vulnerability:
    """Una vulnerabilidad única que afecta a una dependencia instalada.

    Es *por CVE*, no por advisory: `osv_ids` guarda todos los advisories que la
    reportan. `epss` y `kev` los completa prioritize.py.
    """

    package: str
    installed_version: str
    cve: str | None
    osv_ids: list[str] = field(default_factory=list)
    cvss_vector: str | None = None
    cvss_score: float | None = None
    cvss_version: str | None = None
    cwe_ids: list[str] = field(default_factory=list)
    fixed_version: str | None = None
    summary: str = ""
    details: str = ""
    aliases: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    epss: float = 0.0
    kev: bool = False

    @property
    def identifier(self) -> str:
        """Identificador preferido para mostrar: el CVE si existe, si no el OSV."""
        return self.cve or (self.osv_ids[0] if self.osv_ids else "?")


def _cvss_entry(advisory: dict) -> tuple[str | None, str | None]:
    """(vector, tipo) de severidad. Prefiere CVSS_V3, que es el que sabemos puntuar.

    Sobre el dump real: 69% de los advisories traen V3, 7,3% traen sólo V4 y 23,8% no
    traen severidad. Los que no dan V3 se guardan igual con su tipo, para que el
    consumidor sepa que el score falta por falta de dato y no por un error.
    """
    severities = advisory.get("severity", []) or []
    for wanted in ("CVSS_V3", "CVSS_V4"):
        for entry in severities:
            if entry.get("type") == wanted and entry.get("score"):
                return entry["score"], wanted
    return None, None


def _cve_of(advisory: dict) -> str | None:
    for alias in advisory.get("aliases", []) or []:
        if str(alias).startswith("CVE-"):
            return alias
    return None


def scan(requirements: Iterable[Requirement], index: dict[str, list[dict]]) -> list[Vulnerability]:
    """requirements + índice OSV -> vulnerabilidades únicas que afectan esas versiones.

    Deduplica por (paquete, CVE) uniendo los datos de los advisories que reportan el
    mismo fallo (ver el docstring del módulo). Los advisories sin alias a CVE quedan
    como hallazgos propios, identificados por su id de OSV: no se pueden cruzar con
    EPSS ni KEV, pero tampoco se pierden.
    """
    findings: dict[tuple[str, str], Vulnerability] = {}

    for requirement in requirements:
        if not requirement.is_scannable:
            continue
        package, version = requirement.name, requirement.version

        for advisory in index.get(package, []):
            if not advisory_affects(advisory, package, version):
                continue

            osv_id = advisory.get("id", "")
            cve = _cve_of(advisory)
            vector, cvss_version = _cvss_entry(advisory)
            cwe_ids = advisory.get("database_specific", {}).get("cwe_ids") or []
            fixed = fixed_version(advisory, package, version)
            key = (package, cve or osv_id)

            existing = findings.get(key)
            if existing is None:
                findings[key] = Vulnerability(
                    package=package,
                    installed_version=version,
                    cve=cve,
                    osv_ids=[osv_id],
                    cvss_vector=vector,
                    cvss_version=cvss_version,
                    cwe_ids=list(dict.fromkeys(cwe_ids)),
                    fixed_version=fixed,
                    summary=advisory.get("summary", "") or "",
                    details=advisory.get("details", "") or "",
                    aliases=list(advisory.get("aliases", []) or []),
                    references=[r.get("url") for r in advisory.get("references", []) or [] if r.get("url")],
                )
                continue

            # Mismo CVE por otro advisory: se completa lo que falte en vez de duplicar.
            if osv_id not in existing.osv_ids:
                existing.osv_ids.append(osv_id)
            existing.cwe_ids = list(dict.fromkeys(existing.cwe_ids + list(cwe_ids)))
            existing.aliases = list(dict.fromkeys(existing.aliases + list(advisory.get("aliases", []) or [])))
            existing.references = list(
                dict.fromkeys(
                    existing.references
                    + [r.get("url") for r in advisory.get("references", []) or [] if r.get("url")]
                )
            )
            # Un vector V3 desplaza a un V4 sin puntuar; si no, gana el primero que llegó.
            if vector and (existing.cvss_vector is None or (cvss_version == "CVSS_V3" and existing.cvss_version != "CVSS_V3")):
                existing.cvss_vector, existing.cvss_version = vector, cvss_version
            if fixed and (existing.fixed_version is None or Version(fixed) < Version(existing.fixed_version)):
                existing.fixed_version = fixed
            if not existing.summary:
                existing.summary = advisory.get("summary", "") or ""
            if not existing.details:
                existing.details = advisory.get("details", "") or ""

    return list(findings.values())
