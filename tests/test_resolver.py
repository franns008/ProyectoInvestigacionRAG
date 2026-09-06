"""
Tests del resolver: dado un requirements.txt fijo, salen exactamente estas CVE.

Es el único pedazo del sistema que se *verifica* en vez de medirse estadísticamente
(ver docs/escaneo_dependencias.md). Corre en segundos, sin stack, sin GPU y sin LLM.
"""

import pytest

from deps.resolver import (
    advisory_affects,
    build_index,
    fixed_version,
    normalize_name,
    parse_requirements,
    scan,
)


# ======================================================================
# Normalización PEP 503
# ======================================================================
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Pillow", "pillow"),
        ("Flask-SQLAlchemy", "flask-sqlalchemy"),
        ("zope.interface", "zope-interface"),
        ("ruamel_yaml", "ruamel-yaml"),
        ("Foo..__Bar", "foo-bar"),
        ("  requests  ", "requests"),
    ],
)
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


# ======================================================================
# Parseo del manifiesto
# ======================================================================
def test_parsea_pin_exacto():
    (req,) = parse_requirements("pillow==5.2.0\n")
    assert (req.name, req.version, req.is_scannable) == ("pillow", "5.2.0", True)


def test_ignora_comentarios_y_lineas_vacias():
    text = "# comentario\n\npillow==5.2.0  # al final de la línea\n"
    assert [r.name for r in parse_requirements(text)] == ["pillow"]


def test_une_lineas_continuadas():
    (req,) = parse_requirements("pillow== \\\n5.2.0\n")
    assert req.version == "5.2.0"


def test_reporta_numero_de_linea():
    reqs = parse_requirements("# nota\ndjango==4.2.0\npillow==5.2.0\n")
    assert [(r.name, r.line_number) for r in reqs] == [("django", 2), ("pillow", 3)]


def test_lo_no_escaneable_vuelve_con_motivo():
    """Nada se descarta en silencio: el escaneo tiene que poder decir qué no miró."""
    text = "django>=2.2\npillow==5.2.*\n-r otro.txt\n--index-url https://x\n"
    reqs = parse_requirements(text)
    assert len(reqs) == 4
    assert all(not r.is_scannable and r.skip_reason for r in reqs)


def test_extras_y_markers_se_escanean():
    """`celery[redis]==5.3.0` fija una versión: los extras no cambian eso."""
    reqs = parse_requirements('celery[redis]==5.3.0\nurllib3==1.24.1 ; python_version < "3.10"\n')
    assert all(r.is_scannable for r in reqs)
    assert [r.name for r in reqs] == ["celery", "urllib3"]
    assert reqs[1].marker is not None


# ======================================================================
# Lógica de rangos de OSV
# ======================================================================
def test_rango_introduced_fixed(advisories):
    """pillow: [0, 10.0.1) — el borde `fixed` NO está afectado."""
    ghsa = next(a for a in advisories if a["id"] == "GHSA-j7hp-h8jx-5ppr")
    assert advisory_affects(ghsa, "pillow", "5.2.0")
    assert advisory_affects(ghsa, "pillow", "10.0.0")
    assert not advisory_affects(ghsa, "pillow", "10.0.1")
    assert not advisory_affects(ghsa, "pillow", "11.0.0")


def test_rango_last_affected_incluye_el_borde(advisories):
    """mezzanine: `last_affected: 6.0.0` — 6.0.0 SÍ está afectada, 6.0.1 no.

    Es la diferencia con `fixed`, y es un off-by-one fácil de cometer.
    """
    ghsa = next(a for a in advisories if a["id"] == "GHSA-22cc-w7xm-rfhx")
    assert advisory_affects(ghsa, "mezzanine", "6.0.0")
    assert not advisory_affects(ghsa, "mezzanine", "6.0.1")


def test_eventos_intercalados_en_un_solo_rango(advisories):
    """PYSEC-2026-628 mete dos intervalos en un mismo `ranges[]`:

        introduced 2.0a1 -> fixed 2.1.9 -> introduced 2.2a1 -> fixed 2.2.2

    django 2.1.9 cae en el hueco reparado entre ambos y NO está afectada.
    """
    pysec = next(a for a in advisories if a["id"] == "PYSEC-2026-628")
    assert advisory_affects(pysec, "django", "2.0.0")
    assert not advisory_affects(pysec, "django", "2.1.9")
    assert not advisory_affects(pysec, "django", "2.1.15")
    assert advisory_affects(pysec, "django", "2.2.0")
    assert not advisory_affects(pysec, "django", "2.2.2")


def test_ignora_rangos_git(advisories):
    """PYSEC-2020-35 trae un rango GIT cuyo `fixed` es un hash de commit.

    Si no se filtrara por tipo, `Version("eb31d84...")` reventaría el parseo.
    """
    pysec = next(a for a in advisories if a["id"] == "PYSEC-2020-35")
    assert any(r["type"] == "GIT" for x in pysec["affected"] for r in x.get("ranges", []))
    assert advisory_affects(pysec, "django", "2.2.0")
    assert not advisory_affects(pysec, "django", "3.0.3")


def test_ignora_otros_ecosistemas(advisories):
    """GHSA-j7hp-h8jx-5ppr afecta a electron, SkiaSharp, npm... y a pillow.

    El paquete npm `webp` no debe contaminar el índice de PyPI.
    """
    ghsa = next(a for a in advisories if a["id"] == "GHSA-j7hp-h8jx-5ppr")
    ecosistemas = {x["package"]["ecosystem"] for x in ghsa["affected"]}
    assert len(ecosistemas) > 1
    assert build_index([ghsa]).keys() == {"pillow"}
    assert not advisory_affects(ghsa, "webp", "0.1.0")


def test_excluye_advisories_retirados(advisories):
    """GHSA-56pw-mpj4-fxww está `withdrawn` y no tiene aliases.

    Sin filtrarlo aparecería como un hallazgo extra de pillow —con el texto
    "Duplicate Advisory" a la vista— porque la deduplicación por CVE no puede
    emparejarlo con CVE-2023-4863 al no declarar el alias.
    """
    retirado = next(a for a in advisories if a["id"] == "GHSA-56pw-mpj4-fxww")
    assert retirado.get("withdrawn")
    assert retirado.get("aliases") is None

    assert "pillow" not in build_index([retirado])
    assert build_index([retirado], include_withdrawn=True).keys() == {"pillow"}


def test_version_invalida_no_rompe(advisories):
    ghsa = next(a for a in advisories if a["id"] == "GHSA-j7hp-h8jx-5ppr")
    assert not advisory_affects(ghsa, "pillow", "no-es-una-version")


# ======================================================================
# Versión de arreglo
# ======================================================================
def test_fixed_version_no_propone_downgrade(advisories):
    """El caso que rompe la implementación ingenua.

    CVE-2019-11358 publica `fixed` en 2.1.9 y en 2.2.2. Para django 2.2.0 el mínimo
    global es 2.1.9 — que es ANTERIOR a lo instalado. La respuesta correcta es 2.2.2.
    """
    ghsa = next(a for a in advisories if a["id"] == "GHSA-6c3j-c64m-qhgq")
    assert fixed_version(ghsa, "django", "2.2.0") == "2.2.2"
    assert fixed_version(ghsa, "django", "2.0.5") == "2.1.9"


def test_fixed_version_toma_el_menor_aplicable(advisories):
    """CVE-2020-7471 arregla en 1.11.28, 2.2.10 y 3.0.3: para 2.2.0 corresponde 2.2.10."""
    ghsa = next(a for a in advisories if a["id"] == "GHSA-hmr4-m2h5-33qx")
    assert fixed_version(ghsa, "django", "2.2.0") == "2.2.10"


def test_sin_fixed_devuelve_none(advisories):
    """mezzanine sólo tiene `last_affected`: no hay versión de arreglo publicada."""
    ghsa = next(a for a in advisories if a["id"] == "GHSA-22cc-w7xm-rfhx")
    assert fixed_version(ghsa, "mezzanine", "6.0.0") is None


# ======================================================================
# Escaneo end-to-end contra los manifiestos
# ======================================================================
def test_1vuln(scan_manifest):
    (vuln,) = scan_manifest("requirements_1vuln.txt")
    assert vuln.package == "pillow"
    assert vuln.installed_version == "5.2.0"
    assert vuln.cve == "CVE-2023-4863"
    assert vuln.fixed_version == "10.0.1"
    assert vuln.cwe_ids == ["CWE-787"]


def test_1de3vuln(scan_manifest):
    """Tres dependencias, una vulnerable. Las otras dos no deben aparecer."""
    vulns = scan_manifest("requirements_1de3vuln.txt")
    assert [v.package for v in vulns] == ["pillow"]


def test_3vuln(scan_manifest):
    """Tres paquetes vulnerables y CUATRO hallazgos: django aporta dos CVE."""
    vulns = scan_manifest("requirements_3vuln.txt")
    assert {v.package for v in vulns} == {"django", "mezzanine", "pillow"}
    assert {v.identifier for v in vulns} == {
        "CVE-2019-11358",
        "CVE-2020-7471",
        "CVE-2024-25170",
        "CVE-2023-4863",
    }


def test_manifiesto_sin_pin_no_produce_hallazgos(scan_manifest):
    """Lo no escaneable no se escanea, pero tampoco rompe."""
    assert scan_manifest("requirements_sin_pin.txt") == []


# ======================================================================
# Deduplicación por CVE
# ======================================================================
def test_dedup_colapsa_advisories_del_mismo_cve(scan_manifest, advisories):
    """GHSA y PYSEC reportan CVE-2023-4863 por separado: es UN hallazgo, no dos."""
    (vuln,) = scan_manifest("requirements_1vuln.txt")
    assert sorted(vuln.osv_ids) == ["GHSA-j7hp-h8jx-5ppr", "PYSEC-2026-1794"]


def test_dedup_une_los_cwe_del_advisory_hermano(scan_manifest):
    """El hallazgo que justifica deduplicar.

    PYSEC-2026-628 no trae `cwe_ids` (ningún PYSEC lo hace) y GHSA-6c3j-c64m-qhgq sí.
    Al unirlos, la cadena hasta el CWE se recupera para los dos.
    """
    vulns = scan_manifest("requirements_3vuln.txt")
    django_xss = next(v for v in vulns if v.cve == "CVE-2019-11358")
    assert sorted(django_xss.cwe_ids) == ["CWE-1321", "CWE-79"]
    assert len(django_xss.osv_ids) == 2


def test_dedup_conserva_la_menor_version_de_arreglo(scan_manifest):
    vulns = scan_manifest("requirements_3vuln.txt")
    sqli = next(v for v in vulns if v.cve == "CVE-2020-7471")
    assert sqli.fixed_version == "2.2.10"


def test_dedup_prefiere_el_vector_v3(scan_manifest):
    """GHSA-hmr4-m2h5-33qx trae vector v3 y v4; el puntuable es el v3."""
    vulns = scan_manifest("requirements_3vuln.txt")
    sqli = next(v for v in vulns if v.cve == "CVE-2020-7471")
    assert sqli.cvss_version == "CVSS_V3"
    assert sqli.cvss_vector.startswith("CVSS:3.1/")


def test_hallazgo_sin_severidad_no_inventa_datos(scan_manifest):
    vulns = scan_manifest("requirements_3vuln.txt")
    mezzanine = next(v for v in vulns if v.package == "mezzanine")
    assert mezzanine.cvss_vector is None
    assert mezzanine.cwe_ids == []
    assert mezzanine.fixed_version is None
