"""
Tests de la priorización: CVSS calculado desde el vector, y el orden KEV > EPSS > CVSS.

Los scores de referencia son los que publica NVD para esas mismas CVE, así que el
test verifica la implementación de CVSS 3.1 contra la fuente autoritativa y no contra
sí misma.
"""

import pytest

from deps.prioritize import cvss31_base_score, funnel, load_epss, load_kev, prioritize
from deps.resolver import Vulnerability


# ======================================================================
# CVSS 3.1 desde el vector
# ======================================================================
@pytest.mark.parametrize(
    "vector, expected, caso",
    [
        ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", 9.8, "CVE-2020-7471 (SQLi en django)"),
        ("CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:H/I:H/A:H", 8.8, "CVE-2023-4863 (libwebp)"),
        ("CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N", 6.1, "CVE-2019-11358 (XSS, scope changed)"),
        ("CVSS:3.1/AV:N/AC:H/PR:N/UI:R/S:C/C:H/I:N/A:N", 6.1, "CVE-2023-32681 (requests)"),
        ("CVSS:3.1/AV:L/AC:H/PR:H/UI:R/S:U/C:N/I:N/A:N", 0.0, "impacto nulo -> score 0"),
        ("CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", 9.8, "v3.0 usa la misma fórmula"),
    ],
)
def test_cvss31_base_score(vector, expected, caso):
    assert cvss31_base_score(vector) == expected, caso


@pytest.mark.parametrize(
    "vector",
    [
        None,
        "",
        "CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N",  # v4: no se puntúa
        "CVSS:2.0/AV:N/AC:L/Au:N/C:P/I:P/A:P",                              # v2: tampoco
        "CVSS:3.1/AV:N/AC:L",                                               # incompleto
        "CVSS:3.1/AV:X/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",                     # métrica inválida
        "basura",
    ],
)
def test_cvss31_devuelve_none_en_vez_de_inventar(vector):
    """Un score que no se puede calcular es None, nunca un 0.0 que ordenaría mal."""
    assert cvss31_base_score(vector) is None


def test_roundup_redondea_hacia_arriba():
    """CVSS 3.1 define Roundup(), que NO es el round() de Python.

    Este vector da 3.6335... crudo. Roundup() exige 3.7 porque redondea siempre hacia
    arriba al siguiente decimal; un round() común devolvería 3.6 y el score quedaría
    por debajo del que publica NVD.
    """
    assert cvss31_base_score("CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:L/I:N/A:N") == 3.7


# ======================================================================
# Carga de EPSS y KEV
# ======================================================================
def test_load_epss_saltea_la_metadata(epss):
    assert epss["CVE-2023-4863"] == 1.0
    assert epss["CVE-2019-11358"] == pytest.approx(0.87234)
    assert "cve" not in epss  # la fila de encabezado no es un CVE


def test_load_kev(kev):
    assert kev == {"CVE-2023-4863"}


# ======================================================================
# Orden
# ======================================================================
def _vuln(cve, vector=None, package="x"):
    return Vulnerability(package=package, installed_version="1.0", cve=cve, cvss_vector=vector)


def test_kev_gana_aunque_tenga_menos_cvss(epss, kev):
    """El corazón de la priorización.

    Una CVE con CVSS 9.8 pero sin exploit conocido va DESPUÉS de una con CVSS 8.8 que
    CISA confirma que se está explotando hoy. Es la diferencia entre qué tan grave
    sería y qué está siendo atacado ahora.
    """
    critica_sin_exploit = _vuln("CVE-2020-7471", "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H")
    en_kev = _vuln("CVE-2023-4863", "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:H/I:H/A:H")

    ordenadas = prioritize([critica_sin_exploit, en_kev], epss, kev={"CVE-2023-4863"})

    assert ordenadas[0].cve == "CVE-2023-4863"
    assert ordenadas[0].cvss_score == 8.8 and ordenadas[1].cvss_score == 9.8


def test_epss_desempata_antes_que_cvss(epss):
    """Sin nadie en KEV, manda la probabilidad de exploit."""
    alta_cvss = _vuln("CVE-2020-7471", "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H")  # 9.8, EPSS 0.65
    alto_epss = _vuln("CVE-2019-11358", "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N")  # 6.1, EPSS 0.87

    ordenadas = prioritize([alta_cvss, alto_epss], epss, kev=set())

    assert [v.cve for v in ordenadas] == ["CVE-2019-11358", "CVE-2020-7471"]


def test_cvss_desempata_sin_epss():
    a = _vuln("CVE-0000-0001", "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H")  # 9.8
    b = _vuln("CVE-0000-0002", "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N")  # 6.1

    assert [v.cve for v in prioritize([b, a])] == ["CVE-0000-0001", "CVE-0000-0002"]


def test_sin_score_no_se_promueve():
    """Un hallazgo sin CVSS calculable queda último: no se le supone gravedad."""
    con_score = _vuln("CVE-0000-0001", "CVSS:3.1/AV:L/AC:H/PR:H/UI:R/S:U/C:L/I:N/A:N")
    sin_score = _vuln("CVE-0000-0002", None)

    ordenadas = prioritize([sin_score, con_score])

    assert [v.cve for v in ordenadas] == ["CVE-0000-0001", "CVE-0000-0002"]
    assert ordenadas[1].cvss_score is None


def test_orden_estable_con_todo_empatado():
    """Mismos scores -> orden determinístico por paquete e identificador."""
    a = _vuln("CVE-0000-0002", package="bravo")
    b = _vuln("CVE-0000-0001", package="alfa")

    assert [v.package for v in prioritize([a, b])] == ["alfa", "bravo"]


def test_hallazgo_sin_cve_no_cruza_epss_ni_kev():
    """Los advisories sin alias a CVE no se pueden cruzar; no deben ensuciar el orden."""
    huerfano = Vulnerability(package="x", installed_version="1.0", cve=None, osv_ids=["PYSEC-9999-1"])

    (resultado,) = prioritize([huerfano], {"CVE-2023-4863": 1.0}, {"CVE-2023-4863"})

    assert resultado.epss == 0.0 and resultado.kev is False
    assert resultado.identifier == "PYSEC-9999-1"


# ======================================================================
# El embudo
# ======================================================================
def test_funnel(epss, kev, scan_manifest):
    """El embudo sobre el manifiesto de 3 vulnerables, con los datos de los fixtures."""
    vulns = prioritize(scan_manifest("requirements_3vuln.txt"), epss, kev)

    assert funnel(vulns) == {
        "total": 4,
        "paquetes": 3,     # django aporta dos CVE: hallazgos != paquetes
        "cvss_alto": 2,    # CVE-2020-7471 (9.8) y CVE-2023-4863 (8.8)
        "epss_alto": 3,
        "kev": 1,          # sólo CVE-2023-4863
        "sin_cvss": 1,     # mezzanine no trae severidad
        "sin_cve": 0,
    }
    assert vulns[0].cve == "CVE-2023-4863"
