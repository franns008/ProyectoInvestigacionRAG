"""Fixtures compartidas de los tests del escaneo de dependencias.

`src/pipeline` se agrega al path porque es el directorio que el container monta como
`/app/pipelines`: dentro de Docker, `deps` se importa igual que acá.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "pipeline"))

FIXTURES = Path(__file__).parent / "fixtures"
REQUIREMENTS = Path(__file__).parent / "test_requirements"


@pytest.fixture(scope="session")
def advisories() -> list[dict]:
    """Los advisories de tests/fixtures/osv, tal cual los publica OSV.

    Están congelados a propósito: el dump real cambia todos los días y un test que
    dependiera de él no sería reproducible.
    """
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted((FIXTURES / "osv").glob("*.json"))]


@pytest.fixture(scope="session")
def index(advisories):
    from deps.resolver import build_index

    return build_index(advisories)


@pytest.fixture(scope="session")
def epss():
    from deps.prioritize import load_epss

    return load_epss(FIXTURES / "epss_sample.csv")


@pytest.fixture(scope="session")
def kev():
    from deps.prioritize import load_kev

    return load_kev(FIXTURES / "kev_sample.json")


@pytest.fixture
def scan_manifest(index):
    """Escanea un requirements.txt de tests/test_requirements y devuelve los hallazgos."""
    from deps.resolver import parse_requirements, scan

    def run(filename: str):
        text = (REQUIREMENTS / filename).read_text(encoding="utf-8")
        return scan(parse_requirements(text), index)

    return run
