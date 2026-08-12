from pathlib import Path
import re
import json
import hashlib
import logging

logger = logging.getLogger("IndexingRAG")  
CWE_RE = re.compile(r"^CWE-\d+$") 
class XMLCWEConverter:
    """XML de MITRE (esquema CWE, namespace cwe-7) → un Document por Weakness."""

    def run(self, sources: list[Path]):
        import xml.etree.ElementTree as ET
        from haystack import Document

        documents = []
        skipped = 0
        ns = {'cwe': 'http://cwe.mitre.org/cwe-7'}

        for file_path in sources:
            try:
                tree = ET.parse(file_path)
            except ET.ParseError as e:
                logger.error(f"[XMLCWE] No se pudo parsear {file_path.name}: {e}")
                continue

            root = tree.getroot()

            for weakness in root.findall('.//cwe:Weakness', ns):
                cwe_id = weakness.get('ID')
                name = weakness.get('Name')

                # Si falta el ID, no podemos generar un doc.id único ni confiable — se descarta
                if not cwe_id:
                    skipped += 1
                    continue

                desc_elem = weakness.find('cwe:Description', ns)
                description = self._extract_text(desc_elem)

                # Fallback: si Description viene vacía, probamos con Extended_Description
                if not description:
                    ext_elem = weakness.find('cwe:Extended_Description', ns)
                    description = self._extract_text(ext_elem)

                name = (name or "").strip()
                description = (description or "").strip()

                content = f"Vulnerabilidad CWE-{cwe_id}: {name}\nDescripción: {description}".strip()

                # Filtro de calidad: descarta chunks sin sustancia real
                # (sin nombre Y sin descripción, o contenido alfanumérico insuficiente)
                alnum_chars = sum(c.isalnum() for c in content)
                if alnum_chars < 20:
                    logger.warning(
                        f"[XMLCWE] Descartado CWE-{cwe_id} por contenido insuficiente "
                        f"(alnum_chars={alnum_chars}): '{content[:80]}'"
                    )
                    skipped += 1
                    continue

                doc = Document(
                    id=f"cwe-{cwe_id}",
                    content=content,
                    meta={
                        "source_type": "cwe_weakness",
                        "source": f"CWE-{cwe_id}",
                        "cwe_id": int(cwe_id) if cwe_id.isdigit() else cwe_id,
                        "name": name or "Sin nombre",
                        "source_file": str(file_path),
                    }
                )
                documents.append(doc)

        logger.info(
            f"[XMLCWE] {len(documents)} documentos CWE generados, "
            f"{skipped} descartados por datos insuficientes."
        )
        return {"documents": documents}

    @staticmethod
    def _extract_text(elem) -> str:
        """
        Extrae todo el texto de un elemento XML, incluyendo el de sub-tags anidados
        (ej. <xhtml:p>, <xhtml:div> dentro de <Description>). A diferencia de
        elem.text (que solo trae el texto directo del nodo), esto recorre todo
        el árbol y concatena cada fragmento de texto encontrado.
        """
        if elem is None:
            return ""
        parts = [t.strip() for t in elem.itertext() if t and t.strip()]
        return " ".join(parts)

def _english_description(cve: dict) -> str:
    """Descripción en inglés (lo único que se embebe). Fallback: la primera que haya."""
    descriptions = cve.get("descriptions", [])
    for d in descriptions:
        if d.get("lang") == "en":
            return d.get("value", "")
    return descriptions[0].get("value", "") if descriptions else ""


def _cvss(metrics: dict) -> dict:
    """Extrae score/vector/severity. Prioriza v3.1 > v3.0 para 'v3', guarda v2 aparte.

    Cada métrica en el JSON de NVD es una LISTA; tomamos el primer elemento
    (habitualmente el 'Primary' de nvd@nist.gov).
    """
    out = {
        "cvss_v3_score": None, "cvss_v3_vector": None,
        "cvss_v2_score": None, "cvss_v2_vector": None,
        "severity": None,
    }
    for key in ("cvssMetricV31", "cvssMetricV30"):
        entries = metrics.get(key)
        if entries:
            data = entries[0].get("cvssData", {})
            out["cvss_v3_score"] = data.get("baseScore")
            out["cvss_v3_vector"] = data.get("vectorString")
            out["severity"] = data.get("baseSeverity") or entries[0].get("baseSeverity")
            break
    entries_v2 = metrics.get("cvssMetricV2")
    if entries_v2:
        data = entries_v2[0].get("cvssData", {})
        out["cvss_v2_score"] = data.get("baseScore")
        out["cvss_v2_vector"] = data.get("vectorString")
        if out["severity"] is None:
            out["severity"] = entries_v2[0].get("baseSeverity")
    return out



def _vendors_products(cve: dict) -> tuple[list[str], list[str]]:
    """Vendors y productos únicos, parseados de los CPE de configurations.
    Vendors: Ej. "microsoft", "adobe", "apache".
    Products: Ej. "windows_10", "acrobat_reader", "http_server".
    Los CPE son strings con formato estandarizado: cpe:2.3:<part>:<vendor>:<product>:<version>:...

    CPE 2.3 = cpe:2.3:<part>:<vendor>:<product>:<version>:...  (índices 3 y 4).
    """
    vendors, products = set(), set()
    for config in cve.get("configurations", []):
        for node in config.get("nodes", []):
            for match in node.get("cpeMatch", []):
                parts = match.get("criteria", "").split(":")
                if len(parts) > 4:
                    vendor, product = parts[3], parts[4]
                    if vendor and vendor != "*":
                        vendors.add(vendor)
                    if product and product != "*":
                        products.add(product)
    return sorted(vendors), sorted(products)


def _cwe_ids(cve: dict) -> list[str]:
    ids = set()
    for weakness in cve.get("weaknesses", []):
        for desc in weakness.get("description", []):
            value = desc.get("value", "")
            if CWE_RE.match(value):
                ids.add(value)
    return sorted(ids)

def cve_to_document(cve: dict):
    """Convierte un objeto 'cve' crudo de NVD en un Document de Haystack.

    Devuelve None si el CVE no tiene descripción utilizable (nada que embeber).
    """
    from haystack import Document

    cve_id = cve.get("id")
    description = _english_description(cve)
    if not cve_id or not description.strip():
        return None

    vendors, products = _vendors_products(cve)
    meta = {
        "source_type": "nvd_cve",
        "source": cve_id,
        "cve_id": cve_id,
        "published_date": cve.get("published"),
        "last_modified": cve.get("lastModified"),
        "vendors": vendors,
        "products": products,
        "cwe_ids": _cwe_ids(cve),
        "references": [r.get("url") for r in cve.get("references", []) if r.get("url")],
        **_cvss(cve.get("metrics", {})),
    }
    # id determinístico: reindexar el mismo CVE actualiza la fila, no la duplica.
    doc_id = hashlib.sha256(cve_id.encode()).hexdigest()
    return Document(id=doc_id, content=description, meta=meta)

class NVDJsonConverter:
    """Páginas JSON crudas de la NVD (cves_page_*.json de fetch_nvd.py) → un
    Document atómico por CVE. Ante un CVE repetido entre páginas, gana el de
    lastModified más reciente."""

    def run(self, sources: list[Path]):
        by_id: dict[str, dict] = {}
        for path in sources:
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.error(f"[NVD] No se pudo leer {Path(path).name}: {e}")
                continue
            for entry in data.get("vulnerabilities", []):
                cve = entry.get("cve", {})
                cve_id = cve.get("id")
                if not cve_id:
                    continue
                prev = by_id.get(cve_id)
                if prev is None or cve.get("lastModified", "") >= prev.get("lastModified", ""):
                    by_id[cve_id] = cve

        documents = []
        skipped = 0
        for cve in by_id.values():
            doc = cve_to_document(cve)
            if doc is None:
                skipped += 1
                continue
            documents.append(doc)

        logger.info(
            f"[NVD] {len(documents)} documentos CVE generados, "
            f"{skipped} descartados por falta de descripción."
        )
        return {"documents": documents}

