"""
Tests del contexto del chat (src/pipeline/chat/). Corren sin stack: es lógica pura.
"""

import json

from chat import (
    extract_findings_context,
    finding_vuln_ids,
    findings_vuln_ids,
    format_finding,
    format_findings,
    select_history,
)

FINDING = {
    "vulnerability": "CVE-2023-4863",
    "package": "pillow",
    "installed_version": "5.2.0",
    "fixed_version": "10.0.1",
    "cwe_ids": ["CWE-787"],
    "aliases": ["GHSA-j7hp-h8jx-5ppr", "CVE-2023-4863"],
    "cvss_score": 8.8,
    "epss": 0.93,
    "kev": True,
    "summary": "Heap buffer overflow in libwebp",
    "details": "A crafted WebP image can write out of bounds.",
}


def conversation(*turns):
    return [{"role": "system", "content": json.dumps(FINDING)}, *turns]


# ======================================================================
# extract_findings_context
# ======================================================================
OTHER = {
    "vulnerability": "CVE-2019-11358",
    "package": "django",
    "installed_version": "2.2.0",
    "cwe_ids": ["CWE-79"],
    "summary": "XSS",
}


def attached(*findings):
    return {"role": "system", "content": json.dumps({"findings": list(findings)})}


def test_extrae_los_hallazgos_adjuntos():
    messages = [attached(FINDING, OTHER), {"role": "user", "content": "hola"}]
    assert extract_findings_context(messages) == [FINDING, OTHER]


def test_acepta_el_hallazgo_suelto_de_antes():
    assert extract_findings_context(conversation({"role": "user", "content": "hola"})) == [FINDING]


def test_no_repite_un_hallazgo_adjunto_dos_veces():
    messages = [attached(FINDING, OTHER), attached(FINDING)]
    assert extract_findings_context(messages) == [FINDING, OTHER]


def test_lista_de_adjuntos_vacia_no_trae_hallazgos():
    assert extract_findings_context([attached()]) == []


def test_descarta_lo_que_no_es_un_hallazgo_dentro_de_la_lista():
    messages = [{"role": "system", "content": json.dumps({"findings": [FINDING, {"foo": 1}, "x"]})}]
    assert extract_findings_context(messages) == [FINDING]


def test_sin_system_no_hay_hallazgo():
    assert extract_findings_context([{"role": "user", "content": json.dumps(FINDING)}]) == []
    assert extract_findings_context(None) == []


def test_un_system_prompt_de_texto_no_es_un_hallazgo():
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    assert extract_findings_context(messages) == []


def test_un_system_json_ajeno_no_es_un_hallazgo():
    messages = [{"role": "system", "content": json.dumps({"foo": 1})}]
    assert extract_findings_context(messages) == []


# ======================================================================
# finding_vuln_ids
# ======================================================================
def test_ids_canonicos_sin_duplicados_ni_ghsa():
    assert finding_vuln_ids(FINDING) == ["CVE-2023-4863", "CWE-787"]


def test_hallazgo_sin_cve_aporta_solo_los_cwe():
    finding = {"vulnerability": "GHSA-xxxx-yyyy-zzzz", "cve": None, "cwe_ids": ["CWE-79"]}
    assert finding_vuln_ids(finding) == ["CWE-79"]


def test_sin_hallazgo_no_hay_ids():
    assert finding_vuln_ids(None) == []


# ======================================================================
# format_finding
# ======================================================================
def test_formato_con_todos_los_campos():
    text = format_finding(FINDING)
    assert "- Identifier: CVE-2023-4863 (aliases: GHSA-j7hp-h8jx-5ppr)" in text
    assert "- Affected package: pillow 5.2.0 (fixed in 10.0.1)" in text
    assert "CVSS=8.8 | EPSS=0.93 | in CISA KEV: yes" in text
    assert "- Weakness: CWE-787" in text
    assert "Heap buffer overflow in libwebp" in text


def test_formato_con_lo_que_manda_la_extension_hoy():
    """Sin CVSS/EPSS/KEV/aliases (el contextFor actual) no aparece la línea de scores."""
    minimal = {k: FINDING[k] for k in ("vulnerability", "package", "installed_version", "cwe_ids", "summary")}
    minimal["fixed_version"] = None
    text = format_finding(minimal)
    assert "no fixed version published" in text
    assert "CVSS" not in text


def test_el_advisory_largo_se_recorta():
    text = format_finding({**FINDING, "details": "palabra " * 1000})
    assert text.endswith(" […]")
    assert len(text) < 1600


# ======================================================================
# select_history
# ======================================================================
def test_historial_sin_system_ni_turno_actual():
    messages = conversation(
        {"role": "user", "content": "¿qué tan grave es?"},
        {"role": "assistant", "content": "Es crítica."},
        {"role": "user", "content": "¿cómo lo mitigo?"},
    )
    assert select_history(messages, "¿cómo lo mitigo?") == [
        {"role": "user", "content": "¿qué tan grave es?"},
        {"role": "assistant", "content": "Es crítica."},
    ]


def test_primer_turno_no_tiene_historial():
    messages = conversation({"role": "user", "content": "¿qué es esto?"})
    assert select_history(messages, "¿qué es esto?") == []


def test_se_queda_con_los_ultimos_n():
    turns = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(10)]
    messages = conversation(*turns, {"role": "user", "content": "actual"})
    history = select_history(messages, "actual", max_messages=4)
    assert [t["content"] for t in history] == ["m6", "m7", "m8", "m9"]


def test_recorta_mensajes_largos():
    messages = conversation(
        {"role": "assistant", "content": "palabra " * 200},
        {"role": "user", "content": "actual"},
    )
    (only,) = select_history(messages, "actual", max_chars=50)
    assert len(only["content"]) <= 55
    assert only["content"].endswith(" […]")


def test_max_messages_cero_desactiva_el_historial():
    messages = conversation({"role": "user", "content": "a"}, {"role": "user", "content": "b"})
    assert select_history(messages, "b", max_messages=0) == []


# ======================================================================
# Varios hallazgos adjuntos
# ======================================================================
def test_ids_de_todos_los_adjuntos_sin_repetir():
    assert findings_vuln_ids([FINDING, OTHER, FINDING]) == ["CVE-2023-4863", "CWE-787", "CVE-2019-11358", "CWE-79"]


def test_sin_adjuntos_no_hay_ids_ni_texto():
    assert findings_vuln_ids([]) == []
    assert format_findings([]) == ""


def test_un_solo_adjunto_se_formatea_como_antes():
    assert format_findings([FINDING]) == format_finding(FINDING)


def test_varios_adjuntos_van_numerados():
    text = format_findings([FINDING, OTHER])
    assert text.startswith("[1]\n- Identifier: CVE-2023-4863")
    assert "\n\n[2]\n- Identifier: CVE-2019-11358" in text


def test_con_varios_adjuntos_el_advisory_se_reparte():
    long = dict(FINDING, details="palabra " * 1000)
    text = format_findings([long] * 2 + [OTHER] * 8)
    advisory = next(line for line in text.splitlines() if line.startswith("- Advisory"))
    assert len(advisory) < 400
