/**
 * El webview que muestra los hallazgos.
 *
 * Disciplina de la vista (docs/escaneo_dependencias.md, "Disciplina de las dos
 * superficies"): acá va la herramienta —hallazgos, prioridad, explicación, citas— y
 * NADA de la maquinaria. Nada de scores de retriever, etapas del pipeline ni métricas
 * de evaluación: eso vive en el panel de evidencia, que es la otra pantalla.
 *
 * Todo el texto que viene de los advisories se escapa antes de insertarse: es contenido
 * de terceros.
 */

import * as vscode from "vscode";
import { Finding, ScanResult } from "./scan/types";

export class ResultsPanel {
  private static current: ResultsPanel | undefined;
  private readonly panel: vscode.WebviewPanel;
  private disposables: vscode.Disposable[] = [];

  private constructor() {
    this.panel = vscode.window.createWebviewPanel(
      "cibersec.results",
      "Dependencias vulnerables",
      vscode.ViewColumn.Beside,
      { enableScripts: false, retainContextWhenHidden: true },
    );
    this.panel.onDidDispose(() => this.dispose(), null, this.disposables);
  }

  /** Reusa el panel si ya está abierto, para no acumular pestañas en cada escaneo. */
  static show(): ResultsPanel {
    if (!ResultsPanel.current) {
      ResultsPanel.current = new ResultsPanel();
    }
    ResultsPanel.current.panel.reveal(vscode.ViewColumn.Beside, true);
    return ResultsPanel.current;
  }

  loading(manifest: string, provider: string): void {
    this.panel.webview.html = this.wrap(
      `<p class="muted">Escaneando <code>${escape(manifest)}</code> con el ${escape(provider)}…</p>`,
    );
  }

  error(message: string, detail?: string): void {
    this.panel.webview.html = this.wrap(
      `<h1>No se pudo escanear</h1>
       <p>${escape(message)}</p>
       ${detail ? `<pre>${escape(detail)}</pre>` : ""}`,
    );
  }

  render(result: ScanResult): void {
    this.panel.webview.html = this.wrap(renderResult(result));
  }

  private wrap(body: string): string {
    const csp = "default-src 'none'; style-src 'unsafe-inline';";
    return `<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="${csp}">
<style>${STYLES}</style>
</head>
<body>${body}</body>
</html>`;
  }

  private dispose(): void {
    ResultsPanel.current = undefined;
    this.panel.dispose();
    this.disposables.forEach((d) => d.dispose());
    this.disposables = [];
  }
}

function renderResult(result: ScanResult): string {
  const { funnel, findings, skipped } = result;

  if (funnel.total === 0) {
    return `
      <h1>Sin vulnerabilidades conocidas</h1>
      <p class="muted">${escape(basename(result.manifest))}</p>
      ${renderSkipped(skipped)}`;
  }

  const urgentes = findings.filter((f) => f.kev);
  const resto = funnel.total - urgentes.length;

  return `
    <h1>${plural(funnel.paquetes, "dependencia", "dependencias")} con vulnerabilidades conocidas</h1>
    <p class="muted">${escape(basename(result.manifest))} · ${plural(funnel.total, "CVE", "CVEs")} en total</p>

    ${renderFunnel(funnel)}

    ${
      urgentes.length > 0
        ? `<p class="lead">Priorizadas por explotabilidad real, ${
            urgentes.length === 1
              ? "esta es la única que se está explotando hoy:"
              : `estas ${urgentes.length} se están explotando hoy:`
          }</p>`
        : `<p class="lead">Ninguna está en el catálogo de CISA. Ordenadas por probabilidad de exploit:</p>`
    }

    ${findings.map(renderFinding).join("")}

    ${
      urgentes.length > 0 && resto > 0
        ? `<p class="closing">Las otras ${resto} no tienen exploit conocido circulando.</p>`
        : ""
    }

    ${renderSkipped(skipped)}`;
}

function renderFunnel(f: ReturnType<() => ScanResult["funnel"]>): string {
  const row = (label: string, value: number) =>
    `<tr><td>${label}</td><td class="num">${value}</td>
     <td class="num muted">${f.total ? Math.round((value / f.total) * 100) + "%" : ""}</td></tr>`;

  return `<table class="funnel">
    <tr><td>CVEs que afectan tus versiones</td><td class="num">${f.total}</td><td></td></tr>
    ${row("filtrando por severidad alta (CVSS ≥ 7)", f.cvss_alto)}
    ${row("filtrando por probabilidad de exploit (EPSS ≥ 10%)", f.epss_alto)}
    ${row("explotadas hoy (catálogo CISA KEV)", f.kev)}
  </table>`;
}

function renderFinding(finding: Finding): string {
  const id = finding.cve ?? finding.osv_ids[0] ?? "sin identificador";
  const arreglo = finding.fixed_version
    ? `actualizá a <strong>${escape(finding.fixed_version)}</strong>`
    : `<span class="muted">sin versión de arreglo publicada</span>`;

  const datos = [
    escape(id),
    finding.cvss_score !== null ? `CVSS ${finding.cvss_score}` : "CVSS no disponible",
    `EPSS ${(finding.epss * 100).toFixed(finding.epss >= 0.01 ? 0 : 1)}%`,
    finding.kev ? "<strong>en el catálogo CISA KEV</strong>" : null,
  ].filter(Boolean);

  return `
    <section class="finding ${finding.kev ? "urgent" : ""}">
      <h2>
        ${finding.kev ? '<span class="flag">EXPLOTADA</span>' : ""}
        ${escape(finding.package)} ${escape(finding.installed_version)} — ${arreglo}
      </h2>
      <p class="facts">${datos.join(" · ")}</p>
      ${finding.summary ? `<p>${escape(finding.summary)}</p>` : ""}
      ${
        finding.explanation
          ? `<div class="explanation">${escape(finding.explanation)}</div>`
          : ""
      }
      ${
        finding.cwe_ids.length
          ? `<p class="muted">Clase de debilidad: ${finding.cwe_ids.map(escape).join(", ")}</p>`
          : ""
      }
      ${renderSources(finding)}
    </section>`;
}

function renderSources(finding: Finding): string {
  const fuentes = [
    ...(finding.citations ?? []),
    ...finding.osv_ids,
    ...finding.cwe_ids.map((cwe) => `${cwe} (MITRE)`),
  ];
  if (finding.kev) {
    fuentes.push("CISA KEV");
  }
  if (fuentes.length === 0) {
    return "";
  }
  return `<p class="sources"><span class="muted">Fuentes:</span> ${fuentes.map(escape).join(" · ")}</p>`;
}

function renderSkipped(skipped: ScanResult["skipped"]): string {
  const relevantes = skipped.filter((s) => !s.raw.startsWith("--"));
  if (relevantes.length === 0) {
    return "";
  }
  const filas = relevantes
    .map(
      (s) =>
        `<li>${s.line ? `<span class="muted">línea ${s.line}:</span> ` : ""}<code>${escape(
          s.raw,
        )}</code> — ${escape(s.reason)}</li>`,
    )
    .join("");
  return `<section class="skipped">
    <h3>No escaneadas (${relevantes.length})</h3>
    <ul>${filas}</ul>
  </section>`;
}

/** Sólo el nombre del archivo: la ruta completa no le dice nada al usuario. */
function basename(fsPath: string): string {
  return fsPath.split(/[\\/]/).pop() || fsPath;
}

function plural(n: number, singular: string, plural_: string): string {
  return `${n} ${n === 1 ? singular : plural_}`;
}

function escape(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

const STYLES = `
  body {
    font-family: var(--vscode-font-family);
    font-size: var(--vscode-font-size);
    color: var(--vscode-foreground);
    padding: 1.5rem 2rem 3rem;
    line-height: 1.5;
    max-width: 60rem;
  }
  h1 { font-size: 1.3rem; font-weight: 600; margin: 0 0 .25rem; }
  h2 { font-size: 1rem; font-weight: 600; margin: 0 0 .4rem; }
  h3 { font-size: .9rem; font-weight: 600; margin: 0 0 .5rem; }
  p { margin: .4rem 0; }
  code { font-family: var(--vscode-editor-font-family); font-size: .9em; }
  .muted { color: var(--vscode-descriptionForeground); }
  .lead { margin: 1.25rem 0 .75rem; }
  .closing { margin-top: 1.5rem; color: var(--vscode-descriptionForeground); }

  .funnel { border-collapse: collapse; margin: 1rem 0; width: 100%; max-width: 32rem; }
  .funnel td { padding: .2rem .5rem .2rem 0; }
  .funnel td.num { text-align: right; font-variant-numeric: tabular-nums; width: 4rem; }
  .funnel tr:first-child td { font-weight: 600; }

  .finding {
    border: 1px solid var(--vscode-panel-border);
    border-left: 3px solid var(--vscode-panel-border);
    border-radius: 3px;
    padding: .85rem 1rem;
    margin: .6rem 0;
  }
  .finding.urgent { border-left-color: var(--vscode-editorError-foreground); }
  .flag {
    background: var(--vscode-editorError-foreground);
    color: var(--vscode-editor-background);
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: .04em;
    padding: .1rem .4rem;
    border-radius: 2px;
    vertical-align: .1em;
    margin-right: .3rem;
  }
  .facts { font-variant-numeric: tabular-nums; color: var(--vscode-descriptionForeground); }
  .facts strong { color: var(--vscode-foreground); }
  .explanation { margin: .6rem 0; }
  .sources { font-size: .85em; margin-top: .6rem; }

  .skipped { margin-top: 2rem; border-top: 1px solid var(--vscode-panel-border); padding-top: 1rem; }
  .skipped ul { margin: 0; padding-left: 1.2rem; }
  .skipped li { margin: .2rem 0; color: var(--vscode-descriptionForeground); }

  pre {
    background: var(--vscode-textCodeBlock-background);
    padding: .75rem; border-radius: 3px; overflow-x: auto; white-space: pre-wrap;
  }
`;
