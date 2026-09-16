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

import * as vscode from 'vscode';
import { ChatPanel, ChatOptions } from './chatPanel';
import { Finding, ScanResult } from './scan/types';
import { BASE_STYLES, escapeHtml as escape, nonceValue } from './styles';

export class ResultsPanel {
    private static current: ResultsPanel | undefined;
    private readonly panel: vscode.WebviewPanel;
    private disposables: vscode.Disposable[] = [];

    private constructor() {
        this.panel = vscode.window.createWebviewPanel(
            'cibersec.results',
            'Dependencias vulnerables',
            vscode.ViewColumn.Beside,
            { enableScripts: true, retainContextWhenHidden: true },
        );
        this.panel.webview.onDidReceiveMessage(
            (message: { type?: string; finding?: Finding }) => {
                if (message.type === 'openChat' && message.finding) {
                    ChatPanel.show(message.finding, this.chatOptions());
                }
            },
            null,
            this.disposables,
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
       ${detail ? `<pre>${escape(detail)}</pre>` : ''}`,
        );
    }

    render(result: ScanResult): void {
        this.panel.webview.html = this.wrap(renderResult(result));
    }

    private wrap(body: string): string {
        const nonce = nonceValue();
        const csp = `default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';`;
        return `<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="${csp}">
<style>${BASE_STYLES}${STYLES}</style>
<script nonce="${nonce}">
  const vscode = acquireVsCodeApi();
  document.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-finding]");
    if (!button) return;
    vscode.postMessage({ type: "openChat", finding: JSON.parse(button.dataset.finding) });
  });
</script>
</head>
<body>${body}</body>
</html>`;
    }

    private chatOptions(): ChatOptions {
        const config = vscode.workspace.getConfiguration('cibersec');
        return {
            url: config.get<string>('ragUrl', 'http://localhost:9099'),
            model: config.get<string>('chatModel', 'pipeline_ciberseguridad'),
            apiKey: config.get<string>('ragApiKey', '0p3n-w3bui'),
            timeoutMs: config.get<number>('chatTimeoutSeconds', 240) * 1000,
        };
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
        <header class="page-header">
            <p class="eyebrow">INFORME DE SEGURIDAD</p>
            <h1>${plural(funnel.paquetes, 'dependencia', 'dependencias')} con vulnerabilidades conocidas</h1>
            <p class="muted">${escape(basename(result.manifest))}</p>
            <div class="summary-strip">
                <div><strong>${funnel.total}</strong><span>vulnerabilidades</span></div>
                <div><strong>${funnel.paquetes}</strong><span>librerías</span></div>
                <div class="${urgentes.length ? 'danger' : ''}"><strong>${urgentes.length}</strong><span>explotadas</span></div>
            </div>
        </header>

        ${urgentes.length ? renderUrgentFindings(urgentes) : renderNoUrgentFindings()}

        <section class="library-section">
            <div class="section-heading">
                <div>
                    <p class="eyebrow">DETALLE POR LIBRERIA</p>
                    <h2>Vulnerabilidades agrupadas</h2>
                </div>
                <span class="section-count">${plural(funnel.paquetes, 'librería', 'librerías')}</span>
            </div>
            ${renderPackageGroups(findings)}
        </section>

    ${renderFunnel(funnel)}

    ${
        urgentes.length > 0 && resto > 0
            ? `<p class="closing">Las otras ${resto} vulnerabilidades no tienen exploit conocido circulando.</p>`
            : ''
    }

    ${renderSkipped(skipped)}`;
}

function renderUrgentFindings(findings: Finding[]): string {
        return `<section class="urgent-section">
            <div class="section-heading urgent-heading">
                <div>
                    <p class="eyebrow">PRIORIDAD INMEDIATA</p>
                    <h2>Explotadas activamente</h2>
                </div>
                <span class="urgent-count">${findings.length}</span>
            </div>
            <p class="section-intro">Estas vulnerabilidades aparecen en el catálogo CISA KEV y requieren atención prioritaria.</p>
            ${findings.map(renderFinding).join('')}
        </section>`;
}

function renderNoUrgentFindings(): string {
        return `<section class="quiet-section">
            <p class="eyebrow">PRIORIDAD INMEDIATA</p>
            <p><strong>No hay vulnerabilidades explotadas activamente.</strong> Ningún hallazgo aparece en el catálogo CISA KEV.</p>
        </section>`;
}

function renderPackageGroups(findings: Finding[]): string {
    const groups = new Map<string, Finding[]>();
    for (const finding of findings) {
        const key = finding.package.toLowerCase().replace(/[-_.]+/g, '-');
        const group = groups.get(key) ?? [];
        group.push(finding);
        groups.set(key, group);
    }

    return [...groups.values()].map((group) =>
        renderPackageGroup([...group].sort(compareFindingsBySeverity)),
    ).join('');
}

function compareFindingsBySeverity(left: Finding, right: Finding): number {
    const leftScore = left.cvss_score ?? -1;
    const rightScore = right.cvss_score ?? -1;
    return rightScore - leftScore;
}

function renderPackageGroup(findings: Finding[]): string {
    const packageName = findings[0].package;
    const fixedVersions = findings
        .map((finding) => finding.fixed_version)
        .filter((version): version is string => Boolean(version));
    const highest = fixedVersions.reduce(
        (current, version) =>
            !current || compareVersions(version, current) > 0
                ? version
                : current,
        '',
    );
    const arreglo = highest
        ? `versión que resuelve todas: <strong>${escape(highest)}</strong>`
        : `<span class="muted">sin versión de arreglo publicada</span>`;

    return `<details class="package-group" open>
    <summary><strong>${escape(packageName)}</strong> · ${plural(findings.length, 'vulnerabilidad', 'vulnerabilidades')} · ${arreglo}</summary>
    <div class="package-findings">${findings.map(renderFinding).join('')}</div>
  </details>`;
}

function renderFunnel(f: ReturnType<() => ScanResult['funnel']>): string {
    const row = (label: string, value: number) =>
        `<tr><td>${label}</td><td class="num">${value}</td>
     <td class="num muted">${f.total ? Math.round((value / f.total) * 100) + '%' : ''}</td></tr>`;

    return `<table class="funnel">
    <tr><td>CVEs que afectan tus versiones</td><td class="num">${f.total}</td><td></td></tr>
    ${row('filtrando por severidad alta (CVSS ≥ 7)', f.cvss_alto)}
    ${row('filtrando por probabilidad de exploit (EPSS ≥ 10%)', f.epss_alto)}
    ${row('explotadas hoy (catálogo CISA KEV)', f.kev)}
  </table>`;
}

function renderFinding(finding: Finding): string {
    const id = finding.cve ?? finding.osv_ids[0] ?? 'sin identificador';
    const arreglo = finding.fixed_version
        ? `actualizá a <strong>${escape(finding.fixed_version)}</strong>`
        : `<span class="muted">sin versión de arreglo publicada</span>`;

    const datos = [
        escape(id),
        finding.cvss_score !== null
            ? `CVSS ${finding.cvss_score}`
            : 'CVSS no disponible',
        `EPSS ${(finding.epss * 100).toFixed(finding.epss >= 0.01 ? 0 : 1)}%`,
        finding.kev ? '<strong>en el catálogo CISA KEV</strong>' : null,
    ].filter(Boolean);

        const severity = severityLabel(finding.cvss_score);
        return `
        <section class="finding ${finding.kev ? 'urgent' : ''} ${severity.className}">
      <h2>
        ${finding.kev ? '<span class="flag">EXPLOTADA</span>' : ''}
                ${escape(finding.package)} ${escape(finding.installed_version)} <span class="finding-fix">— ${arreglo}</span>
      </h2>
            <p class="facts"><span class="severity-badge">${severity.label}</span> ${datos.join(' · ')}</p>
      ${finding.summary ? `<p>${escape(finding.summary)}</p>` : ''}
      ${
          finding.explanation
              ? `<div class="explanation">${escape(finding.explanation)}</div>`
              : ''
      }
      ${
          finding.cwe_ids.length
              ? `<p class="muted">Clase de debilidad: ${finding.cwe_ids.map(escape).join(', ')}</p>`
              : ''
      }
      <button class="button chat-button" data-finding="${escape(JSON.stringify(finding))}">
        Hablar en profundidad
      </button>
      ${renderSources(finding)}
    </section>`;
}

function severityLabel(score: number | null): { label: string; className: string } {
    if (score === null) return { label: 'SIN CVSS', className: 'severity-unknown' };
    if (score >= 9) return { label: 'CRITICA', className: 'severity-critical' };
    if (score >= 7) return { label: 'ALTA', className: 'severity-high' };
    if (score >= 4) return { label: 'MEDIA', className: 'severity-medium' };
    return { label: 'BAJA', className: 'severity-low' };
}

function compareVersions(left: string, right: string): number {
    const parsedLeft = parseVersion(left);
    const parsedRight = parseVersion(right);

    for (let index = 0; index < parsedLeft.release.length; index += 1) {
        const difference =
            parsedLeft.release[index] - (parsedRight.release[index] ?? 0);
        if (difference !== 0) return difference;
    }
    for (let index = 0; index < parsedRight.release.length; index += 1) {
        const difference =
            (parsedLeft.release[index] ?? 0) - parsedRight.release[index];
        if (difference !== 0) return difference;
    }

    if (parsedLeft.stage !== parsedRight.stage) {
        return parsedLeft.stage - parsedRight.stage;
    }
    return parsedLeft.stageNumber - parsedRight.stageNumber;
}

function parseVersion(version: string): {
    release: number[];
    stage: number;
    stageNumber: number;
} {
    const match = version
        .trim()
        .match(
            /^v?(\d+(?:\.\d+)*)(?:(?:[-_.]?(a|alpha|b|beta|rc|pre|preview))(\d*)|[-_.]?post(\d*))?/i,
        );
    if (!match) {
        return { release: [0], stage: 0, stageNumber: 0 };
    }

    const label = match[2]?.toLowerCase();
    const stage = label
        ? label.startsWith('a')
            ? 1
            : label.startsWith('b')
              ? 2
              : 3
        : 4;
    const stageNumber = Number(match[3] ?? match[4] ?? 0);
    return {
        release: match[1].split('.').map(Number),
        stage,
        stageNumber,
    };
}

function renderSources(finding: Finding): string {
    const fuentes = [
        ...(finding.citations ?? []),
        ...finding.osv_ids,
        ...finding.cwe_ids.map((cwe) => `${cwe} (MITRE)`),
    ];
    if (finding.kev) {
        fuentes.push('CISA KEV');
    }
    if (fuentes.length === 0) {
        return '';
    }
    return `<p class="sources"><span class="muted">Fuentes:</span> ${fuentes.map(escape).join(' · ')}</p>`;
}

function renderSkipped(skipped: ScanResult['skipped']): string {
    const relevantes = skipped.filter((s) => !s.raw.startsWith('--'));
    if (relevantes.length === 0) {
        return '';
    }
    const filas = relevantes
        .map(
            (s) =>
                `<li>${s.line ? `<span class="muted">línea ${s.line}:</span> ` : ''}<code>${escape(
                    s.raw,
                )}</code> — ${escape(s.reason)}</li>`,
        )
        .join('');
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

/** Específicos del panel de resultados; la base compartida está en styles.ts. */
const STYLES = `
    .page-header { padding-bottom: 1.25rem; border-bottom: 1px solid var(--vscode-panel-border); }
    .eyebrow { margin: 0 0 .25rem; color: var(--vscode-textLink-foreground); font-size: .7rem; font-weight: 700; letter-spacing: .08em; }
    .page-header h1 { margin-bottom: .1rem; }
    .summary-strip { display: flex; gap: .5rem; margin-top: 1rem; flex-wrap: wrap; }
    .summary-strip div { min-width: 8rem; padding: .65rem .8rem; border: 1px solid var(--vscode-panel-border); border-radius: 4px; background: var(--vscode-textCodeBlock-background); }
    .summary-strip strong { display: block; font-size: 1.25rem; line-height: 1.1; }
    .summary-strip span { color: var(--vscode-descriptionForeground); font-size: .8rem; }
    .summary-strip .danger { border-color: var(--vscode-editorError-foreground); }
    .summary-strip .danger strong { color: var(--vscode-editorError-foreground); }
    .urgent-section, .library-section { margin-top: 1.5rem; }
    .urgent-section { border-left: 3px solid var(--vscode-editorError-foreground); padding-left: 1rem; }
    .quiet-section { margin-top: 1.5rem; padding: .8rem 1rem; border: 1px solid var(--vscode-panel-border); border-radius: 4px; }
    .section-heading { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin-bottom: .5rem; }
    .section-heading h2 { margin: 0; font-size: 1.05rem; }
    .section-count, .urgent-count { color: var(--vscode-descriptionForeground); font-size: .8rem; }
    .urgent-count { display: inline-grid; place-items: center; min-width: 1.6rem; height: 1.6rem; border-radius: 50%; background: var(--vscode-editorError-foreground); color: var(--vscode-editor-background); font-weight: 700; }
    .urgent-heading .eyebrow { color: var(--vscode-editorError-foreground); }
    .section-intro { color: var(--vscode-descriptionForeground); margin: 0 0 .75rem; }
  .closing { margin-top: 1.5rem; color: var(--vscode-descriptionForeground); }

    .package-group { margin: .75rem 0; border: 1px solid var(--vscode-panel-border); border-radius: 4px; overflow: hidden; }
  .package-group > summary {
    cursor: pointer;
        padding: .8rem 1rem;
    color: var(--vscode-foreground);
        background: var(--vscode-textCodeBlock-background);
        list-style-position: inside;
  }
  .package-group > summary strong { font-size: 1.05em; }
    .package-group > summary:hover { background: var(--vscode-list-hoverBackground); }
    .package-findings { padding: .15rem .75rem .55rem; }

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
    .finding h2 { line-height: 1.35; }
    .finding-fix { font-weight: 400; }
    .severity-badge { display: inline-block; font-size: .68rem; font-weight: 700; letter-spacing: .04em; padding: .08rem .35rem; border-radius: 2px; color: var(--vscode-editor-background); background: var(--vscode-descriptionForeground); }
    .severity-critical .severity-badge, .severity-high .severity-badge { background: var(--vscode-editorError-foreground); }
    .severity-medium .severity-badge { background: var(--vscode-editorWarning-foreground); }
    .severity-low .severity-badge, .severity-unknown .severity-badge { color: var(--vscode-foreground); background: var(--vscode-badge-background); }
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
  .chat-button { margin-top: .5rem; }

  .skipped { margin-top: 2rem; border-top: 1px solid var(--vscode-panel-border); padding-top: 1rem; }
  .skipped ul { margin: 0; padding-left: 1.2rem; }
  .skipped li { margin: .2rem 0; color: var(--vscode-descriptionForeground); }
`;
