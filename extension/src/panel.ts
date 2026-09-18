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
import { PreviewPackage } from './scan/manifest';
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

    loading(manifest: string, packages: PreviewPackage[]): void {
        this.panel.webview.html = this.wrap(renderScanning(manifest, packages));
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
<body><div class="page">${body}</div></body>
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

/**
 * Pantalla de espera: el manifiesto y sus dependencias, recorridas por una onda.
 *
 * Los nombres son los reales —la extensión ya tiene el archivo abierto, así que no hay
 * que esperar al escáner para mostrarlos—. La animación es ambiental: no marca avance
 * por paquete, que sería falso (resolver uno son 3,5 ms de los 1.742 que tarda el
 * escaneo; el resto se va en abrir los catálogos).
 */
function renderScanning(manifest: string, packages: PreviewPackage[]): string {
    const items = packages
        .map(
            (item, index) => `<li class="scan-item ${item.pinned ? '' : 'loose'}" style="--i:${index}">
                <span class="scan-name">${escape(item.name)}</span>
                <span class="scan-version">${item.version ? escape(item.version) : ''}</span>
            </li>`,
        )
        .join('');

    return `<div class="scanning">
        <header class="page-header">
            <h1>Analizando dependencias</h1>
            <p class="muted">${escape(manifest)}</p>
        </header>
        <ul class="scan-grid" aria-label="Dependencias del manifiesto">${items}</ul>
    </div>`;
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
            <p class="label">Informe de seguridad</p>
            <h1>${plural(funnel.paquetes, 'dependencia', 'dependencias')} con vulnerabilidades conocidas</h1>
            <p class="muted">${escape(basename(result.manifest))}</p>
            <div class="summary-strip nums">
                <div><strong>${funnel.total}</strong><span>vulnerabilidades</span></div>
                <div><strong>${funnel.paquetes}</strong><span>librerías</span></div>
                <div class="${urgentes.length ? 'danger' : ''}"><strong>${urgentes.length}</strong><span>explotadas</span></div>
            </div>
        </header>

        ${urgentes.length ? renderUrgentFindings(urgentes) : renderNoUrgentFindings()}

        <section class="library-section">
            <div class="section-heading">
                <div>
                    <p class="label">Detalle por librería</p>
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
            <div class="section-heading">
                <div>
                    <p class="label">Prioridad inmediata</p>
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
            <p class="label">Prioridad inmediata</p>
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
        <section class="finding card ${finding.kev ? 'urgent' : ''}">
      <h2>
        ${finding.kev ? '<span class="badge solid">Explotada</span>' : ''}
                ${escape(finding.package)} ${escape(finding.installed_version)} <span class="finding-fix">— ${arreglo}</span>
      </h2>
            <p class="facts nums"><span class="badge ${severity.variant}">${severity.label}</span> ${datos.join(' · ')}</p>
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
      <button class="button quiet chat-button" data-finding="${escape(JSON.stringify(finding))}">
        Hablar en profundidad
      </button>
      ${renderSources(finding)}
    </section>`;
}

/**
 * Severidad → variante de `.badge`. El rojo lleno queda para KEV y sólo para KEV: acá
 * la crítica llega hasta el contorno rojo, y de media para abajo no gasta color.
 */
function severityLabel(score: number | null): { label: string; variant: string } {
    if (score === null) return { label: 'Sin CVSS', variant: '' };
    if (score >= 9) return { label: 'Crítica', variant: 'danger' };
    if (score >= 7) return { label: 'Alta', variant: 'warning' };
    if (score >= 4) return { label: 'Media', variant: '' };
    return { label: 'Baja', variant: '' };
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

/**
 * Sólo lo específico del informe. Todo lo que comparte con el chat —superficie,
 * badges, chips, botones, escalas, foco— vive en BASE_STYLES y acá no se redefine.
 */
const STYLES = `
  .page-header { padding-bottom: var(--sp-4); border-bottom: 1px solid var(--border); }
  .page-header h1 { margin-bottom: var(--sp-1); }

  .summary-strip { display: flex; flex-wrap: wrap; gap: var(--sp-2); margin-top: var(--sp-4); }
  .summary-strip > div {
    min-width: 8rem;
    padding: var(--sp-2) var(--sp-3);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface);
  }
  .summary-strip strong { display: block; font-size: var(--fs-xl); line-height: 1.2; }
  .summary-strip span { color: var(--muted); font-size: var(--fs-sm); }
  .summary-strip > .danger { border-color: var(--danger); }
  .summary-strip > .danger strong { color: var(--danger); }

  .urgent-section, .library-section { margin-top: var(--sp-5); }
  .urgent-section { border-left: 2px solid var(--danger); padding-left: var(--sp-4); }
  .quiet-section {
    margin-top: var(--sp-5);
    padding: var(--sp-3) var(--sp-4);
    border: 1px solid var(--border);
    border-radius: var(--radius);
  }
  .section-heading {
    display: flex; align-items: baseline; justify-content: space-between;
    gap: var(--sp-4); margin-bottom: var(--sp-2);
  }
  .section-heading h2 { margin: 0; }
  .section-count, .urgent-count {
    flex: none;
    color: var(--muted); font-size: var(--fs-sm); font-variant-numeric: tabular-nums;
  }
  .section-intro { margin: 0 0 var(--sp-3); color: var(--muted); }
  .closing { margin-top: var(--sp-5); color: var(--muted); }

  .package-group {
    margin: var(--sp-3) 0;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
  }
  .package-group > summary {
    padding: var(--sp-3) var(--sp-4);
    color: var(--vscode-foreground);
    background: var(--surface);
  }
  .package-group > summary:hover { background: var(--vscode-list-hoverBackground); }
  .package-findings { padding: 0 var(--sp-3) var(--sp-2); }

  .funnel { width: 100%; max-width: 32rem; margin: var(--sp-4) 0; border-collapse: collapse; }
  .funnel td { padding: var(--sp-1) var(--sp-2) var(--sp-1) 0; }
  .funnel td.num { width: 4rem; text-align: right; font-variant-numeric: tabular-nums; }
  .funnel tr:first-child td { font-weight: 600; }

  /* El hallazgo es la superficie compartida; lo único propio es la marca de urgencia. */
  .finding { margin: var(--sp-2) 0; border-left-width: 2px; }
  .finding.urgent { border-left-color: var(--danger); }
  .finding h2 { font-size: var(--fs-md); }
  .finding-fix { font-weight: 400; color: var(--muted); }
  .facts { color: var(--muted); font-size: var(--fs-sm); }
  .facts strong { color: var(--vscode-foreground); }
  .explanation { margin: var(--sp-2) 0; }
  .sources { margin-top: var(--sp-2); color: var(--muted); font-size: var(--fs-sm); }
  .chat-button { margin-top: var(--sp-2); }

  .skipped {
    margin-top: var(--sp-6);
    padding-top: var(--sp-4);
    border-top: 1px solid var(--border);
  }
  .skipped ul { margin: 0; padding-left: var(--sp-5); }
  .skipped li { margin: var(--sp-1) 0; color: var(--muted); }

  /* --- Pantalla de espera ------------------------------------------------------- */
  .scanning {
    display: flex; flex-direction: column; gap: var(--sp-4);
    min-height: calc(100vh - var(--sp-5) - var(--sp-6));
  }
  .scan-grid {
    flex: 1; align-content: start;
    display: grid; grid-template-columns: repeat(auto-fill, minmax(13rem, 1fr));
    gap: var(--sp-2);
    margin: 0; padding: 0; list-style: none;
  }
  .scan-item {
    display: flex; align-items: baseline; justify-content: space-between; gap: var(--sp-2);
    padding: var(--sp-2) var(--sp-3);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--muted);
    /* El retardo por índice es lo que hace que la onda viaje por la lista. */
    animation: scan-wave 2.6s ease-in-out infinite;
    animation-delay: calc(var(--i) * 90ms);
  }
  /* Sin pin exacto: el resolver las va a descartar, así que no participan de la onda. */
  .scan-item.loose { opacity: .45; animation: none; }
  .scan-name { font-family: var(--vscode-editor-font-family); font-size: var(--fs-sm); }
  .scan-version { font-size: var(--fs-xs); font-variant-numeric: tabular-nums; }

  @keyframes scan-wave {
    0%, 55%, 100% { border-color: var(--border); color: var(--muted); background: transparent; }
    22% {
      border-color: var(--vscode-focusBorder);
      color: var(--vscode-foreground);
      background: var(--surface);
    }
  }
  @media (prefers-reduced-motion: reduce) { .scan-item { animation: none; } }
`;
