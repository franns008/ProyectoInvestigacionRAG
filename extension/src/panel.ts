/**
 * El webview que muestra los hallazgos: la herramienta, no la maquinaria del RAG
 * (docs/escaneo_dependencias.md, "Disciplina de las dos superficies"). Todo texto de los
 * advisories se escapa antes de insertarse.
 */

import * as vscode from 'vscode';
import { ChatView } from './chatView';
import { PreviewPackage } from './scan/manifest';
import { chip } from './references';
import { Finding, ScanResult, findingKey, identifierOf, packageKey } from './scan/types';
import { escapeHtml as escape } from './util';
import { mediaRoot, openHttpLink, webviewPage } from './webview';

export class ResultsPanel {
    private static current: ResultsPanel | undefined;
    private readonly panel: vscode.WebviewPanel;
    private disposables: vscode.Disposable[] = [];
    /** Los hallazgos del último escaneo, por clave: el botón sólo manda la clave, no el hallazgo. */
    private findings = new Map<string, Finding>();

    private constructor(
        private readonly chat: ChatView,
        private readonly extensionUri: vscode.Uri,
    ) {
        this.panel = vscode.window.createWebviewPanel(
            'cibersec.results',
            'Dependencias vulnerables',
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
                localResourceRoots: [mediaRoot(extensionUri)],
            },
        );
        this.panel.webview.onDidReceiveMessage(
            (message: { type?: string; key?: string; url?: string }) => {
                if (message.type === 'toggleContext' && message.key) {
                    const finding = this.findings.get(message.key);
                    if (finding) void this.chat.toggle(finding);
                    return;
                }
                if (message.type === 'openExternal') {
                    void openHttpLink(message.url);
                }
            },
            null,
            this.disposables,
        );
        this.panel.onDidDispose(() => this.dispose(), null, this.disposables);
        // Los botones reflejan qué está adjunto, también cuando se quita desde el chat.
        this.disposables.push(
            chat.onDidChangeAttachments((keys) => {
                void this.panel.webview.postMessage({ type: 'attached', keys: [...keys] });
            }),
        );
    }

    /** Reusa el panel si ya está abierto, para no acumular pestañas en cada escaneo. */
    static show(chat: ChatView, extensionUri: vscode.Uri): ResultsPanel {
        if (!ResultsPanel.current) {
            ResultsPanel.current = new ResultsPanel(chat, extensionUri);
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
        this.findings = new Map(result.findings.map((f) => [findingKey(f), f]));
        this.panel.webview.html = this.wrap(renderResult(result));
    }

    private wrap(body: string): string {
        return webviewPage(this.panel.webview, this.extensionUri, 'panel', `<div class="page">${body}</div>`, {
            attached: [...this.chat.attachedKeys()],
        });
    }

    private dispose(): void {
        ResultsPanel.current = undefined;
        this.panel.dispose();
        this.disposables.forEach((d) => d.dispose());
        this.disposables = [];
    }
}

/**
 * Pantalla de espera: las dependencias reales del manifiesto, recorridas por una onda
 * ambiental (no marca avance por paquete, que sería falso).
 */
function renderScanning(manifest: string, packages: PreviewPackage[]): string {
    const items = packages
        .map(
            (item) => `<li class="scan-item ${item.pinned ? '' : 'loose'}">
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

    return `
        <header class="page-header">
            <h1>${plural(funnel.paquetes, 'dependencia', 'dependencias')} con vulnerabilidades conocidas</h1>
            <p class="muted">${escape(basename(result.manifest))}</p>
            ${renderSummaryLine(funnel.total, funnel.paquetes, urgentes.length)}
        </header>

        ${renderFunnel(funnel)}

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

    ${renderSkipped(skipped)}`;
}

/** Los tres números de la cabecera, en una línea; sólo el de explotadas lleva color. */
function renderSummaryLine(total: number, paquetes: number, explotadas: number): string {
    const explotadasTexto = explotadas
        ? `<span class="danger"><strong>${explotadas}</strong> explotadas</span>`
        : `<span class="muted">ninguna explotada</span>`;
    return `<p class="summary-line nums">
        <strong>${total}</strong> vulnerabilidades ·
        <strong>${paquetes}</strong> ${paquetes === 1 ? 'librería' : 'librerías'} ·
        ${explotadasTexto}
    </p>`;
}

function renderUrgentFindings(findings: Finding[]): string {
        return `<section class="urgent-section">
            <div class="section-heading">
                <div>
                    <p class="label">Prioridad inmediata</p>
                    <h2>Explotadas activamente</h2>
                </div>
                <span class="urgent-count">${plural(findings.length, 'hallazgo', 'hallazgos')}</span>
            </div>
            ${findings.map((f) => renderFinding(f, 'urgente')).join('')}
        </section>`;
}

function renderNoUrgentFindings(): string {
        return `<section class="quiet-section">
            <p class="label">Prioridad inmediata</p>
            <p><strong>No hay vulnerabilidades explotadas activamente.</strong> Ningún hallazgo aparece en el catálogo CISA KEV.</p>
        </section>`;
}

/**
 * Agrupa por librería conservando el orden en que llegan: `prioritize()` ya los ordena por
 * KEV, EPSS y CVSS. No reordenar acá: la vista no calcula prioridades.
 */
function renderPackageGroups(findings: Finding[]): string {
    const groups = new Map<string, Finding[]>();
    for (const finding of findings) {
        const key = packageKey(finding.package);
        const group = groups.get(key) ?? [];
        group.push(finding);
        groups.set(key, group);
    }

    return [...groups.values()].map(renderPackageGroup).join('');
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
    <div class="package-findings">${findings.map((f) => renderFinding(f, 'grupo')).join('')}</div>
  </details>`;
}

/**
 * El embudo: cuántos hallazgos sobreviven a cada criterio. Va arriba de todo, porque es
 * la premisa del informe, y con `th` de verdad para los lectores de pantalla.
 */
function renderFunnel(f: ScanResult['funnel']): string {
    const row = (label: string, value: number) =>
        `<tr><td>${label}</td><td class="num">${value}</td>
     <td class="num muted">${f.total ? Math.round((value / f.total) * 100) + '%' : ''}</td></tr>`;

    return `<section class="funnel-section">
    <h2>Resumen</h2>
    <table class="funnel">
      <thead>
        <tr><th scope="col">Criterio</th><th scope="col" class="num">Vulnerabilidades</th><th scope="col" class="num">% de ${f.total}</th></tr>
      </thead>
      <tbody>
        ${row('severidad alta (CVSS ≥ 7)', f.cvss_alto)}
        ${row('probabilidad de explotación (EPSS ≥ 10%)', f.epss_alto)}
        ${row('explotadas hoy (catálogo CISA KEV)', f.kev)}
      </tbody>
    </table>
  </section>`;
}

/** Clip de adjuntar, como en Copilot. SVG inline: el CSP es default-src 'none' y no hay assets. */
const ICONO_ADJUNTAR = `<svg class="icono" viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false"><path fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round" d="M13.5 7.5 8.1 12.9a3.25 3.25 0 0 1-4.6-4.6l5.6-5.6a2.1 2.1 0 0 1 3 3L6.5 11.3a1 1 0 0 1-1.4-1.4l5.2-5.2"/></svg>`;

/**
 * Adjuntar el hallazgo al chat o quitarlo. En "Explotadas" lleva ícono y etiqueta; en los
 * grupos, sólo ícono. El nombre accesible siempre incluye el identificador.
 */
function renderAction(finding: Finding, seccion: Seccion): string {
    const id = `${identifierOf(finding, 'este hallazgo')} de ${finding.package}`;
    const off = escape(`Agregar ${id} al chat`);
    const on = escape(`Quitar ${id} del chat`);
    const attrs = `class="button quiet chat-button" data-finding="${escape(findingKey(finding))}" data-label-off="${off}" data-label-on="${on}" aria-pressed="false" title="${off}" aria-label="${off}"`;

    return seccion === 'urgente'
        ? `<button ${attrs} data-with-text>${ICONO_ADJUNTAR}<span class="label">Agregar al chat</span></button>`
        : `<button ${attrs} data-icon-only>${ICONO_ADJUNTAR}</button>`;
}

/** Dónde se está dibujando el hallazgo, que cambia qué hace falta decir. */
type Seccion = 'urgente' | 'grupo';

/**
 * Una tarjeta de hallazgo, en tres líneas y cada cosa dicha una vez:
 *   1. Qué hacer        — paquete, versión instalada, versión que lo arregla.
 *   2. Qué es           — el resumen del advisory.
 *   3. Con qué respaldo — identificadores enlazados a su fuente, y la acción.
 */
function renderFinding(finding: Finding, seccion: Seccion = 'grupo'): string {
    const severity = severityLabel(finding.cvss_score);
    // Dentro de "Explotadas activamente" la marca sobra: lo dice el título de la
    // sección. En el grupo de la librería el hallazgo se lee fuera de ese contexto y
    // ahí sí hace falta, porque es lo único que lo distingue de sus vecinos.
    const explotada = finding.kev && seccion === 'grupo';
    // El título es la acción: "paquete versión → versión que lo arregla".
    const destino = finding.fixed_version
        ? `<span class="finding-fix">→ ${escape(finding.fixed_version)}</span>`
        : `<span class="finding-fix muted">→ sin arreglo publicado</span>`;

    // Cuando no hay CVE el identificador es el propio OSV, y ahí sí hay que mostrarlo.
    const identificadores = finding.cve ? [finding.cve] : finding.osv_ids;
    const meta = [
        ...identificadores.map(chip),
        finding.cvss_score !== null ? `CVSS ${finding.cvss_score}` : 'sin CVSS',
        `EPSS ${formatEpss(finding.epss)}`,
        ...finding.cwe_ids.map(chip),
    ];

    return `
    <article class="finding card ${finding.kev ? 'urgent' : ''}">
      <h3 class="finding-head">
        <span class="finding-title">${escape(finding.package)} ${escape(finding.installed_version)} ${destino}</span>
        <span class="finding-tags">
          ${explotada ? '<span class="tag danger">Explotada</span>' : ''}
          <span class="tag ${severity.variant}">${severity.label}</span>
        </span>
      </h3>
      ${finding.explanation ? renderExplanation(finding) : renderSummary(finding)}
      <div class="finding-foot">
        <span class="finding-meta nums">${meta.join(' · ')}</span>
        ${renderAction(finding, seccion)}
      </div>
    </article>`;
}

/** El resumen del advisory sin el prefijo "Paquete: ", que ya es el título de la tarjeta. */
function summaryText(finding: Finding): string {
    if (!finding.summary) return '';
    const prefijo = new RegExp(`^${escapeRegExp(finding.package)}\\s*:\\s*`, 'i');
    return finding.summary.replace(prefijo, '');
}

/** Sin explicación del LLM, el resumen del advisory es el texto principal de la tarjeta. */
function renderSummary(finding: Finding): string {
    const texto = summaryText(finding);
    return texto ? `<p class="finding-summary">${escape(texto)}</p>` : '';
}

/**
 * La explicación del LLM, con el resumen del advisory debajo como cita: es casi siempre
 * su fuente, y contra eso se contrasta la prosa del modelo.
 */
function renderExplanation(finding: Finding): string {
    if (!finding.explanation) return '';
    const original = summaryText(finding);
    return `<div class="explanation">
      <p>${escape(finding.explanation)}</p>
      ${original ? `<blockquote class="advisory-original" lang="en">${escape(original)}</blockquote>` : ''}
    </div>`;
}

function formatEpss(epss: number): string {
    return `${(epss * 100).toFixed(epss >= 0.01 ? 0 : 1)}%`;
}

function escapeRegExp(value: string): string {
    return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function severityLabel(score: number | null): { label: string; variant: string } {
    if (score === null) return { label: 'Sin CVSS', variant: 'unknown' };
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
