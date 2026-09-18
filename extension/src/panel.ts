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
import { chip } from './references';
import { Finding, ScanResult } from './scan/types';
import { BASE_STYLES, escapeHtml as escape, nonceValue } from './styles';

export class ResultsPanel {
    private static current: ResultsPanel | undefined;
    private readonly panel: vscode.WebviewPanel;
    private disposables: vscode.Disposable[] = [];
    /**
     * Los hallazgos del último escaneo, por clave. Serializar cada uno dentro de un
     * atributo `data-` costaba 291 KB de HTML en un informe de 115 hallazgos, la mayor
     * parte en `details` y `references`, que el DOM no usa para nada: el botón sólo
     * necesita decir cuál se abre.
     */
    private findings = new Map<string, Finding>();

    private constructor() {
        this.panel = vscode.window.createWebviewPanel(
            'cibersec.results',
            'Dependencias vulnerables',
            vscode.ViewColumn.Beside,
            { enableScripts: true, retainContextWhenHidden: true },
        );
        this.panel.webview.onDidReceiveMessage(
            (message: { type?: string; key?: string; url?: string }) => {
                if (message.type === 'openChat' && message.key) {
                    const finding = this.findings.get(message.key);
                    if (finding) ChatPanel.show(finding, this.chatOptions());
                    return;
                }
                if (message.type === 'openExternal' && message.url && /^https?:\/\//i.test(message.url)) {
                    void vscode.env.openExternal(vscode.Uri.parse(message.url));
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
        this.findings = new Map(result.findings.map((f) => [findingKey(f), f]));
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
    if (button) {
      vscode.postMessage({ type: "openChat", key: button.dataset.finding });
      return;
    }
    const link = event.target.closest("a[href]");
    if (link) {
      event.preventDefault();
      vscode.postMessage({ type: "openExternal", url: link.getAttribute("href") });
    }
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

/**
 * Los tres números de la cabecera, en una línea.
 *
 * Eran tres tiles con el número en grande sobre un recinto con fondo. Ese patrón es el
 * de un dashboard, y acá lo que se busca es lo contrario (docs/escaneo_dependencias.md:
 * "aburrida y profesional"). El único que lleva color es el de explotadas, que es el que
 * decide qué se hace hoy.
 */
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
    <div class="package-findings">${findings.map((f) => renderFinding(f, 'grupo')).join('')}</div>
  </details>`;
}

/**
 * El embudo: cuántos hallazgos sobreviven a cada criterio de filtrado.
 *
 * Va arriba de todo, antes de los hallazgos. Es la premisa —por qué hay que mirar siete
 * y no ciento quince— y estaba detrás de la conclusión, al final de una página de
 * quince pantallas, donde no lo leía nadie.
 *
 * Lleva encabezados de columna de verdad: es una tabla de datos, y sin th un lector
 * de pantalla la recorre sin saber qué significa cada celda.
 *
 * El total no es una fila de la tabla: es la base contra la que se calculan los
 * porcentajes, y por eso era la única sin uno. Va en la línea que la introduce.
 */
function renderFunnel(f: ScanResult['funnel']): string {
    const row = (label: string, value: number) =>
        `<tr><td>${label}</td><td class="num">${value}</td>
     <td class="num muted">${f.total ? Math.round((value / f.total) * 100) + '%' : ''}</td></tr>`;

    return `<section class="funnel-section">
    <h2>Resumen</h2>
    <p class="muted">De las ${f.total} que afectan tus versiones, cuántas quedan con cada criterio:</p>
    <table class="funnel">
      <thead>
        <tr><th scope="col">Criterio</th><th scope="col" class="num">Quedan</th><th scope="col" class="num">%</th></tr>
      </thead>
      <tbody>
        ${row('severidad alta (CVSS ≥ 7)', f.cvss_alto)}
        ${row('probabilidad de explotación (EPSS ≥ 10%)', f.epss_alto)}
        ${row('explotadas hoy (catálogo CISA KEV)', f.kev)}
      </tbody>
    </table>
  </section>`;
}

/** Dónde se está dibujando el hallazgo, que cambia qué hace falta decir. */
type Seccion = 'urgente' | 'grupo';

/**
 * Una tarjeta de hallazgo, en tres líneas.
 *
 * Antes eran seis bloques con mucha repetición medida sobre los 115 hallazgos del
 * manifiesto de cobertura: KEV se decía cuatro veces en la misma tarjeta (riel, tag,
 * "en el catálogo CISA KEV" y "CISA KEV" en Fuentes), el CWE dos, y la línea de Fuentes
 * repetía identificadores ya visibles en 109 de 115 casos. Cada cosa se dice una vez:
 *
 *   1. Qué hacer      — paquete, versión instalada, versión que lo arregla.
 *   2. Qué es         — el resumen del advisory.
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
        <span>${escape(finding.package)} ${escape(finding.installed_version)} ${destino}</span>
        <span class="finding-tags">
          ${explotada ? '<span class="tag danger">Explotada</span>' : ''}
          <span class="tag ${severity.variant}">${severity.label}</span>
        </span>
      </h3>
      ${renderSummary(finding)}
      ${renderExplanation(finding)}
      <div class="finding-foot">
        <span class="finding-meta nums">${meta.join(' · ')}</span>
        <button class="button chat-button" data-finding="${escape(findingKey(finding))}">
          Profundizar
        </button>
      </div>
    </article>`;
}

/**
 * El resumen del advisory, sin el prefijo con el nombre del paquete.
 *
 * 21 de los 115 hallazgos del manifiesto de cobertura arrancan con "Marimo: ", "Ray: "
 * y similares. El paquete ya es el título de la tarjeta.
 */
function renderSummary(finding: Finding): string {
    if (!finding.summary) return '';
    const prefijo = new RegExp(`^${escapeRegExp(finding.package)}\\s*:\\s*`, 'i');
    const texto = finding.summary.replace(prefijo, '');
    return `<p class="finding-summary">${escape(texto)}</p>`;
}

/**
 * La explicación del LLM, con sus citas pegadas.
 *
 * Las citas viven acá y no en una línea de "Fuentes" al pie de la tarjeta: son las
 * fuentes de la prosa que escribió el modelo, no de los campos duros que arma Python
 * desde los datos estructurados. Mezclarlas sugería que todo el contenido tiene el
 * mismo respaldo, y no lo tiene.
 */
function renderExplanation(finding: Finding): string {
    if (!finding.explanation) return '';
    const citas = (finding.citations ?? []).filter(Boolean);
    return `<div class="explanation">
      <p>${escape(finding.explanation)}</p>
      ${citas.length ? `<p class="citations muted">Citado: ${citas.map(escape).join(' · ')}</p>` : ''}
    </div>`;
}

function formatEpss(epss: number): string {
    return `${(epss * 100).toFixed(epss >= 0.01 ? 0 : 1)}%`;
}

/** Identifica un hallazgo entre el webview y la extensión. `scan` dedupe por este par. */
function findingKey(finding: Finding): string {
    return `${finding.package}|${finding.cve ?? finding.osv_ids[0] ?? '?'}`;
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

/**
 * Sólo lo específico del informe. Todo lo que comparte con el chat —superficie,
 * tags, chips, botones, escalas, foco— vive en BASE_STYLES y acá no se redefine.
 */
const STYLES = `
  .page-header { padding-bottom: var(--sp-4); border-bottom: 1px solid var(--border); }
  .page-header h1 { margin-bottom: var(--sp-1); }

  .summary-line { margin: var(--sp-2) 0 0; color: var(--muted); }
  .summary-line strong { color: var(--vscode-foreground); }
  .summary-line .danger, .summary-line .danger strong { color: var(--danger); }

  .urgent-section, .library-section { margin-top: var(--sp-5); }
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

  .funnel-section { margin-top: var(--sp-5); }
  .funnel { width: 100%; max-width: 32rem; margin: var(--sp-2) 0 0; border-collapse: collapse; }
  .funnel th, .funnel td { padding: var(--sp-1) var(--sp-2) var(--sp-1) 0; text-align: left; }
  .funnel th { font-size: var(--fs-sm); font-weight: 400; color: var(--muted); }
  .funnel .num { width: 4rem; text-align: right; font-variant-numeric: tabular-nums; }
  .funnel-section h2 { margin-bottom: var(--sp-1); }
  .funnel-section > .muted { margin: 0 0 var(--sp-2); }

  /* El hallazgo es la superficie compartida; lo único propio es la marca de urgencia. */
  .finding { margin: var(--sp-2) 0; border-width: 2px; border-left-width: 3px; }
  .finding.urgent { border-left-color: var(--danger); }

  /* Título a la izquierda, estado a la derecha: los tags se leen en columna. */
  .finding-head {
    display: flex; align-items: baseline; justify-content: space-between;
    gap: var(--sp-3); margin: 0;
    font-size: var(--fs-lg);
  }
  .finding-tags { flex: none; display: flex; gap: var(--sp-2); }
  .finding-fix { font-weight: 400; color: var(--muted); }

  .finding-summary { margin: var(--sp-2) 0 0; }
  .explanation { margin: var(--sp-3) 0 0; }
  .citations { margin: var(--sp-1) 0 0; font-size: var(--fs-sm); }

  /* Respaldo y acción en la misma línea: ninguno de los dos merece un renglón propio. */
  .finding-foot {
    display: flex; align-items: flex-end; justify-content: space-between;
    gap: var(--sp-3);
    margin-top: var(--sp-3);
  }
  /* min-width:0 deja que la meta envuelva por dentro en vez de empujar al botón. */
  .finding-meta { min-width: 0; color: var(--muted); font-size: var(--fs-sm); }
  .chat-button { flex: none; }

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
