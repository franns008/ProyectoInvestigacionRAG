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
import { ChatView } from './chatView';
import { PreviewPackage } from './scan/manifest';
import { chip } from './references';
import { Finding, ScanResult, findingKey } from './scan/types';
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

    private constructor(private readonly chat: ChatView) {
        this.panel = vscode.window.createWebviewPanel(
            'cibersec.results',
            'Dependencias vulnerables',
            vscode.ViewColumn.Beside,
            { enableScripts: true, retainContextWhenHidden: true },
        );
        this.panel.webview.onDidReceiveMessage(
            (message: { type?: string; key?: string; url?: string }) => {
                if (message.type === 'toggleContext' && message.key) {
                    const finding = this.findings.get(message.key);
                    if (finding) void this.chat.toggle(finding);
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
        // Los botones reflejan qué está adjunto, también cuando se quita desde el chat.
        this.disposables.push(
            chat.onDidChangeAttachments((keys) => {
                void this.panel.webview.postMessage({ type: 'attached', keys: [...keys] });
            }),
        );
    }

    /** Reusa el panel si ya está abierto, para no acumular pestañas en cada escaneo. */
    static show(chat: ChatView): ResultsPanel {
        if (!ResultsPanel.current) {
            ResultsPanel.current = new ResultsPanel(chat);
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
  // Marca los botones de los hallazgos que ya están en el chat. El estado inicial va
  // embebido porque un postMessage anterior a la carga del webview se pierde.
  function markAttached(keys) {
    const attached = new Set(keys);
    for (const button of document.querySelectorAll("button[data-finding]")) {
      const on = attached.has(button.dataset.finding);
      button.setAttribute("aria-pressed", String(on));
      const name = on ? button.dataset.labelOn : button.dataset.labelOff;
      button.title = name;
      button.setAttribute("aria-label", name);
      const text = button.querySelector(".label");
      if (text) text.textContent = on ? "En el chat" : "Agregar al chat";
    }
  }
  document.addEventListener("DOMContentLoaded", () => markAttached(${JSON.stringify([...this.chat.attachedKeys()])}));
  window.addEventListener("message", (event) => {
    if (event.data.type === "attached") markAttached(event.data.keys);
  });
  document.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-finding]");
    if (button) {
      vscode.postMessage({ type: "toggleContext", key: button.dataset.finding });
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

/**
 * Agrupa por librería conservando el orden en que llegan los hallazgos.
 *
 * Acá se los reordenaba por CVSS descendente, con el CVSS ausente valiendo -1. Eso
 * hundía justo lo que hay que mirar primero: en el grupo de pyspark, CVE-2022-33891
 * —que CISA confirma explotada, EPSS 93%— caía tercera debajo de una con CVSS 9.9 y
 * EPSS 1,1%, y CVE-2023-32007, con EPSS 76%, quedaba última de once por no tener CVSS
 * publicado. La vista terminaba ordenando por la métrica que el proyecto demuestra que
 * no ordena el trabajo (ver el embudo, arriba en esta misma pantalla).
 *
 * `prioritize()` en Python ya los manda ordenados por KEV, después EPSS y CVSS sólo
 * como desempate. El Map conserva el orden de inserción, así que no reordenar es
 * respetar esa decisión — y es la regla que fija types.ts: los campos duros los arma
 * Python, la vista no calcula.
 */
function renderPackageGroups(findings: Finding[]): string {
    const groups = new Map<string, Finding[]>();
    for (const finding of findings) {
        const key = finding.package.toLowerCase().replace(/[-_.]+/g, '-');
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
 * porcentajes, y por eso era la única sin uno. No hace falta repetirlo acá — está dos
 * renglones más arriba, en la línea de resumen del encabezado.
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
 * La acción de la tarjeta: adjuntar el hallazgo al chat, o quitarlo si ya está.
 *
 * En "Explotadas activamente" son siete tarjetas y la acción es la esperable, así que
 * lleva ícono y etiqueta. En los grupos de librería se repite en las ciento quince: ahí
 * va sólo el ícono, porque un botón con texto repetido esa cantidad de veces termina
 * pesando más que los hallazgos.
 *
 * En los dos casos el nombre accesible incluye el identificador. Sin eso serían ciento
 * veintidós botones llamados igual, y el de sólo ícono no tendría nombre en absoluto.
 * `aria-pressed` y los nombres los ajusta el script según lo que esté adjunto.
 */
function renderAction(finding: Finding, seccion: Seccion): string {
    const id = `${finding.cve ?? finding.osv_ids[0] ?? 'este hallazgo'} de ${finding.package}`;
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

/**
 * El resumen del advisory, sin el prefijo con el nombre del paquete.
 *
 * 21 de los 115 hallazgos del manifiesto de cobertura arrancan con "Marimo: ", "Ray: "
 * y similares. El paquete ya es el título de la tarjeta.
 */
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
 * La explicación del LLM, con el resumen original del advisory debajo.
 *
 * El resumen no se descarta: casi siempre es la fuente de la explicación (el CVE rara
 * vez está en el corpus y el advisory entra siempre al contexto), así que es contra lo
 * que se contrasta la prosa del modelo. Va debajo y como cita, no como segunda
 * descripción: es respaldo, no contenido.
 *
 * La línea de "Citado:" se sacó: el advisory citado es casi siempre el mismo que se
 * muestra entre comillas, y los identificadores ya están en los chips del pie.
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

/** Identifica un hallazgo entre el webview y la extensión. `scan` dedupe por este par. */
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
    /* .package-group recorta (overflow:hidden), así que sin esto un nombre de paquete
       largo se corta contra el borde en vez de envolver. */
    overflow-wrap: anywhere;
  }
  .package-group > summary:hover { background: var(--vscode-list-hoverBackground); }
  .package-findings { padding: 0 var(--sp-3) var(--sp-2); }

  /* Mismo relleno que las tarjetas: el resumen es un bloque más, no texto suelto. */
  .funnel-section {
    margin-top: var(--sp-4);
    padding: var(--sp-3) var(--sp-4);
    border-radius: var(--radius);
    background: var(--card-bg);
    /* La tabla no envuelve: cuando no entra, scrollea adentro en vez de salirse. */
    overflow-x: auto;
  }
  /* Todo el ancho de la caja: el criterio a la izquierda y las cifras contra el borde
     derecho, con el mismo padding de los dos lados. */
  .funnel { width: 100%; margin: var(--sp-2) 0 0; border-collapse: collapse; }
  /* Sin envolver: si el panel es angosto, la caja scrollea (overflow-x en .funnel-section). */
  .funnel th, .funnel td {
    padding: var(--sp-1) var(--sp-2) var(--sp-1) 0;
    text-align: left;
    white-space: nowrap;
  }
  .funnel th:last-child, .funnel td:last-child { padding-right: 0; }
  .funnel th { font-size: var(--fs-sm); font-weight: 400; color: var(--muted); }
  /* width:1% encoge las cifras a su contenido y deja todo el sobrante al criterio. El
     padding izquierdo las separa de él. */
  .funnel .num {
    width: 1%;
    padding-left: var(--sp-4);
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
  .funnel-section h2 { margin-bottom: var(--sp-2); }

  /* El hallazgo es la superficie compartida; lo único propio es la marca de urgencia. */
  .finding { margin: var(--sp-2) 0; border-width: 1.5px; border-left-width: 2.5px; }
  .finding.urgent { border-left-color: var(--danger); }

  /* Título a la izquierda, estado a la derecha: los tags se leen en columna. */
  .finding-head {
    display: flex; align-items: baseline; justify-content: space-between;
    gap: var(--sp-3); margin: 0;
    font-size: var(--fs-lg);
    font-weight: 700;
  }
  /* Mismo criterio que .finding-meta en el pie: el título envuelve por dentro en vez de
     empujar a los tags fuera de la tarjeta. Un ítem flex tiene min-width:auto, así que
     sin esto el nombre del paquete impone su ancho mínimo y desborda la línea entera.
     overflow-wrap cubre el caso en que un solo token ya no entra: min-width:0 deja que
     el span se encoja, pero no parte la palabra. */
  .finding-title { min-width: 0; overflow-wrap: anywhere; }
  .finding-tags { flex: none; display: flex; gap: var(--sp-2); }
  .finding-fix { font-weight: 400; color: var(--muted); }

  .finding-summary { margin: var(--sp-2) 0 0; }
  .explanation { margin: var(--sp-3) 0 0; }
  .explanation > p:first-child { margin-top: 0; }
  /* El texto del advisory, tal cual: se lee como fuente, más quieto que la explicación. */
  .advisory-original {
    margin: var(--sp-2) 0 0;
    padding-left: var(--sp-3);
    border-left: 2px solid var(--border-strong);
    color: var(--muted);
    font-size: var(--fs-sm);
  }

  /* Respaldo y acción en la misma línea: ninguno de los dos merece un renglón propio. */
  .finding-foot {
    display: flex; align-items: flex-end; justify-content: space-between;
    gap: var(--sp-3);
    margin-top: var(--sp-3);
  }
  /* min-width:0 deja que la meta envuelva por dentro en vez de empujar al botón. */
  .finding-meta { min-width: 0; color: var(--muted); font-size: var(--fs-sm); }
  .chat-button { flex: none; }

  /* El tratamiento discreto lo pone .button.quiet; acá va sólo la forma. */
  .chat-button { display: inline-flex; align-items: center; gap: .4em; }
  /* Sólo ícono: caja cuadrada, para que no quede un botón ancho con un glifo al medio. */
  .chat-button[data-icon-only] {
    justify-content: center;
    width: 1.9rem; height: 1.9rem;
    padding: 0;
  }
  .icono { flex: none; }
  /* Adjunto: el botón queda marcado, como un toggle de la barra de herramientas. */
  .chat-button[aria-pressed="true"] {
    color: var(--vscode-inputOption-activeForeground, var(--vscode-foreground));
    background: var(--vscode-inputOption-activeBackground, var(--surface));
    box-shadow: inset 0 0 0 1px var(--vscode-inputOption-activeBorder, var(--border-strong));
  }

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

  /* --- Panel angosto ------------------------------------------------------------ *
   *
   * Mismo umbral que en BASE_STYLES, donde la página achica su margen.
   *
   * Las dos líneas de la tarjeta —título contra tags, respaldo contra acción— son
   * pares de "contenido a la izquierda, estado a la derecha". El de la derecha no
   * puede encogerse (los tags y el botón son cajas de tamaño fijo) y el de la
   * izquierda ya envuelve todo lo que puede, así que por debajo de cierto ancho la
   * línea no cierra y lo fijo termina fuera de la tarjeta.
   *
   * Envolver dentro de la misma línea no alcanza: con space-between, lo que baja al
   * segundo renglón queda pegado a la izquierda, o sea en columna pero a medias y con
   * el alineado roto. Se pasa a columna de una vez, que es lo que efectivamente es.
   * Se pierde el barrido vertical por la derecha, pero a este ancho ese barrido no
   * existía: cabe una tarjeta y se lee de arriba a abajo.
   */
  @media (max-width: 30rem) {
    .finding-head, .finding-foot, .section-heading {
      flex-direction: column;
      align-items: flex-start;
      gap: var(--sp-2);
    }
    /* Dos tags no siempre entran juntos a este ancho. */
    .finding-tags { flex-wrap: wrap; }
    .section-heading { gap: var(--sp-1); }
    /* Ya no compite con la meta por la línea: puede ocupar el ancho que necesite. */
    .chat-button { max-width: 100%; }
  }
`;
