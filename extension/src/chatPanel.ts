/**
 * Chat sobre un hallazgo ("Hablar en profundidad").
 *
 * El primer mensaje de la conversación es el contexto del escaneo (`contextFor`). El
 * panel lo muestra en "Ver contexto cargado" y, debajo de cada respuesta, lista los
 * CVE/CWE que citó el modelo, marcando los que no venían del escaneo: pueden salir del
 * corpus del RAG o ser inventados, y eso es justo lo que conviene poder revisar.
 *
 * Las respuestas son markdown no confiable: se renderizan en el host con
 * `renderMarkdown`, que escapa todo antes de dar formato.
 */

import * as vscode from "vscode";
import { renderMarkdown } from "./markdown";
import { chip } from "./references";
import { Finding } from "./scan/types";
import { BASE_STYLES, escapeHtml as escape, nonceValue } from "./styles";

export interface ChatOptions {
  url: string;
  model: string;
  apiKey: string;
  timeoutMs: number;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export class ChatPanel {
  private static current: ChatPanel | undefined;
  private readonly panel: vscode.WebviewPanel;
  private readonly messages: ChatMessage[];
  private disposables: vscode.Disposable[] = [];

  private constructor(private readonly finding: Finding, private readonly options: ChatOptions) {
    this.messages = [{ role: "user", content: JSON.stringify(contextFor(finding)) }];
    this.panel = vscode.window.createWebviewPanel(
      "cibersec.vulnerabilityChat",
      `Chat: ${identifierOf(finding)}`,
      vscode.ViewColumn.Beside,
      { enableScripts: true, retainContextWhenHidden: true },
    );
    this.panel.webview.onDidReceiveMessage(this.onMessage, this, this.disposables);
    this.panel.onDidDispose(() => this.dispose(), null, this.disposables);
    this.panel.webview.html = this.html();
  }

  static show(finding: Finding, options: ChatOptions): void {
    if (ChatPanel.current) {
      if (identifierOf(ChatPanel.current.finding) === identifierOf(finding)) {
        ChatPanel.current.panel.reveal(vscode.ViewColumn.Beside, true);
        return;
      }
      ChatPanel.current.panel.dispose();
    }
    ChatPanel.current = new ChatPanel(finding, options);
  }

  private readonly onMessage = async (message: { type?: string; text?: string; url?: string }) => {
    if (message.type === "openExternal" && message.url && /^https?:\/\//i.test(message.url)) {
      await vscode.env.openExternal(vscode.Uri.parse(message.url));
      return;
    }
    if (message.type !== "send" || !message.text?.trim()) return;
    const text = message.text.trim();
    this.messages.push({ role: "user", content: text });
    try {
      const answer = await this.ask();
      this.messages.push({ role: "assistant", content: answer });
      this.panel.webview.postMessage({
        type: "answer",
        html: renderMarkdown(answer) + renderCitedIds(answer, this.finding),
      });
    } catch (error) {
      this.panel.webview.postMessage({
        type: "error",
        text: `No se pudo consultar el pipeline: ${(error as Error).message}`,
      });
    }
  };

  private async ask(): Promise<string> {
    const response = await fetch(`${trimSlash(this.options.url)}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.options.apiKey}`,
      },
      body: JSON.stringify({ model: this.options.model, stream: false, messages: this.messages }),
      signal: AbortSignal.timeout(this.options.timeoutMs),
    });
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    const body = (await response.json()) as {
      choices?: { message?: { content?: string } }[];
    };
    const answer = body.choices?.[0]?.message?.content;
    if (!answer) throw new Error("la respuesta no trae contenido");
    return answer;
  }

  private html(): string {
    const nonce = nonceValue();
    const csp = `default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';`;
    const finding = this.finding;
    const fix = finding.fixed_version
      ? `arreglada en <strong>${escape(finding.fixed_version)}</strong>`
      : `<span class="muted">sin versión de arreglo publicada</span>`;
    return `<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="${csp}">
<style>${BASE_STYLES}${CHAT_STYLES}</style>
</head>
<body>
<main class="page">
  <header>
    <h1>${escape(identifierOf(finding))}</h1>
    <p class="muted">${escape(finding.package)} ${escape(finding.installed_version)} — ${fix}</p>
    ${renderLoadedContext(finding)}
  </header>
  <div id="messages" aria-live="polite">
    <p class="muted intro">Preguntá lo que quieras sobre esta vulnerabilidad. El contexto del escaneo ya está cargado.</p>
  </div>
  <form id="form">
    <textarea id="input" rows="1" placeholder="Escribí tu pregunta (Enter envía, Shift+Enter nueva línea)" required></textarea>
    <button id="send" class="button" type="submit">Enviar</button>
  </form>
</main>
<script nonce="${nonce}">${CHAT_SCRIPT}</script>
</body>
</html>`;
  }

  private dispose(): void {
    ChatPanel.current = undefined;
    this.disposables.forEach((disposable) => disposable.dispose());
    this.disposables = [];
  }
}

function contextFor(finding: Finding) {
  return {
    vulnerability: identifierOf(finding),
    package: finding.package,
    installed_version: finding.installed_version,
    fixed_version: finding.fixed_version,
    cwe_ids: finding.cwe_ids,
    summary: finding.summary,
    details: finding.details,
  };
}

/** Panel desplegable con lo que la extensión le mandó al pipeline como primer mensaje. */
function renderLoadedContext(finding: Finding): string {
  const vulnIds = unique([finding.cve, ...finding.osv_ids, ...finding.aliases]);
  const row = (label: string, value: string) =>
    value ? `<div class="context-row"><span class="muted">${label}</span><span>${value}</span></div>` : "";
  return `<details class="loaded-context">
    <summary>Ver contexto cargado</summary>
    <div class="card context-panel">
      ${row("Vulnerabilidad", vulnIds.map(chip).join(" "))}
      ${row("Debilidad (CWE)", finding.cwe_ids.map(chip).join(" "))}
      ${row("Paquete", `<code>${escape(finding.package)} ${escape(finding.installed_version)}</code>`)}
      ${row("Arreglo", finding.fixed_version ? `<code>${escape(finding.fixed_version)}</code>` : "")}
      <details class="raw">
        <summary class="muted">JSON enviado al pipeline</summary>
        <pre><code>${escape(JSON.stringify(contextFor(finding), null, 2))}</code></pre>
      </details>
    </div>
  </details>`;
}

/** CVE/CWE que aparecen en la respuesta, marcando los que no venían del escaneo. */
function renderCitedIds(answer: string, finding: Finding): string {
  const cited = unique((answer.match(/\b(?:CVE-\d{4}-\d{4,7}|CWE-\d{1,5})\b/gi) ?? []).map((id) => id.toUpperCase()));
  if (cited.length === 0) return "";
  const known = new Set(
    [finding.cve, ...finding.osv_ids, ...finding.aliases, ...finding.cwe_ids]
      .filter((id): id is string => Boolean(id))
      .map((id) => id.toUpperCase()),
  );
  const outside = cited.filter((id) => !known.has(id));
  const items = cited
    .map((id) => (known.has(id) ? chip(id) : `${chip(id)}<span class="tag warning outside" title="No viene del escaneo: puede salir del corpus del RAG o ser inventado">fuera del escaneo</span>`))
    .join(" ");
  return `<details class="cited">
    <summary class="muted">IDs citados en la respuesta (${cited.length}${outside.length ? `, ${outside.length} fuera del escaneo` : ""})</summary>
    <p>${items}</p>
  </details>`;
}

function unique(values: (string | null | undefined)[]): string[] {
  return [...new Set(values.filter((value): value is string => Boolean(value)))];
}

function identifierOf(finding: Finding): string {
  return finding.cve ?? finding.osv_ids[0] ?? "sin identificador";
}

function trimSlash(url: string): string {
  return url.replace(/\/+$/, "");
}

const CHAT_SCRIPT = `
const vscode = acquireVsCodeApi();
const form = document.getElementById("form");
const input = document.getElementById("input");
const send = document.getElementById("send");
const messages = document.getElementById("messages");
let typing = null;
const MAX_LINES = 3;

// Arranca en una línea y crece con el contenido hasta MAX_LINES; de ahí en más
// scrollea, dejando visibles las últimas líneas (el overflow queda arriba).
function autosize() {
  const style = getComputedStyle(input);
  const borders = parseFloat(style.borderTopWidth) + parseFloat(style.borderBottomWidth);
  const padding = parseFloat(style.paddingTop) + parseFloat(style.paddingBottom);
  const max = parseFloat(style.lineHeight) * MAX_LINES + padding + borders;
  input.style.height = "auto";
  const needed = input.scrollHeight + borders;
  input.style.height = Math.min(needed, max) + "px";
  input.style.overflowY = needed > max ? "auto" : "hidden";
  if (input.selectionEnd === input.value.length) input.scrollTop = input.scrollHeight;
}
input.addEventListener("input", autosize);
autosize();

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});
form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text || input.disabled) return;
  addMessage("user", (item) => { item.textContent = text; });
  input.value = "";
  autosize();
  setBusy(true);
  vscode.postMessage({ type: "send", text });
});
document.addEventListener("click", (event) => {
  const link = event.target.closest("a[href]");
  if (!link) return;
  event.preventDefault();
  vscode.postMessage({ type: "openExternal", url: link.getAttribute("href") });
});
window.addEventListener("message", (event) => {
  const message = event.data;
  if (message.type === "answer") { addMessage("assistant", (item) => { item.innerHTML = message.html; }); setBusy(false); }
  if (message.type === "error") { addMessage("error", (item) => { item.textContent = message.text; }); setBusy(false); }
});
function setBusy(busy) {
  input.disabled = busy;
  send.disabled = busy;
  if (busy) {
    typing = document.createElement("div");
    typing.className = "message typing";
    typing.setAttribute("aria-label", "El asistente está escribiendo");
    typing.innerHTML = "<span></span><span></span><span></span>";
    messages.appendChild(typing);
    messages.scrollTop = messages.scrollHeight;
  } else {
    if (typing) typing.remove();
    typing = null;
    input.focus();
  }
}
function addMessage(role, fill) {
  const item = document.createElement("div");
  item.className = "message " + role + (role === "assistant" ? " card" : "");
  fill(item);
  messages.appendChild(item);
  messages.scrollTop = messages.scrollHeight;
}
input.focus();
`;

/**
 * Sólo lo específico del chat: el layout de columna a pantalla completa y el hilo de
 * mensajes. Superficie, chips, tags, botones y escalas salen de BASE_STYLES.
 */
const CHAT_STYLES = `
  html, body { height: 100%; }
  body { overflow: hidden; }
  main {
    display: flex; flex-direction: column;
    height: 100vh;
    padding-bottom: var(--sp-4);
  }
  header { flex: none; }

  /* Desplegable discreto: es una ayuda, no una sección del documento. */
  .loaded-context { margin-top: var(--sp-1); }
  .loaded-context > summary {
    display: inline-block; list-style: none;
    font-size: var(--fs-sm); color: var(--vscode-textLink-foreground);
  }
  .loaded-context > summary::-webkit-details-marker { display: none; }
  .loaded-context > summary::before { content: "▸ "; }
  .loaded-context[open] > summary::before { content: "▾ "; }
  .loaded-context > summary:hover { text-decoration: underline; }
  .context-panel { margin-top: var(--sp-2); max-height: 40vh; overflow-y: auto; }
  .context-row { display: flex; flex-wrap: wrap; gap: var(--sp-3); margin: var(--sp-1) 0; }
  .context-row > .muted { min-width: 8rem; }
  .raw { margin-top: var(--sp-2); }
  .raw > summary { font-size: var(--fs-sm); }

  .outside { margin: 0 var(--sp-2) 0 var(--sp-1); }

  #messages { flex: 1; min-height: 0; overflow-y: auto; margin: var(--sp-4) 0 var(--sp-3); }
  .intro { padding-bottom: var(--sp-3); border-bottom: 1px solid var(--border); }
  .message { margin: var(--sp-2) 0; }
  .message.user {
    white-space: pre-wrap;
    padding: var(--sp-2) var(--sp-3);
    border-left: 2px solid var(--vscode-textLink-foreground);
    border-radius: var(--radius-sm);
    background: var(--vscode-textBlockQuote-background);
  }
  .message.error { color: var(--vscode-errorForeground); white-space: pre-wrap; }

  /* Markdown de la respuesta: hereda del sistema y sólo ajusta el ritmo vertical. */
  .message.assistant > :first-child { margin-top: 0; }
  .message.assistant > :last-child { margin-bottom: 0; }
  .message.assistant ul, .message.assistant ol { margin: var(--sp-2) 0; padding-left: var(--sp-5); }
  .message.assistant li { margin: var(--sp-1) 0; }
  .message.assistant h2, .message.assistant h3 { margin: var(--sp-4) 0 var(--sp-2); }
  .message.assistant blockquote {
    margin: var(--sp-2) 0;
    padding: var(--sp-1) var(--sp-3);
    border-left: 2px solid var(--vscode-textBlockQuote-border);
    background: var(--vscode-textBlockQuote-background);
  }
  .message.assistant :not(pre) > code {
    padding: 0 .3em;
    border-radius: var(--radius-sm);
    background: var(--surface);
  }
  .message.assistant hr { border: 0; border-top: 1px solid var(--border); }
  .table { overflow-x: auto; }
  .message.assistant table { margin: var(--sp-2) 0; border-collapse: collapse; }
  .message.assistant th, .message.assistant td {
    padding: var(--sp-1) var(--sp-2);
    border: 1px solid var(--border);
    text-align: left;
  }
  .cited {
    margin-top: var(--sp-3);
    padding-top: var(--sp-2);
    border-top: 1px solid var(--border);
    font-size: var(--fs-sm);
  }

  form { flex: none; display: flex; align-items: flex-end; gap: var(--sp-2); }
  textarea {
    flex: 1; resize: none; overflow-y: hidden;
    padding: var(--sp-1) var(--sp-2);
    font: inherit; line-height: 1.5;
    color: var(--vscode-input-foreground);
    background: var(--vscode-input-background);
    border: 1px solid var(--vscode-input-border, var(--border));
    border-radius: var(--radius-sm);
  }
  textarea::placeholder { color: var(--vscode-input-placeholderForeground, var(--muted)); }
  form .button { padding-block: var(--sp-1); }

  .typing {
    display: inline-flex; align-items: center; gap: var(--sp-1);
    padding: var(--sp-2) var(--sp-3);
    border: 1px solid var(--border);
    border-radius: var(--radius);
  }
  .typing span {
    width: .45rem; height: .45rem; border-radius: 50%;
    background: var(--muted);
    animation: typing 1.2s infinite ease-in-out;
  }
  .typing span:nth-child(2) { animation-delay: .2s; }
  .typing span:nth-child(3) { animation-delay: .4s; }
  @keyframes typing {
    0%, 60%, 100% { opacity: .3; transform: translateY(0); }
    30% { opacity: 1; transform: translateY(-.25rem); }
  }
  @media (prefers-reduced-motion: reduce) { .typing span { animation: none; opacity: .6; } }
`;
