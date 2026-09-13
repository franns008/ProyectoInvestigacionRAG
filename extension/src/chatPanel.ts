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
<main>
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
    .map((id) => (known.has(id) ? chip(id) : `${chip(id)}<span class="outside" title="No viene del escaneo: puede salir del corpus del RAG o ser inventado">fuera del escaneo</span>`))
    .join(" ");
  return `<details class="cited">
    <summary class="muted">IDs citados en la respuesta (${cited.length}${outside.length ? `, ${outside.length} fuera del escaneo` : ""})</summary>
    <p>${items}</p>
  </details>`;
}

function chip(id: string): string {
  const url = referenceUrl(id);
  return url
    ? `<a class="chip" href="${escape(url)}" title="${escape(url)}">${escape(id)}</a>`
    : `<span class="chip">${escape(id)}</span>`;
}

function referenceUrl(id: string): string | null {
  const cwe = id.match(/^CWE-(\d+)$/i);
  if (cwe) return `https://cwe.mitre.org/data/definitions/${cwe[1]}.html`;
  if (/^CVE-\d{4}-\d+$/i.test(id)) return `https://nvd.nist.gov/vuln/detail/${id.toUpperCase()}`;
  if (/^[A-Z]+-[\w-]+$/i.test(id)) return `https://osv.dev/vulnerability/${encodeURIComponent(id)}`;
  return null;
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

const CHAT_STYLES = `
  html, body { height: 100%; }
  body { margin: 0; padding: 0; max-width: none; overflow: hidden; }
  main {
    box-sizing: border-box;
    display: flex; flex-direction: column;
    height: 100vh; max-width: 60rem;
    padding: 1.5rem 2rem 1rem;
  }
  header { flex: none; }

  details > summary { cursor: pointer; }
  .loaded-context { margin: .15rem 0 0; }
  .loaded-context > summary {
    display: inline-block; list-style: none;
    font-size: .85em; color: var(--vscode-textLink-foreground);
  }
  .loaded-context > summary::-webkit-details-marker { display: none; }
  .loaded-context > summary::before { content: "▸ "; }
  .loaded-context[open] > summary::before { content: "▾ "; }
  .loaded-context > summary:hover { text-decoration: underline; }
  .context-panel { max-height: 40vh; overflow-y: auto; }
  .context-row { display: flex; gap: .75rem; margin: .3rem 0; flex-wrap: wrap; }
  .context-row > .muted { min-width: 8rem; }
  .raw { margin-top: .6rem; }

  .chip {
    font-family: var(--vscode-editor-font-family); font-size: .85em;
    border: 1px solid var(--vscode-panel-border); border-radius: 2px;
    padding: .05rem .35rem; text-decoration: none; white-space: nowrap;
  }
  .outside {
    color: var(--vscode-editorWarning-foreground);
    font-size: .75rem; margin: 0 .5rem 0 .25rem;
  }

  #messages { flex: 1; min-height: 0; overflow-y: auto; margin: 1rem 0 .75rem; }
  .intro { border-bottom: 1px solid var(--vscode-panel-border); padding-bottom: .75rem; }
  .message { margin: .6rem 0; }
  .message.user {
    white-space: pre-wrap;
    padding: .6rem .85rem; border-radius: 3px;
    background: var(--vscode-textBlockQuote-background);
    border-left: 3px solid var(--vscode-textLink-foreground);
  }
  .message.error { color: var(--vscode-errorForeground); white-space: pre-wrap; }
  .message.assistant > :first-child { margin-top: 0; }
  .message.assistant ul, .message.assistant ol { margin: .4rem 0; padding-left: 1.4rem; }
  .message.assistant li { margin: .15rem 0; }
  .message.assistant h2, .message.assistant h3 { margin: .9rem 0 .4rem; }
  .message.assistant blockquote {
    margin: .5rem 0; padding: .1rem .85rem;
    border-left: 3px solid var(--vscode-textBlockQuote-border);
    background: var(--vscode-textBlockQuote-background);
  }
  .message.assistant :not(pre) > code {
    background: var(--vscode-textCodeBlock-background); padding: .05rem .3rem; border-radius: 2px;
  }
  .message.assistant hr { border: 0; border-top: 1px solid var(--vscode-panel-border); }
  .table { overflow-x: auto; }
  .message.assistant table { border-collapse: collapse; margin: .5rem 0; }
  .message.assistant th, .message.assistant td {
    border: 1px solid var(--vscode-panel-border); padding: .25rem .5rem; text-align: left;
  }
  .cited { margin-top: .75rem; font-size: .9em; border-top: 1px solid var(--vscode-panel-border); padding-top: .5rem; }

  form { flex: none; display: flex; gap: .5rem; align-items: flex-end; }
  textarea {
    flex: 1; resize: none; overflow-y: hidden; box-sizing: border-box;
    color: var(--vscode-input-foreground); background: var(--vscode-input-background);
    border: 1px solid var(--vscode-input-border, var(--vscode-panel-border)); border-radius: 3px;
    padding: .4rem .6rem; font: inherit; line-height: 1.4;
  }
  form .button { padding-block: .4rem; line-height: 1.4; }
  textarea:focus { outline: 1px solid var(--vscode-focusBorder); outline-offset: -1px; }
  .typing {
    display: inline-flex; gap: .3rem; align-items: center;
    padding: .7rem .9rem; border-radius: 3px;
    border: 1px solid var(--vscode-panel-border);
  }
  .typing span {
    width: .45rem; height: .45rem; border-radius: 50%;
    background: var(--vscode-descriptionForeground);
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
