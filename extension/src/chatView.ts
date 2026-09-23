/**
 * Chat general de la extensión, siempre disponible en la barra lateral.
 *
 * Reemplaza al chat que se abría por hallazgo ("Profundizar"). Ahora hay un único chat y
 * el usuario le **adjunta** hallazgos desde el informe, como se adjuntan archivos en
 * Copilot. Los adjuntos se ven como fichas arriba del input y son contexto persistente:
 * viajan en cada pregunta hasta que se quitan con la ×. Sin adjuntos es un chat común
 * contra el RAG.
 *
 * El pipeline no guarda estado. Cada request lleva el historial entero y, si hay
 * adjuntos, un primer mensaje `system` con `{"findings": [...]}`, que el pipeline lee
 * aparte (src/pipeline/chat/context.py) y pone en el prompt.
 *
 * Las respuestas son markdown no confiable: se renderizan en el host con
 * `renderMarkdown`, que escapa todo antes de dar formato.
 */

import * as vscode from "vscode";
import { renderMarkdown } from "./markdown";
import { chip } from "./references";
import { Finding, findingKey } from "./scan/types";
import { BASE_STYLES, escapeHtml as escape, nonceValue } from "./styles";

export interface ChatOptions {
  url: string;
  model: string;
  apiKey: string;
  timeoutMs: number;
}

interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

/** Lo que ya se dibujó en el hilo, para rehacerlo si VSCode recrea la vista. */
interface Entry {
  role: "user" | "assistant" | "error";
  html: string;
}

/**
 * Tope de adjuntos. Cada hallazgo suma su advisory al prompt y el generador local corre
 * con un contexto chico: más que esto deja sin lugar a los documentos del corpus.
 */
export const MAX_ATTACHMENTS = 6;

export class ChatView implements vscode.WebviewViewProvider, vscode.Disposable {
  static readonly viewType = "cibersec.chat";

  private view: vscode.WebviewView | undefined;
  private history: ChatMessage[] = [];
  private transcript: Entry[] = [];
  private readonly attachments = new Map<string, Finding>();
  private busy = false;
  private readonly changed = new vscode.EventEmitter<ReadonlySet<string>>();

  /** Avisa qué hallazgos quedaron adjuntos, para que el informe marque sus botones. */
  readonly onDidChangeAttachments = this.changed.event;

  constructor(private readonly options: () => ChatOptions) {}

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    view.webview.options = { enableScripts: true };
    view.webview.html = this.html();
    view.webview.onDidReceiveMessage(this.onMessage);
    view.onDidDispose(() => {
      if (this.view === view) this.view = undefined;
    });
  }

  attachedKeys(): ReadonlySet<string> {
    return new Set(this.attachments.keys());
  }

  /** Lo que hace el botón del informe: adjunta, o quita si ya estaba. */
  async toggle(finding: Finding): Promise<void> {
    const key = findingKey(finding);
    if (this.attachments.has(key)) {
      this.detach(key);
      return;
    }
    if (this.attachments.size >= MAX_ATTACHMENTS) {
      vscode.window.showWarningMessage(
        `El chat admite hasta ${MAX_ATTACHMENTS} hallazgos adjuntos. Quitá alguno antes de agregar otro.`,
      );
      return;
    }
    this.attachments.set(key, finding);
    this.contextChanged();
    await this.reveal();
  }

  detach(key: string): void {
    if (this.attachments.delete(key)) this.contextChanged();
  }

  /** Vacía historial y adjuntos: una conversación nueva no hereda el tema de la anterior. */
  newChat(): void {
    this.history = [];
    this.transcript = [];
    this.attachments.clear();
    this.changed.fire(this.attachedKeys());
    if (this.view) this.view.webview.html = this.html();
  }

  dispose(): void {
    this.changed.dispose();
  }

  private async reveal(): Promise<void> {
    if (this.view) {
      this.view.show(true);
    } else {
      // Primera vez: el comando `<id>.focus` que VSCode genera para cada vista la crea.
      await vscode.commands.executeCommand(`${ChatView.viewType}.focus`);
    }
  }

  private contextChanged(): void {
    this.changed.fire(this.attachedKeys());
    void this.view?.webview.postMessage({
      type: "context",
      html: renderAttachments([...this.attachments.values()]),
    });
  }

  private readonly onMessage = async (message: { type?: string; text?: string; url?: string; key?: string }) => {
    if (message.type === "openExternal" && message.url && /^https?:\/\//i.test(message.url)) {
      await vscode.env.openExternal(vscode.Uri.parse(message.url));
      return;
    }
    if (message.type === "remove" && message.key) {
      this.detach(message.key);
      return;
    }
    if (message.type !== "send" || !message.text?.trim() || this.busy) return;

    const text = message.text.trim();
    // Se congela lo adjunto al momento de preguntar: si el usuario quita una ficha
    // mientras espera, los IDs citados se comparan contra lo que efectivamente se mandó.
    const findings = [...this.attachments.values()];
    this.history.push({ role: "user", content: text });
    this.transcript.push({ role: "user", html: escape(text) });
    this.busy = true;
    try {
      const answer = await this.ask(findings);
      this.history.push({ role: "assistant", content: answer });
      const html = renderMarkdown(answer) + renderCitedIds(answer, findings);
      this.transcript.push({ role: "assistant", html });
      void this.view?.webview.postMessage({ type: "answer", html });
    } catch (error) {
      // El turno fallido sale del historial: si quedara, la próxima pregunta llevaría
      // dos mensajes de usuario seguidos y uno sin respuesta.
      this.history.pop();
      const text = `No se pudo consultar el pipeline: ${(error as Error).message}`;
      this.transcript.push({ role: "error", html: escape(text) });
      void this.view?.webview.postMessage({ type: "error", text });
    } finally {
      this.busy = false;
    }
  };

  private async ask(findings: Finding[]): Promise<string> {
    const { url, model, apiKey, timeoutMs } = this.options();
    // `system` y no `user`: es el tema de la charla, no algo que dijo el usuario.
    const context: ChatMessage[] = findings.length
      ? [{ role: "system", content: JSON.stringify({ findings: findings.map(contextFor) }) }]
      : [];
    const response = await fetch(`${trimSlash(url)}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({ model, stream: false, messages: [...context, ...this.history] }),
      signal: AbortSignal.timeout(timeoutMs),
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
    const thread = this.transcript.length
      ? this.transcript.map(renderEntry).join("")
      : `<p class="muted intro">Preguntá lo que quieras sobre ciberseguridad. Para hablar de
          hallazgos concretos, adjuntalos desde el informe del escaneo con el clip: van como
          contexto en cada pregunta hasta que los quites.</p>`;
    return `<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="${csp}">
<style>${BASE_STYLES}${CHAT_STYLES}</style>
</head>
<body>
<main>
  <div id="messages" aria-live="polite">${thread}${this.busy ? TYPING : ""}</div>
  <div class="composer">
    <div id="context">${renderAttachments([...this.attachments.values()])}</div>
    <form id="form">
      <textarea id="input" rows="1" placeholder="Preguntá algo (Enter envía, Shift+Enter nueva línea)" required${this.busy ? " disabled" : ""}></textarea>
      <button id="send" class="button" type="submit"${this.busy ? " disabled" : ""}>Enviar</button>
    </form>
  </div>
</main>
<script nonce="${nonce}">${CHAT_SCRIPT}</script>
</body>
</html>`;
  }
}

function contextFor(finding: Finding) {
  return {
    vulnerability: identifierOf(finding),
    package: finding.package,
    installed_version: finding.installed_version,
    fixed_version: finding.fixed_version,
    cwe_ids: finding.cwe_ids,
    cvss_score: finding.cvss_score,
    epss: finding.epss,
    kev: finding.kev,
    summary: finding.summary,
    details: finding.details,
  };
}

/** Las fichas de adjuntos y, desplegable, el JSON exacto que se manda al pipeline. */
function renderAttachments(findings: Finding[]): string {
  if (findings.length === 0) {
    return `<p class="muted hint">Sin hallazgos adjuntos.</p>`;
  }
  const items = findings
    .map((finding) => {
      const id = identifierOf(finding);
      const label = escape(`Quitar ${id} del contexto`);
      return `<span class="attachment" title="${escape(finding.summary ?? id)}">
        <span class="attachment-id">${escape(id)}</span>
        <span class="muted">${escape(finding.package)} ${escape(finding.installed_version)}</span>
        <button type="button" class="remove" data-remove="${escape(findingKey(finding))}" title="${label}" aria-label="${label}">×</button>
      </span>`;
    })
    .join("");
  const payload = { findings: findings.map(contextFor) };
  return `<div class="attachments" aria-label="Hallazgos adjuntos">${items}</div>
    <details class="loaded-context">
      <summary>Ver contexto enviado (${findings.length})</summary>
      <pre><code>${escape(JSON.stringify(payload, null, 2))}</code></pre>
    </details>`;
}

function renderEntry(entry: Entry): string {
  const extra = entry.role === "assistant" ? " card" : "";
  return `<div class="message ${entry.role}${extra}">${entry.html}</div>`;
}

/** CVE/CWE que aparecen en la respuesta, marcando los que no venían de los adjuntos. */
function renderCitedIds(answer: string, findings: Finding[]): string {
  const cited = unique((answer.match(/\b(?:CVE-\d{4}-\d{4,7}|CWE-\d{1,5})\b/gi) ?? []).map((id) => id.toUpperCase()));
  if (cited.length === 0) return "";
  const known = new Set(
    findings
      .flatMap((finding) => [finding.cve, ...finding.osv_ids, ...finding.aliases, ...finding.cwe_ids])
      .filter((id): id is string => Boolean(id))
      .map((id) => id.toUpperCase()),
  );
  const outside = cited.filter((id) => !known.has(id));
  const items = cited
    .map((id) => (known.has(id) ? chip(id) : `${chip(id)}<span class="tag warning outside" title="No viene de los hallazgos adjuntos: puede salir del corpus del RAG o ser inventado">fuera del contexto</span>`))
    .join(" ");
  return `<details class="cited">
    <summary class="muted">IDs citados en la respuesta (${cited.length}${outside.length ? `, ${outside.length} fuera del contexto` : ""})</summary>
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

const TYPING = `<div class="message typing" aria-label="El asistente está escribiendo"><span></span><span></span><span></span></div>`;

const CHAT_SCRIPT = `
const vscode = acquireVsCodeApi();
const form = document.getElementById("form");
const input = document.getElementById("input");
const send = document.getElementById("send");
const messages = document.getElementById("messages");
const context = document.getElementById("context");
const MAX_LINES = 5;

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
  const intro = messages.querySelector(".intro");
  if (intro) intro.remove();
  addMessage("user", (item) => { item.textContent = text; });
  input.value = "";
  autosize();
  setBusy(true);
  vscode.postMessage({ type: "send", text });
});
document.addEventListener("click", (event) => {
  const remove = event.target.closest("button[data-remove]");
  if (remove) {
    vscode.postMessage({ type: "remove", key: remove.dataset.remove });
    return;
  }
  const link = event.target.closest("a[href]");
  if (!link) return;
  event.preventDefault();
  vscode.postMessage({ type: "openExternal", url: link.getAttribute("href") });
});
window.addEventListener("message", (event) => {
  const message = event.data;
  if (message.type === "answer") { addMessage("assistant", (item) => { item.innerHTML = message.html; }); setBusy(false); }
  if (message.type === "error") { addMessage("error", (item) => { item.textContent = message.text; }); setBusy(false); }
  if (message.type === "context") { context.innerHTML = message.html; }
});
function setBusy(busy) {
  input.disabled = busy;
  send.disabled = busy;
  const typing = messages.querySelector(".typing");
  if (busy && !typing) {
    messages.insertAdjacentHTML("beforeend", ${JSON.stringify(TYPING)});
    messages.scrollTop = messages.scrollHeight;
  } else if (!busy) {
    if (typing) typing.remove();
    input.focus();
  }
}
function addMessage(role, fill) {
  const item = document.createElement("div");
  item.className = "message " + role + (role === "assistant" ? " card" : "");
  fill(item);
  const typing = messages.querySelector(".typing");
  messages.insertBefore(item, typing);
  messages.scrollTop = messages.scrollHeight;
}
messages.scrollTop = messages.scrollHeight;
if (!input.disabled) input.focus();
`;

/**
 * Sólo lo específico del chat: la columna de la barra lateral, el hilo de mensajes y las
 * fichas de adjuntos. Superficie, chips, tags, botones y escalas salen de BASE_STYLES.
 */
const CHAT_STYLES = `
  html, body { height: 100%; }
  body { overflow: hidden; padding: 0; }
  main {
    display: flex; flex-direction: column;
    height: 100vh;
    padding: 0 var(--sp-3) var(--sp-3);
  }

  #messages { flex: 1; min-height: 0; overflow-y: auto; padding-top: var(--sp-3); }
  .intro { margin: 0; }
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
  .message.assistant pre { overflow-x: auto; }
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
  .outside { margin: 0 var(--sp-2) 0 var(--sp-1); }

  /* Adjuntos + input: el bloque fijo de abajo, como el de Copilot. */
  .composer {
    flex: none;
    margin-top: var(--sp-2);
    padding: var(--sp-2);
    border: 1px solid var(--vscode-input-border, var(--border));
    border-radius: var(--radius);
    background: var(--vscode-input-background);
  }
  .composer:focus-within { border-color: var(--vscode-focusBorder); }
  .hint { margin: 0 0 var(--sp-1); font-size: var(--fs-sm); }
  .attachments { display: flex; flex-wrap: wrap; gap: var(--sp-1); margin-bottom: var(--sp-1); }
  .attachment {
    display: inline-flex; align-items: center; gap: var(--sp-1);
    max-width: 100%;
    padding: 0 0 0 var(--sp-2);
    font-size: var(--fs-sm);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--surface);
  }
  .attachment > span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .attachment-id { font-family: var(--vscode-editor-font-family); }
  .remove {
    flex: none;
    width: 1.5rem; height: 1.5rem;
    padding: 0;
    color: var(--muted);
    background: none; border: 0; border-radius: var(--radius-sm);
    cursor: pointer;
  }
  .remove:hover { color: var(--vscode-foreground); background: var(--vscode-toolbar-hoverBackground); }
  .remove:focus-visible { outline: 1px solid var(--vscode-focusBorder); }

  .loaded-context { margin-bottom: var(--sp-1); font-size: var(--fs-sm); }
  .loaded-context > summary { color: var(--vscode-textLink-foreground); cursor: pointer; }
  .loaded-context pre { max-height: 30vh; overflow: auto; }

  form { display: flex; align-items: flex-end; gap: var(--sp-2); }
  textarea {
    flex: 1; resize: none; overflow-y: hidden;
    padding: var(--sp-1) 0;
    font: inherit; line-height: 1.5;
    color: var(--vscode-input-foreground);
    background: transparent;
    border: 0;
    outline: none;
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
