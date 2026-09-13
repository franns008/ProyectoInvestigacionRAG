import * as vscode from "vscode";
import { Finding } from "./scan/types";

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
    const id = finding.cve ?? finding.osv_ids[0] ?? "sin identificador";
    this.messages = [{ role: "user", content: contextFor(finding) }];
    this.panel = vscode.window.createWebviewPanel(
      "cibersec.vulnerabilityChat",
      `Chat: ${id}`,
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

  private readonly onMessage = async (message: { type?: string; text?: string }) => {
    if (message.type !== "send" || !message.text?.trim()) return;
    const text = message.text.trim();
    this.messages.push({ role: "user", content: text });
    this.panel.webview.postMessage({ type: "status", text: "Consultando el RAG..." });
    try {
      const answer = await this.ask();
      this.messages.push({ role: "assistant", content: answer });
      this.panel.webview.postMessage({ type: "answer", text: answer });
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
    const nonce = Math.random().toString(36).slice(2);
    const id = this.finding.cve ?? this.finding.osv_ids[0] ?? "sin identificador";
    const csp = `default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';`;
    return `<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8">
      <meta http-equiv="Content-Security-Policy" content="${csp}"><style>${CHAT_STYLES}</style>
      </head><body><main><h1>${escape(id)}</h1>
      <p class="muted">${escape(this.finding.package)} ${escape(this.finding.installed_version)}</p>
      <div id="messages" aria-live="polite"><p class="context">Contexto cargado desde el escaneo. Preguntá lo que quieras sobre esta vulnerabilidad.</p></div>
      <form id="form"><textarea id="input" rows="3" placeholder="Escribí tu pregunta..." required></textarea>
      <button type="submit">Enviar</button></form><p id="status" class="muted"></p></main>
      <script nonce="${nonce}">${CHAT_SCRIPT}</script></body></html>`;
  }

  private dispose(): void {
    ChatPanel.current = undefined;
    this.disposables.forEach((disposable) => disposable.dispose());
    this.disposables = [];
  }
}

function contextFor(finding: Finding): string {
  return JSON.stringify({
    vulnerability: finding.cve ?? finding.osv_ids[0] ?? "sin identificador",
    package: finding.package,
    installed_version: finding.installed_version,
    fixed_version: finding.fixed_version,
    cwe_ids: finding.cwe_ids,
    summary: finding.summary,
    details: finding.details,
  });
}

function identifierOf(finding: Finding): string {
  return finding.cve ?? finding.osv_ids[0] ?? "sin identificador";
}

function trimSlash(url: string): string { return url.replace(/\/+$/, ""); }
function escape(value: string): string {
  return value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

const CHAT_SCRIPT = `
const vscode = acquireVsCodeApi();
const form = document.getElementById("form");
const input = document.getElementById("input");
const messages = document.getElementById("messages");
const status = document.getElementById("status");
form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  addMessage(text, "user"); input.value = ""; input.disabled = true;
  vscode.postMessage({ type: "send", text });
});
window.addEventListener("message", (event) => {
  const message = event.data;
  if (message.type === "answer") { addMessage(message.text, "assistant"); input.disabled = false; input.focus(); status.textContent = ""; }
  if (message.type === "status") status.textContent = message.text;
  if (message.type === "error") { addMessage(message.text, "error"); input.disabled = false; status.textContent = ""; }
});
function addMessage(text, role) { const item = document.createElement("p"); item.className = "message " + role; item.textContent = text; messages.appendChild(item); item.scrollIntoView({ block: "nearest" }); }
`;

const CHAT_STYLES = `
  body { font-family: var(--vscode-font-family); color: var(--vscode-foreground); padding: 1.5rem; }
  main { max-width: 52rem; margin: 0 auto; }
  h1 { font-size: 1.3rem; margin: 0; }
  .muted { color: var(--vscode-descriptionForeground); }
  #messages { min-height: 18rem; margin: 1.5rem 0; }
  .context, .message { white-space: pre-wrap; padding: .75rem; border-radius: 3px; margin: .6rem 0; }
  .context { color: var(--vscode-descriptionForeground); border-bottom: 1px solid var(--vscode-panel-border); }
  .message.user { background: var(--vscode-textBlockQuote-background); border-left: 3px solid var(--vscode-textLink-foreground); }
  .message.assistant { background: var(--vscode-editor-inactiveSelectionBackground); }
  .message.error { color: var(--vscode-errorForeground); }
  form { display: flex; gap: .5rem; align-items: flex-end; }
  textarea { flex: 1; resize: vertical; color: var(--vscode-input-foreground); background: var(--vscode-input-background); border: 1px solid var(--vscode-input-border); padding: .6rem; font: inherit; }
  button { color: var(--vscode-button-foreground); background: var(--vscode-button-background); border: 0; border-radius: 3px; padding: .6rem .9rem; cursor: pointer; }
  button:hover { background: var(--vscode-button-hoverBackground); }
`;