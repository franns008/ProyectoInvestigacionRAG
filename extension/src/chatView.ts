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
import {
  DEFAULT_LIMITS,
  estimateTokens,
  looksTruncated,
  meterData,
  MeterData,
  parseLimits,
  PipelineLimits,
} from "./contextMeter";
import { renderMarkdown } from "./markdown";
import { chip } from "./references";
import { Finding, findingKey, identifierOf, wireFinding } from "./scan/types";
import { escapeHtml as escape, trimSlash } from "./util";
import { mediaRoot, openHttpLink, webviewPage } from "./webview";

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
const MAX_ATTACHMENTS = 6;

export class ChatView implements vscode.WebviewViewProvider, vscode.Disposable {
  static readonly viewType = "cibersec.chat";

  private view: vscode.WebviewView | undefined;
  private history: ChatMessage[] = [];
  private transcript: Entry[] = [];
  private readonly attachments = new Map<string, Finding>();
  private busy = false;
  /** Sube con cada pregunta y con cada limpieza: una respuesta de un turno viejo se descarta. */
  private turn = 0;
  private inFlight: AbortController | undefined;
  /** Valves del pipeline para el medidor; hasta leerlos, los defaults del código. */
  private limits: PipelineLimits = DEFAULT_LIMITS;
  private limitsFromPipeline = false;
  private readonly changed = new vscode.EventEmitter<ReadonlySet<string>>();

  /** Avisa qué hallazgos quedaron adjuntos, para que el informe marque sus botones. */
  readonly onDidChangeAttachments = this.changed.event;

  constructor(
    private readonly options: () => ChatOptions,
    private readonly extensionUri: vscode.Uri,
  ) {}

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    view.webview.options = { enableScripts: true, localResourceRoots: [mediaRoot(this.extensionUri)] };
    view.webview.html = this.html(view.webview);
    view.webview.onDidReceiveMessage(this.onMessage);
    view.onDidDispose(() => {
      if (this.view === view) this.view = undefined;
    });
    void this.loadLimits();
  }

  /**
   * Lee los valves del pipeline (`GET /<pipeline>/valves` del servidor de Pipelines) para
   * que el medidor use la misma ventana y los mismos topes que el prompt real. Se relee
   * al abrir la vista, al cambiar la configuración y después de cada respuesta, así sigue
   * a los cambios de valves en el servidor. Si falla, quedan los últimos valores leídos.
   */
  async loadLimits(): Promise<void> {
    const { url, model, apiKey } = this.options();
    try {
      const response = await fetch(`${trimSlash(url)}/${encodeURIComponent(model)}/valves`, {
        headers: { Authorization: `Bearer ${apiKey}` },
        signal: AbortSignal.timeout(5000),
      });
      if (!response.ok) throw new Error(`${response.status}`);
      this.limits = parseLimits(await response.json());
      this.limitsFromPipeline = true;
    } catch {
      // Servidor caído o pipeline sin valves: el medidor lo aclara en el tooltip.
    }
    this.refreshMeter();
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
    this.attachments.clear();
    this.changed.fire(this.attachedKeys());
    this.clear();
  }

  /**
   * Borra la conversación pero deja los adjuntos: sirve para liberar contexto sin perder
   * el tema. Si hay una respuesta en camino, se cancela.
   */
  clear(): void {
    this.turn++;
    this.inFlight?.abort();
    this.inFlight = undefined;
    this.busy = false;
    this.history = [];
    this.transcript = [];
    if (this.view) this.view.webview.html = this.html(this.view.webview);
  }

  /** Redibuja el medidor: cambió el historial, los adjuntos o la configuración. */
  refreshMeter(): void {
    void this.view?.webview.postMessage({ type: "meter", meter: this.meter() });
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
    this.refreshMeter();
  }

  private meter(): MeterData {
    return meterData({
      limits: this.limits,
      fromPipeline: this.limitsFromPipeline,
      advisories: [...this.attachments.values()].map((finding) =>
        [finding.summary, finding.details].filter(Boolean).join("\n"),
      ),
      history: this.history,
    });
  }

  private readonly onMessage = async (message: { type?: string; text?: string; url?: string; key?: string }) => {
    if (message.type === "openExternal") {
      await openHttpLink(message.url);
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
    const turn = ++this.turn;
    const controller = new AbortController();
    this.inFlight = controller;
    this.history.push({ role: "user", content: text });
    this.transcript.push({ role: "user", html: escape(text) });
    this.busy = true;
    this.refreshMeter();
    let answer = "";
    let failure: Error | undefined;
    let finishReason: string | undefined;
    try {
      finishReason = await this.ask(findings, controller.signal, (delta) => {
        answer += delta;
        if (turn === this.turn) void this.view?.webview.postMessage({ type: "answerDelta", text: delta });
      });
      if (!answer) throw new Error("la respuesta no trae contenido");
    } catch (error) {
      failure = error as Error;
    }
    // Se limpió el chat mientras esperaba: esta respuesta ya no tiene dónde ir.
    if (turn !== this.turn) return;
    this.inFlight = undefined;
    this.busy = false;

    if (failure && !answer) {
      // El turno fallido sale del historial: si quedara, la próxima pregunta llevaría
      // dos mensajes de usuario seguidos y uno sin respuesta.
      this.history.pop();
      const text = `No se pudo consultar el pipeline: ${describe(failure)}`;
      this.transcript.push({ role: "error", html: escape(text) });
      void this.view?.webview.postMessage({ type: "error", text });
      this.refreshMeter();
      return;
    }

    // Una respuesta cortada a la mitad se queda, con el aviso: lo que llegó sirve y la
    // conversación puede seguir desde ahí.
    const maxTokens = this.limits.max_tokens;
    const truncation = failure
      ? `La respuesta quedó incompleta: ${describe(failure)}.`
      : finishReason === "length" || looksTruncated(answer, maxTokens)
        ? `La respuesta parece cortada por el límite de largo (≈${estimateTokens(answer)} de ${maxTokens} tokens).`
        : undefined;
    this.history.push({ role: "assistant", content: answer });
    const answerHtml = (live: boolean) =>
      renderMarkdown(answer) + (truncation ? renderTruncation(truncation, live) : "") + renderCitedIds(answer, findings);
    this.transcript.push({ role: "assistant", html: answerHtml(false) });
    void this.view?.webview.postMessage({ type: "answer", html: answerHtml(true) });
    void this.loadLimits();
  };

  /**
   * Pide la respuesta en streaming (SSE de OpenAI) y va pasando cada trozo a `onDelta`.
   * Devuelve el `finish_reason` si el servidor lo informó; el texto lo va juntando
   * quien llama, así una respuesta que se corta a la mitad no se pierde.
   */
  private async ask(
    findings: Finding[],
    cancel: AbortSignal,
    onDelta: (text: string) => void,
  ): Promise<string | undefined> {
    const { url, model, apiKey, timeoutMs } = this.options();
    // `system` y no `user`: es el tema de la charla, no algo que dijo el usuario.
    const context: ChatMessage[] = findings.length
      ? [{ role: "system", content: JSON.stringify({ findings: findings.map(wireFinding) }) }]
      : [];
    const response = await fetch(`${trimSlash(url)}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({ model, stream: true, messages: [...context, ...this.history] }),
      signal: AbortSignal.any([cancel, AbortSignal.timeout(timeoutMs)]),
    });
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    if (!response.body) throw new Error("la respuesta no permite streaming");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finishReason: string | undefined;
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (!data || data === "[DONE]") continue;
        const chunk = JSON.parse(data) as {
          choices?: {
            delta?: { content?: string };
            message?: { content?: string };
            finish_reason?: string | null;
          }[];
        };
        const choice = chunk.choices?.[0];
        finishReason = choice?.finish_reason ?? finishReason;
        const text = choice?.delta?.content ?? choice?.message?.content ?? "";
        if (text) onDelta(text);
      }
      if (done) break;
    }
    return finishReason;
  }

  private html(webview: vscode.Webview): string {
    const thread = this.transcript.length
      ? this.transcript.map(renderEntry).join("")
      : `<p class="muted intro">Preguntá lo que quieras sobre ciberseguridad. Para hablar de
          hallazgos concretos, adjuntalos desde el informe del escaneo con el clip: van como
          contexto en cada pregunta hasta que los quites.</p>`;
    return webviewPage(webview, this.extensionUri, "chat", `<main>
  <div id="messages" aria-live="polite">${thread}</div>
  <div class="composer">
    <div id="context">${renderAttachments([...this.attachments.values()])}</div>
    <form id="form">
      <textarea id="input" rows="1" placeholder="Preguntá sobre ciberseguridad…" title="Enter envía, Shift+Enter agrega una línea" aria-label="Pregunta" required></textarea>
      <div class="meter" id="meter">
        <button type="button" class="meter-ring" aria-describedby="meter-tip" aria-label="Contexto usado">${ICONO_ANILLO}</button>
        <div class="meter-tip" id="meter-tip" role="tooltip"></div>
      </div>
      <button id="send" class="send" type="submit" title="Enviar (Enter)" aria-label="Enviar">${ICONO_ENVIAR}</button>
    </form>
  </div>
</main>`, { meter: this.meter(), busy: this.busy });
  }
}

/** Las fichas de adjuntos y, desplegable, el JSON exacto que se manda al pipeline. */
function renderAttachments(findings: Finding[]): string {
  // Sin adjuntos no se dibuja nada: #context:empty se oculta y el input queda solo.
  if (findings.length === 0) return "";
  const items = findings
    .map((finding) => {
      const id = identifierOf(finding, "sin identificador");
      const label = escape(`Quitar ${id} del contexto`);
      return `<span class="attachment" title="${escape(finding.summary ?? id)}">
        <span class="attachment-id">${escape(id)}</span>
        <span class="muted">${escape(finding.package)} ${escape(finding.installed_version)}</span>
        <button type="button" class="remove" data-remove="${escape(findingKey(finding))}" title="${label}" aria-label="${label}">×</button>
      </span>`;
    })
    .join("");
  const payload = { findings: findings.map(wireFinding) };
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

/** Aviso bajo una respuesta cortada. `live` agrega el botón de seguir, sólo en el hilo en vivo. */
function renderTruncation(text: string, live: boolean): string {
  const button = live
    ? ` <button type="button" class="link-button" data-continue>Continuar la respuesta</button>`
    : "";
  return `<p class="truncated"><span class="tag warning">cortada</span> ${escape(text)}${button}</p>`;
}

/** Texto legible para un error del fetch: el timeout y la cancelación traen nombres crípticos. */
function describe(error: Error): string {
  if (error.name === "TimeoutError") return "se agotó el tiempo de espera";
  if (error.name === "AbortError") return "se canceló la consulta";
  return error.message;
}

function unique(values: (string | null | undefined)[]): string[] {
  return [...new Set(values.filter((value): value is string => Boolean(value)))];
}

/** Flecha de enviar. */
const ICONO_ENVIAR = `<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false"><path fill="currentColor" d="M1.72 1.05a.5.5 0 0 1 .54-.07l12.5 6.5a.5.5 0 0 1 0 .89l-12.5 6.5a.5.5 0 0 1-.7-.6L3.43 8 1.56 1.73a.5.5 0 0 1 .16-.68ZM4.37 8.5l-1.4 4.63L13.4 8 2.97 2.87 4.37 7.5H9a.5.5 0 0 1 0 1H4.37Z"/></svg>`;

/** Anillo del medidor de contexto; el script ajusta el arco con stroke-dasharray. */
const ICONO_ANILLO = `<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false"><circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" stroke-opacity=".25" stroke-width="2"/><circle class="meter-arc" cx="8" cy="8" r="6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" transform="rotate(-90 8 8)" stroke-dasharray="0 38"/></svg>`;
