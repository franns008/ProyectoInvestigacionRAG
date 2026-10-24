// Script del chat de la barra lateral (src/chatView.ts). El estado inicial lo manda el
// host en data-state del <body>; lo demás llega por postMessage.
const vscode = acquireVsCodeApi();
const state = JSON.parse(document.body.dataset.state);
const form = document.getElementById("form");
const input = document.getElementById("input");
const send = document.getElementById("send");
const messages = document.getElementById("messages");
const context = document.getElementById("context");
const meterBox = document.getElementById("meter");
const meterTip = document.getElementById("meter-tip");
const MAX_LINES = 5;
// Lo que manda "Continuar la respuesta": el historial ya trae la parte cortada.
const CONTINUE_TEXT = "Continuá la respuesta anterior desde donde se cortó, sin repetir lo que ya dijiste.";
let meter = state.meter;
let streaming = null;

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

// Medidor de contexto: lo que el chat tiene cargado ahora, según las partes que estima el host.
function formatTokens(n) {
  return n.toLocaleString("es-AR");
}
function renderMeter() {
  const used = meter.parts.reduce((sum, part) => sum + part.tokens, 0);
  const ratio = used / meter.window;
  const percent = Math.round(ratio * 100);
  const level = ratio >= 1 ? "over" : ratio >= 0.9 ? "danger" : ratio >= 0.75 ? "warning" : "ok";
  meterBox.dataset.level = level;
  const circumference = 2 * Math.PI * 6;
  meterBox.querySelector(".meter-arc").setAttribute(
    "stroke-dasharray", (Math.min(ratio, 1) * circumference).toFixed(2) + " " + circumference.toFixed(2),
  );
  meterBox.querySelector(".meter-ring").setAttribute("aria-label", "Contexto usado: " + percent + "%");

  meterTip.replaceChildren();
  const title = document.createElement("strong");
  title.textContent = "Contexto: ≈" + formatTokens(used) + " de " + formatTokens(meter.window) + " tokens (" + percent + "%)";
  const table = document.createElement("table");
  for (const part of meter.parts) {
    const row = table.insertRow();
    row.insertCell().textContent = part.label;
    const tokens = row.insertCell();
    tokens.textContent = formatTokens(part.tokens);
    tokens.className = "num";
  }
  const note = document.createElement("p");
  note.className = "muted";
  note.textContent = level === "over"
    ? "No entra en la ventana: Ollama recorta el principio del prompt, que son las instrucciones. Quitá adjuntos o limpiá la conversación."
    : level === "danger" || level === "warning"
      ? "Queda poco lugar. Limpiar la conversación libera el historial y deja los adjuntos."
      : used
        ? "Estimación: el prompt real lo arma el pipeline con los documentos que recupera."
        : "El chat está vacío: todavía no hay nada en el contexto.";
  meterTip.append(title);
  if (meter.parts.length) meterTip.append(table);
  meterTip.append(note);
  if (!meter.fromPipeline) {
    const source = document.createElement("p");
    source.className = "muted";
    source.textContent = "No se pudieron leer los valves del pipeline: se usan sus valores por defecto.";
    meterTip.append(source);
  }
}
renderMeter();

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
  input.value = "";
  autosize();
  sendText(text);
});
function sendText(text) {
  // Un "Continuar" viejo ya no aplica: la respuesta que cortó dejó de ser la última.
  for (const button of messages.querySelectorAll("button[data-continue]")) button.remove();
  const intro = messages.querySelector(".intro");
  if (intro) intro.remove();
  addMessage("user", (item) => { item.textContent = text; });
  setBusy(true);
  vscode.postMessage({ type: "send", text });
}
document.addEventListener("click", (event) => {
  if (event.target.closest("button[data-continue]")) {
    if (!input.disabled) sendText(CONTINUE_TEXT);
    return;
  }
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
  // Mientras llega, la respuesta es texto plano; al cerrar se reemplaza por el markdown
  // que renderizó el host.
  if (message.type === "answerDelta") {
    if (!streaming) {
      const typing = messages.querySelector(".typing");
      if (typing) typing.remove();
      addMessage("assistant", (item) => { item.classList.add("streaming"); streaming = item; });
    }
    streaming.textContent += message.text;
    messages.scrollTop = messages.scrollHeight;
  }
  if (message.type === "answer") {
    if (streaming) {
      streaming.classList.remove("streaming");
      streaming.innerHTML = message.html;
      streaming = null;
    } else {
      addMessage("assistant", (item) => { item.innerHTML = message.html; });
    }
    setBusy(false);
  }
  if (message.type === "error") {
    // Una respuesta cortada a la mitad no queda en el hilo: el host tampoco la guarda.
    if (streaming) { streaming.remove(); streaming = null; }
    addMessage("error", (item) => { item.textContent = message.text; });
    setBusy(false);
  }
  if (message.type === "context") { context.innerHTML = message.html; }
  if (message.type === "meter") { meter = message.meter; renderMeter(); }
});
function setBusy(busy) {
  input.disabled = busy;
  send.disabled = busy;
  const typing = messages.querySelector(".typing");
  if (busy && !typing) {
    const indicator = document.createElement("div");
    indicator.className = "message typing";
    indicator.setAttribute("aria-label", "El asistente está escribiendo");
    indicator.append(document.createElement("span"), document.createElement("span"), document.createElement("span"));
    messages.append(indicator);
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
setBusy(state.busy);
messages.scrollTop = messages.scrollHeight;
