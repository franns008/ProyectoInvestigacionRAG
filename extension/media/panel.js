// Script del informe (src/panel.ts). El estado inicial lo manda el host en data-state
// del <body>: un postMessage anterior a la carga del webview se pierde.
const vscode = acquireVsCodeApi();
const state = JSON.parse(document.body.dataset.state);

// Marca los botones de los hallazgos que ya están en el chat.
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
markAttached(state.attached);

// La onda de la pantalla de espera escalona cada paquete por su posición.
document.querySelectorAll(".scan-item").forEach((item, index) => item.style.setProperty("--i", index));

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
