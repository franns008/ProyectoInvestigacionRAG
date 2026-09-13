/**
 * Estilos que comparten los webviews de la extensión (resultados y chat), para que las
 * dos pantallas se vean como la misma herramienta. Sólo usan variables del tema de
 * VSCode: nada de colores fijos.
 */

export const BASE_STYLES = `
  body {
    font-family: var(--vscode-font-family);
    font-size: var(--vscode-font-size);
    color: var(--vscode-foreground);
    padding: 1.5rem 2rem 3rem;
    line-height: 1.5;
    max-width: 60rem;
  }
  h1 { font-size: 1.3rem; font-weight: 600; margin: 0 0 .25rem; }
  h2 { font-size: 1rem; font-weight: 600; margin: 0 0 .4rem; }
  h3 { font-size: .9rem; font-weight: 600; margin: 0 0 .5rem; }
  p { margin: .4rem 0; }
  code { font-family: var(--vscode-editor-font-family); font-size: .9em; }
  a { color: var(--vscode-textLink-foreground); }
  a:hover { color: var(--vscode-textLink-activeForeground); }
  .muted { color: var(--vscode-descriptionForeground); }

  .card {
    border: 1px solid var(--vscode-panel-border);
    border-left: 3px solid var(--vscode-panel-border);
    border-radius: 3px;
    padding: .85rem 1rem;
    margin: .6rem 0;
  }

  .button {
    border: 1px solid var(--vscode-button-border, transparent);
    border-radius: 3px;
    padding: .35rem .65rem;
    font: inherit;
    color: var(--vscode-button-foreground);
    background: var(--vscode-button-background);
    cursor: pointer;
  }
  .button:hover { background: var(--vscode-button-hoverBackground); }
  .button:disabled { opacity: .5; cursor: default; }
  .button.secondary {
    color: var(--vscode-button-secondaryForeground);
    background: var(--vscode-button-secondaryBackground);
  }
  .button.secondary:hover { background: var(--vscode-button-secondaryHoverBackground); }

  pre {
    background: var(--vscode-textCodeBlock-background);
    padding: .75rem; border-radius: 3px; overflow-x: auto; white-space: pre-wrap;
  }
`;

export function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function nonceValue(): string {
  return Math.random().toString(36).slice(2);
}
