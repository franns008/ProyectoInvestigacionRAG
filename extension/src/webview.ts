import * as vscode from "vscode";
import { escapeHtml } from "./util";

/** Abre en el navegador un link pedido por un webview, sólo si es http(s). */
export async function openHttpLink(url: string | undefined): Promise<void> {
  if (url && /^https?:\/\//i.test(url)) {
    await vscode.env.openExternal(vscode.Uri.parse(url));
  }
}

/** Carpeta de los CSS y JS de los webviews: lo único que pueden cargar (`localResourceRoots`). */
export function mediaRoot(extensionUri: vscode.Uri): vscode.Uri {
  return vscode.Uri.joinPath(extensionUri, "media");
}

/**
 * Documento de un webview: `base.css`, `<name>.css` y `<name>.js` de media/, con una CSP
 * que no deja cargar nada más. `state` le llega al script como JSON en `data-state`.
 */
export function webviewPage(
  webview: vscode.Webview,
  extensionUri: vscode.Uri,
  name: "chat" | "panel",
  body: string,
  state: unknown,
): string {
  const asset = (file: string) => webview.asWebviewUri(vscode.Uri.joinPath(mediaRoot(extensionUri), file));
  const nonce = Math.random().toString(36).slice(2);
  const csp = `default-src 'none'; style-src ${webview.cspSource}; script-src 'nonce-${nonce}';`;
  return `<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="${csp}">
<link rel="stylesheet" href="${asset("base.css")}">
<link rel="stylesheet" href="${asset(`${name}.css`)}">
</head>
<body data-state="${escapeHtml(JSON.stringify(state))}">
${body}
<script nonce="${nonce}" src="${asset(`${name}.js`)}"></script>
</body>
</html>`;
}
