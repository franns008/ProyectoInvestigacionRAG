import * as vscode from "vscode";
import { BASE_STYLES, nonceValue } from "./styles";

/** Abre en el navegador un link pedido por un webview, sólo si es http(s). */
export async function openHttpLink(url: string | undefined): Promise<void> {
  if (url && /^https?:\/\//i.test(url)) {
    await vscode.env.openExternal(vscode.Uri.parse(url));
  }
}

/** Documento de un webview: CSP sin nada externo, estilos base y un único script con nonce. */
export function webviewPage(styles: string, body: string, script: string): string {
  const nonce = nonceValue();
  const csp = `default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';`;
  return `<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="${csp}">
<style>${BASE_STYLES}${styles}</style>
</head>
<body>
${body}
<script nonce="${nonce}">${script}</script>
</body>
</html>`;
}
