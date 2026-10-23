import * as vscode from "vscode";

/** Abre en el navegador un link pedido por un webview, sólo si es http(s). */
export async function openHttpLink(url: string | undefined): Promise<void> {
  if (url && /^https?:\/\//i.test(url)) {
    await vscode.env.openExternal(vscode.Uri.parse(url));
  }
}
