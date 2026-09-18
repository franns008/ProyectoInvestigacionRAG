/**
 * Identificadores de vulnerabilidad → su fuente autoritativa.
 *
 * Lo usan las dos pantallas: el informe, para que cada CVE/CWE de una tarjeta lleve a
 * NVD o MITRE, y el chat, para el contexto cargado y los IDs que cita el modelo. Estaba
 * duplicado en los dos archivos; el `.chip` ya era un primitivo compartido, así que el
 * enlace que lo acompaña también tiene que serlo.
 */

import { escapeHtml as escape } from "./styles";

export function referenceUrl(id: string): string | null {
  const cwe = id.match(/^CWE-(\d+)$/i);
  if (cwe) return `https://cwe.mitre.org/data/definitions/${cwe[1]}.html`;
  if (/^CVE-\d{4}-\d+$/i.test(id)) return `https://nvd.nist.gov/vuln/detail/${id.toUpperCase()}`;
  if (/^[A-Z]+-[\w-]+$/i.test(id)) return `https://osv.dev/vulnerability/${encodeURIComponent(id)}`;
  return null;
}

/** El identificador como ficha, enlazada cuando sabemos a dónde apunta. */
export function chip(id: string): string {
  const url = referenceUrl(id);
  return url
    ? `<a class="chip" href="${escape(url)}" title="${escape(url)}">${escape(id)}</a>`
    : `<span class="chip">${escape(id)}</span>`;
}
