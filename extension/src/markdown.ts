/**
 * Renderer de markdown mínimo para las respuestas del LLM.
 *
 * Cubre lo que el modelo escribe en la práctica (títulos, listas, negritas, código,
 * tablas, citas) sin sumar dependencias de runtime. La respuesta es texto no confiable:
 * TODO se escapa antes de aplicar el formato, así que no hay forma de que el modelo
 * inyecte HTML. Los links sólo se aceptan con esquema http(s).
 */

import { escapeHtml } from "./styles";

const FENCE = /^\s*```/;
const HEADING = /^\s*(#{1,6})\s+(.*?)\s*#*\s*$/;
const RULE = /^\s*([-*_])(\s*\1){2,}\s*$/;
const QUOTE = /^\s*>\s?/;
const LIST_ITEM = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;
const TABLE_SEPARATOR = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;

export function renderMarkdown(source: string): string {
  return renderBlocks(source.replace(/\r\n?/g, "\n").split("\n"));
}

function renderBlocks(lines: string[]): string {
  const out: string[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) {
      i++;
      continue;
    }

    if (FENCE.test(line)) {
      const code: string[] = [];
      i++;
      while (i < lines.length && !FENCE.test(lines[i])) code.push(lines[i++]);
      i++; // cierre del bloque (o fin del texto si el modelo no lo cerró)
      out.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }

    const heading = line.match(HEADING);
    if (heading) {
      // El h1 del panel es el identificador de la vulnerabilidad: los títulos de la
      // respuesta quedan por debajo.
      const tag = heading[1].length <= 2 ? "h2" : "h3";
      out.push(`<${tag}>${renderInline(heading[2])}</${tag}>`);
      i++;
      continue;
    }

    if (RULE.test(line)) {
      out.push("<hr>");
      i++;
      continue;
    }

    if (line.includes("|") && i + 1 < lines.length && TABLE_SEPARATOR.test(lines[i + 1])) {
      const rows: string[] = [line];
      i += 2;
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) rows.push(lines[i++]);
      out.push(renderTable(rows));
      continue;
    }

    if (QUOTE.test(line)) {
      const quoted: string[] = [];
      while (i < lines.length && QUOTE.test(lines[i])) quoted.push(lines[i++].replace(QUOTE, ""));
      out.push(`<blockquote>${renderBlocks(quoted)}</blockquote>`);
      continue;
    }

    if (LIST_ITEM.test(line)) {
      const block: string[] = [];
      while (i < lines.length) {
        const current = lines[i];
        if (LIST_ITEM.test(current) || (/^\s+\S/.test(current) && block.length > 0)) {
          block.push(current);
          i++;
        } else if (!current.trim() && i + 1 < lines.length && /^\s+\S|^\s*([-*+]|\d+[.)])\s/.test(lines[i + 1])) {
          i++; // línea en blanco entre ítems de la misma lista
        } else {
          break;
        }
      }
      out.push(renderList(block));
      continue;
    }

    const paragraph: string[] = [];
    while (i < lines.length && lines[i].trim() && !startsBlock(lines, i)) paragraph.push(lines[i++].trim());
    out.push(`<p>${paragraph.map(renderInline).join("<br>")}</p>`);
  }

  return out.join("\n");
}

function startsBlock(lines: string[], i: number): boolean {
  const line = lines[i];
  return (
    FENCE.test(line) ||
    HEADING.test(line) ||
    RULE.test(line) ||
    QUOTE.test(line) ||
    LIST_ITEM.test(line) ||
    (line.includes("|") && i + 1 < lines.length && TABLE_SEPARATOR.test(lines[i + 1]))
  );
}

function renderList(block: string[]): string {
  const baseIndent = indentOf(block[0]);
  const ordered = /^\d/.test(block[0].trim());
  const items: { text: string; children: string[] }[] = [];

  for (const line of block) {
    const match = line.match(LIST_ITEM);
    if (match && indentOf(line) <= baseIndent + 1) {
      items.push({ text: match[3], children: [] });
    } else if (items.length > 0) {
      items[items.length - 1].children.push(line.slice(Math.min(indentOf(line), baseIndent + 2)));
    }
  }

  const tag = ordered ? "ol" : "ul";
  const body = items
    .map((item) => {
      const nested = item.children.length ? renderBlocks(dedent(item.children)) : "";
      return `<li>${renderInline(item.text)}${nested}</li>`;
    })
    .join("");
  return `<${tag}>${body}</${tag}>`;
}

function renderTable(rows: string[]): string {
  const cells = (row: string) =>
    row
      .trim()
      .replace(/^\|/, "")
      .replace(/\|$/, "")
      .split("|")
      .map((cell) => renderInline(cell.trim()));
  const [header, ...body] = rows.map(cells);
  return `<div class="table"><table>
    <thead><tr>${header.map((c) => `<th>${c}</th>`).join("")}</tr></thead>
    <tbody>${body.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody>
  </table></div>`;
}

/** Formato en línea. Los spans de código se separan primero para no formatear adentro. */
export function renderInline(text: string): string {
  return text
    .split(/(`[^`]+`)/)
    .map((part, index) => {
      if (index % 2 === 1) return `<code>${escapeHtml(part.slice(1, -1))}</code>`;
      return escapeHtml(part)
        .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2">$1</a>')
        .replace(/\*\*(?=\S)(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/__(?=\S)(.+?)__/g, "<strong>$1</strong>")
        .replace(/(^|[^*\w])\*(?=\S)([^*]+?)\*(?!\*)/g, "$1<em>$2</em>");
    })
    .join("");
}

function indentOf(line: string): number {
  return line.length - line.trimStart().length;
}

function dedent(lines: string[]): string[] {
  const indent = Math.min(...lines.filter((l) => l.trim()).map(indentOf));
  return lines.map((l) => l.slice(Math.min(indent, indentOf(l))));
}
