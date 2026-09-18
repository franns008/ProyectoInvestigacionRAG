/**
 * Sistema visual compartido por los webviews de la extensión (resultados y chat).
 *
 * Reglas, en orden de importancia:
 *
 * 1. **Sólo tokens del tema de VSCode.** Nada de colores fijos: la extensión tiene que
 *    verse nativa en light, dark y high-contrast sin retoques.
 * 2. **Tipografía en `em`, espaciado en `rem`.** El `body` hereda
 *    `--vscode-font-size`, pero `rem` mide contra el root y NO contra el body: usar
 *    `rem` para texto congela los tamaños aunque el usuario agrande la fuente del
 *    editor. Las distancias sí van en `rem`, que es lo que las mantiene estables.
 * 3. **Una escala, no valores sueltos.** Tamaños, espacios y radios salen de las
 *    variables de abajo. Si hace falta un valor nuevo, se agrega a la escala en lugar
 *    de inventarlo en el lugar de uso.
 * 4. **El acento se reserva.** El color de link es para links; el de error, para
 *    peligro real (KEV). Ninguna etiqueta decorativa gasta un acento.
 *
 * El objetivo declarado en docs/escaneo_dependencias.md ("Disciplina de las dos
 * superficies") es que esto se vea aburrido y profesional, como una herramienta que ya
 * existe. Sin versalitas de dashboard ni píldoras de colores compitiendo entre sí.
 *
 * Lo que está acá lo usan las DOS pantallas. Lo específico de cada una vive en su
 * archivo, y sólo puede agregar lo que esta base no cubre: nunca redefinirlo.
 */

export const BASE_STYLES = `
  :root {
    /* Escala tipográfica, relativa a la fuente del editor. */
    --fs-xs: .78em;
    --fs-sm: .88em;
    --fs-md: 1em;
    --fs-lg: 1.15em;
    --fs-xl: 1.35em;

    /* Escala de espaciado. */
    --sp-1: .25rem;
    --sp-2: .5rem;
    --sp-3: .75rem;
    --sp-4: 1rem;
    --sp-5: 1.5rem;
    --sp-6: 2rem;

    --radius: 6px;
    --radius-sm: 3px;

    /* Alias semánticos: el resto del CSS nombra el rol, no el token de VSCode. */
    --border: var(--vscode-panel-border, rgba(128, 128, 128, .35));
    /*
     * Borde de las tarjetas. panel.border es deliberadamente tenue en casi todos los
     * temas —está pensado para separar paneles, no para dibujar recintos— y con 115
     * tarjetas seguidas los límites se pierden. Se deriva del propio color de texto del
     * tema, así que sigue al tema en claro y en oscuro en vez de ser un gris fijo.
     */
    --border-strong: color-mix(in srgb, var(--vscode-foreground) 24%, transparent);
    --surface: var(--vscode-textCodeBlock-background, rgba(128, 128, 128, .08));
    --muted: var(--vscode-descriptionForeground);
    --danger: var(--vscode-editorError-foreground);
    --warning: var(--vscode-editorWarning-foreground);

    /* Ancho de lectura cómodo. Un solo lugar lo define. */
    --measure: 58rem;
  }

  *, *::before, *::after { box-sizing: border-box; }

  /*
   * El webview hereda la fuente y los colores del tema activo de VSCode. Se declaran
   * explícitamente (y no se confía en el default del host) para que el panel se vea
   * igual que el editor de al lado, incluido el fondo, en cualquier tema instalado.
   */
  body {
    margin: 0;
    font-family: var(--vscode-font-family);
    font-size: var(--vscode-font-size);
    font-weight: var(--vscode-font-weight, normal);
    color: var(--vscode-editor-foreground, var(--vscode-foreground));
    background: var(--vscode-editor-background);
    line-height: 1.55;
  }

  /* El ancho lo pone el contenedor, no el body: así el chat no tiene que anularlo. */
  .page {
    max-width: var(--measure);
    padding: var(--sp-5) var(--sp-6) var(--sp-6);
  }

  h1, h2, h3 { font-weight: 600; line-height: 1.3; }
  h1 { font-size: var(--fs-xl); margin: 0 0 var(--sp-1); }
  h2 { font-size: var(--fs-lg); margin: 0 0 var(--sp-1); }
  h3 { font-size: var(--fs-md); margin: 0 0 var(--sp-1); }
  p { margin: var(--sp-2) 0; }

  code { font-family: var(--vscode-editor-font-family); font-size: .92em; }
  a { color: var(--vscode-textLink-foreground); text-decoration: none; }
  a:hover { color: var(--vscode-textLink-activeForeground); text-decoration: underline; }

  .muted { color: var(--muted); }
  /* Cifras que se comparan entre filas: ancho fijo para que las columnas se alineen. */
  .nums { font-variant-numeric: tabular-nums; }

  /* Etiqueta de sección. Sobria a propósito: sin versalitas ni color de acento. */
  .label {
    margin: 0;
    font-size: var(--fs-xs);
    font-weight: 600;
    color: var(--muted);
  }

  /* Foco visible en todo lo interactivo: es lo único con lo que cuenta el teclado. */
  a:focus-visible,
  button:focus-visible,
  summary:focus-visible,
  textarea:focus-visible {
    outline: 1px solid var(--vscode-focusBorder);
    outline-offset: 1px;
    border-radius: var(--radius-sm);
  }

  /* Superficie base. La usan los hallazgos del informe y las respuestas del chat. */
  .card {
    border: 1px solid var(--border-strong);
    border-radius: var(--radius);
    padding: var(--sp-3) var(--sp-4);
  }

  /*
   * Etiqueta de estado (severidad, "explotada"): rellena, no de contorno.
   *
   * Una caja vacía con el borde del color del texto deja el color sólo en las líneas
   * finas, que es lo que la hace ver de neón. Rellenando, el color queda en una
   * superficie y el borde apenas la cierra.
   *
   * Sin borde: la línea de contorno suma una frecuencia más por etiqueta y, con dos
   * etiquetas por tarjeta sobre más de cien tarjetas, es textura que compite con la
   * lectura. El relleno ya delimita.
   *
   * Una sola regla para las tres variantes: el texto lleva el color que la etiqueta
   * significa y el fondo es un tinte de ese mismo color. Así "media" no queda leyéndose
   * distinto de "crítica" y "alta" por usar otro par de tokens.
   */
  .tag {
    display: inline-block;
    font-size: var(--fs-xs);
    font-weight: 600;
    line-height: 1.6;
    padding: 0 .5em;
    border-radius: var(--radius-sm);
    color: var(--muted);
    background: color-mix(in srgb, var(--vscode-foreground) 10%, transparent);
    white-space: nowrap;
  }
  /* saturate() aviva el color del tema sin cambiarle el tono ni salir de sus tokens. */
  .tag.danger {
    color: var(--danger);
    background: color-mix(in srgb, var(--danger) 26%, transparent);
    filter: saturate(1.35);
  }
  .tag.warning {
    color: var(--warning);
    background: color-mix(in srgb, var(--warning) 26%, transparent);
    filter: saturate(1.35);
  }
  /*
   * Sin dato no es un nivel de la escala: es la ausencia de la escala. Va de contorno,
   * que es la única forma de la familia que no afirma un peso, en vez de compartir el
   * relleno neutro con "media" y hacerlas indistinguibles.
   */
  .tag.unknown {
    color: var(--muted);
    background: transparent;
    box-shadow: inset 0 0 0 1px var(--border-strong);
  }

  /* Identificador (CVE, CWE, OSV), enlazable o no. */
  .chip {
    display: inline-block;
    font-family: var(--vscode-editor-font-family);
    font-size: var(--fs-sm);
    line-height: 1.6;
    padding: 0 .4em;
    border: 1px solid transparent;
    border-radius: var(--radius-sm);
    background: var(--surface);
    white-space: nowrap;
  }
  a.chip:hover { border-color: var(--border); text-decoration: none; }

  .button {
    border: 1px solid var(--vscode-button-border, transparent);
    border-radius: var(--radius-sm);
    padding: .3em .8em;
    font: inherit;
    line-height: 1.5;
    color: var(--vscode-button-foreground);
    background: var(--vscode-button-background);
    cursor: pointer;
  }
  .button:hover { background: var(--vscode-button-hoverBackground); }
  .button:disabled { opacity: .5; cursor: default; }
  .button.secondary {
    color: var(--vscode-button-secondaryForeground);
    background: var(--vscode-button-secondaryBackground);
    border-color: transparent;
  }
  .button.secondary:hover { background: var(--vscode-button-secondaryHoverBackground); }
  /* Para acciones que se repiten en cada ítem: presentes, sin competir con el contenido. */
  .button.quiet {
    color: var(--vscode-foreground);
    background: transparent;
    border-color: var(--border-strong);
  }
  .button.quiet:hover { background: var(--vscode-list-hoverBackground); }

  summary { cursor: pointer; }

  pre {
    background: var(--surface);
    padding: var(--sp-3);
    border-radius: var(--radius);
    white-space: pre-wrap;
    overflow-wrap: anywhere;
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
