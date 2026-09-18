/**
 * Lectura del manifiesto para la pantalla de espera, y SÓLO para eso.
 *
 * El parseo que vale es el de Python (`deps.resolver.parse_requirements`): es el que
 * decide qué se escanea y qué se descarta, y el que el informe reporta. Acá se hace una
 * lectura mínima y tolerante con un único fin: poder mostrar los nombres reales de las
 * dependencias mientras el escáner trabaja, sin esperar a que termine ni pagar el costo
 * de levantar el intérprete.
 *
 * Si las dos lecturas difieren en un caso raro, manda la de Python. Por eso esto no
 * decide nada: alimenta una animación.
 */

export interface PreviewPackage {
  name: string;
  /** Sólo cuando la línea trae pin exacto. */
  version?: string;
  /** `false` para lo que el resolver va a descartar por no tener `==`. */
  pinned: boolean;
}

/** Máximo de ítems a mostrar: más que esto no entra en pantalla y no aporta. */
const MAX_PREVIEW = 60;

export function previewPackages(text: string): PreviewPackage[] {
  const packages: PreviewPackage[] = [];

  for (const line of text.split(/\r?\n/)) {
    // Fuera comentarios (incluidos los de fin de línea) y opciones de pip.
    const clean = line.replace(/(^|\s)#.*$/, "").trim();
    if (!clean || clean.startsWith("-")) continue;

    const pinned = clean.match(/^([A-Za-z0-9._-]+)\s*(?:\[[^\]]*\])?\s*==\s*([^\s;,#]+)/);
    if (pinned) {
      packages.push({ name: pinned[1], version: pinned[2], pinned: true });
    } else {
      const loose = clean.match(/^([A-Za-z0-9._-]+)/);
      if (loose) packages.push({ name: loose[1], pinned: false });
    }

    if (packages.length >= MAX_PREVIEW) break;
  }

  return packages;
}
