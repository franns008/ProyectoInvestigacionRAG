/**
 * El contrato entre la extensión y quien resuelve el escaneo.
 *
 * Es deliberadamente el mismo para el escáner local y para el RAG: hoy lo llena
 * `python -m deps.cli --json` y mañana lo llena `pipeline_dependencias` a través del
 * servidor de Pipelines. Cambiar de uno a otro es cambiar de implementación de
 * `ScanProvider`, no tocar la vista.
 *
 * Regla que viene de docs/escaneo_dependencias.md y que NO hay que romper: los campos
 * duros (versión de arreglo, CVSS, EPSS, KEV, identificadores) los arma Python por
 * concatenación desde los datos estructurados. El LLM sólo escribe `explanation`. Un
 * modelo no puede equivocarse en un score que nunca escribe.
 */

/** Una vulnerabilidad única que afecta a una dependencia fijada. */
export interface Finding {
  package: string;
  installed_version: string;
  /** Null cuando el advisory no declara alias a un CVE; ahí se usa `osv_ids[0]`. */
  cve: string | null;
  osv_ids: string[];
  cvss_vector: string | null;
  /** Null si no se pudo calcular (vector v4, o advisory sin severidad). */
  cvss_score: number | null;
  cvss_version: string | null;
  cwe_ids: string[];
  /** Null si el advisory no publica una versión de arreglo aplicable. */
  fixed_version: string | null;
  summary: string;
  details: string;
  aliases: string[];
  references: string[];
  /** Probabilidad de explotación (EPSS), 0..1. */
  epss: number;
  /** Presente en el catálogo de CISA de vulnerabilidades explotadas. */
  kev: boolean;

  /**
   * Párrafo explicativo en lenguaje natural, con sus citas.
   *
   * Lo agrega el proveedor RAG; el escáner local lo deja sin definir. La vista tiene
   * que renderizar bien en ambos casos — es la única diferencia visible entre los dos
   * proveedores, y por eso es opcional desde el día uno.
   */
  explanation?: string;
  /** Fuentes citadas por la explicación. Sólo las puebla el proveedor RAG. */
  citations?: string[];
}

/** El embudo: cuántos hallazgos sobreviven a cada criterio de filtrado. */
export interface Funnel {
  total: number;
  paquetes: number;
  cvss_alto: number;
  epss_alto: number;
  kev: number;
  sin_cvss: number;
  sin_cve: number;
}

/** Una línea del manifiesto que no se pudo escanear, y por qué. */
export interface Skipped {
  line: number;
  raw: string;
  reason: string;
}

export interface ScanResult {
  /** Ruta del manifiesto analizado. El nombre del campo espeja el que emite la CLI. */
  manifest: string;
  funnel: Funnel;
  skipped: Skipped[];
  findings: Finding[];
}

/**
 * Quien sabe resolver un escaneo.
 *
 * Implementaciones: LocalScannerProvider (hoy) y RagProvider (Fase 3). La vista sólo
 * conoce esta interfaz.
 */
export interface ScanProvider {
  /** Nombre corto para mostrar al usuario y en los errores. */
  readonly label: string;
  /** @param manifestPath ruta absoluta al requirements.txt */
  scan(manifestPath: string, token?: { isCancellationRequested: boolean }): Promise<ScanResult>;
}

/** Error con un mensaje pensado para mostrarle al usuario, no para el log. */
export class ScanError extends Error {
  constructor(message: string, readonly detail?: string) {
    super(message);
    this.name = "ScanError";
  }
}
