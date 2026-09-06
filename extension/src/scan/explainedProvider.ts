/**
 * Decorador: el mismo escaneo, más una explicación corta en las primeras tarjetas.
 *
 * El escaneo lo sigue resolviendo el proveedor que envuelve (hoy el escáner local, que
 * corre offline en ~3 s). Este decorador toma **sólo los primeros N hallazgos**, manda sus
 * identificadores al pipeline `pipeline_dependencias` del servidor de Pipelines, y completa
 * `explanation` / `citations`.
 *
 * Por qué decorador y no un proveedor que reemplace al escáner (docs/explicacion_hallazgos.md):
 * si el servidor está caído, **el escaneo sigue funcionando** y sólo faltan los párrafos.
 * Un fallo acá nunca rompe un resultado que ya está bien.
 *
 * Los campos duros (CVSS, EPSS, KEV, versión de arreglo) no se tocan: los sigue armando
 * Python desde los datos estructurados. Del LLM entra únicamente prosa.
 */

import { Finding, ScanProvider, ScanResult } from "./types";

export interface ExplainOptions {
  /** Servidor de Pipelines (API compatible con OpenAI). */
  url: string;
  /** Pipeline que redacta las explicaciones. */
  model: string;
  /** PIPELINES_API_KEY del servidor. */
  apiKey: string;
  /** Cuántas tarjetas se explican. */
  topN: number;
}

/** El LLM tarda ~1 s por tarjeta; con margen para un modelo local lento. */
const TIMEOUT_MS = 90_000;

/** Recorte del texto del advisory que se manda. El servidor vuelve a recortar. */
const MAX_TEXT_CHARS = 4_000;

/** Lo que devuelve el pipeline. */
interface WireExplanation {
  identifier: string;
  explanation: string;
  citations?: string[];
}

export class ExplainedProvider implements ScanProvider {
  readonly label: string;

  constructor(
    private readonly inner: ScanProvider,
    private readonly options: ExplainOptions,
  ) {
    this.label = `${inner.label} + explicaciones`;
  }

  async scan(
    manifestPath: string,
    token?: { isCancellationRequested: boolean },
  ): Promise<ScanResult> {
    const result = await this.inner.scan(manifestPath, token);

    const targets = result.findings.slice(0, Math.max(this.options.topN, 0));
    if (targets.length === 0 || token?.isCancellationRequested) {
      return result;
    }

    try {
      const explanations = await this.request(targets);
      merge(targets, explanations);
    } catch (error) {
      // Degradación deliberada: se pierde la prosa, no el escaneo.
      console.warn("[cibersec] no pude traer las explicaciones:", (error as Error).message);
    }

    return result;
  }

  private async request(findings: Finding[]): Promise<WireExplanation[]> {
    const response = await fetch(`${trimSlash(this.options.url)}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.options.apiKey}`,
      },
      body: JSON.stringify({
        model: this.options.model,
        stream: false,
        messages: [{ role: "user", content: JSON.stringify({ findings: findings.map(payload) }) }],
      }),
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });

    if (!response.ok) {
      throw new Error(`el servidor respondió ${response.status} ${response.statusText}`);
    }

    const body = (await response.json()) as { choices?: { message?: { content?: string } }[] };
    const content = body.choices?.[0]?.message?.content;
    if (!content) {
      throw new Error("la respuesta no trae contenido");
    }

    const parsed = JSON.parse(content) as { explanations?: WireExplanation[]; error?: string };
    if (parsed.error) {
      throw new Error(parsed.error);
    }
    return parsed.explanations ?? [];
  }
}

/** Sólo lo que el pipeline necesita: ids y el texto del advisory. */
function payload(finding: Finding) {
  return {
    package: finding.package,
    installed_version: finding.installed_version,
    cve: finding.cve,
    osv_ids: finding.osv_ids,
    cwe_ids: finding.cwe_ids,
    summary: cut(finding.summary),
    details: cut(finding.details),
  };
}

function cut(text: string | undefined): string {
  return (text ?? "").slice(0, MAX_TEXT_CHARS);
}

/**
 * Cada explicación vuelve a su tarjeta por identificador, no por posición: si el servidor
 * devuelve menos de las que se pidieron (una generación pudo fallar), las demás igual caen
 * donde corresponde.
 */
function merge(findings: Finding[], explanations: WireExplanation[]): void {
  const porId = new Map(explanations.map((e) => [e.identifier, e]));

  for (const finding of findings) {
    const match = porId.get(identifierOf(finding));
    if (match?.explanation) {
      finding.explanation = match.explanation;
      finding.citations = match.citations ?? [];
    }
  }
}

/** Mismo criterio que `Vulnerability.identifier` en Python: el CVE si existe, si no el OSV. */
function identifierOf(finding: Finding): string {
  return finding.cve ?? finding.osv_ids[0] ?? "?";
}

function trimSlash(url: string): string {
  return url.replace(/\/+$/, "");
}
