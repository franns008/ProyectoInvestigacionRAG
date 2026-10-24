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

import { trimSlash } from '../util';
import {
    Finding,
    ScanProvider,
    ScanResult,
    identifierOf,
    packageKey,
    wireFinding,
} from './types';

export interface ExplainOptions {
    /** Servidor de Pipelines (API compatible con OpenAI). */
    url: string;
    /** Pipeline que redacta las explicaciones. */
    model: string;
    /** PIPELINES_API_KEY del servidor. */
    apiKey: string;
    /** Cuántas tarjetas se explican. */
    topN: number;
    /** Cuánto esperar al servidor, en milisegundos (`cibersec.explainTimeoutSeconds`). */
    timeoutMs: number;
}

/**
 * Tope de `cibersec.explainTopN`: lo más que acepta el servidor (MAX_FINDINGS en
 * src/pipeline/explain/payload.py). Si cambia, cambiar los dos.
 */
const MAX_EXPLAIN_TOP_N = 10;

/** Lo que devuelve el pipeline. */
interface WireExplanation {
    identifier: string;
    explanation: string;
    citations?: string[];
}

export class ExplainedProvider implements ScanProvider {
    constructor(
        private readonly inner: ScanProvider,
        private readonly options: ExplainOptions,
    ) {}

    async scan(manifestPath: string): Promise<ScanResult> {
        const result = await this.inner.scan(manifestPath);

        const targets = firstFindingsByPackage(
            result.findings,
            Math.min(Math.max(this.options.topN, 0), MAX_EXPLAIN_TOP_N),
        );
        if (targets.length === 0) {
            return result;
        }

        try {
            const explanations = await this.request(targets);
            merge(targets, explanations);
        } catch (error) {
            // Degradación deliberada: se pierde la prosa, no el escaneo.
            console.warn(
                '[cibersec] no pude traer las explicaciones:',
                describe(error, this.options),
            );
        }

        return result;
    }

    private async request(findings: Finding[]): Promise<WireExplanation[]> {
        const response = await fetch(
            `${trimSlash(this.options.url)}/v1/chat/completions`,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${this.options.apiKey}`,
                },
                body: JSON.stringify({
                    model: this.options.model,
                    stream: false,
                    messages: [
                        {
                            role: 'user',
                            content: JSON.stringify({
                                findings: findings.map(wireFinding),
                            }),
                        },
                    ],
                }),
                signal: AbortSignal.timeout(this.options.timeoutMs),
            },
        );

        if (!response.ok) {
            throw new Error(
                `el servidor respondió ${response.status} ${response.statusText}`,
            );
        }

        const body = (await response.json()) as {
            choices?: { message?: { content?: string } }[];
        };
        const content = body.choices?.[0]?.message?.content;
        if (!content) {
            throw new Error('la respuesta no trae contenido');
        }

        const parsed = JSON.parse(content) as {
            explanations?: WireExplanation[];
            error?: string;
        };
        if (parsed.error) {
            throw new Error(parsed.error);
        }
        return parsed.explanations ?? [];
    }
}

function firstFindingsByPackage(findings: Finding[], limit: number): Finding[] {
    const seen = new Set<string>();
    const targets: Finding[] = [];

    for (const finding of findings) {
        const key = packageKey(finding.package);
        if (seen.has(key)) continue;
        seen.add(key);
        targets.push(finding);
        if (targets.length >= limit) break;
    }

    return targets;
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

/**
 * Un timeout y un servidor caído se ven igual en el `catch` (`TimeoutError` vs
 * `TypeError: fetch failed`) y se arreglan distinto. Vale la pena distinguirlos.
 */
function describe(error: unknown, options: ExplainOptions): string {
    if ((error as Error)?.name === 'TimeoutError') {
        const segundos = Math.round(options.timeoutMs / 1000);
        return (
            `el servidor no respondió en ${segundos} s. Cada explicación es una generación del LLM; ` +
            `subí cibersec.explainTimeoutSeconds o bajá cibersec.explainTopN.`
        );
    }
    return (error as Error).message;
}
