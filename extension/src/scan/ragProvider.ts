/**
 * Proveedor RAG: el mismo escaneo, resuelto por el pipeline, con explicación.
 *
 * **Todavía no está implementado**: depende de la Fase 3 de
 * docs/escaneo_dependencias.md, que es la que crea `pipeline_dependencias` y lo publica
 * como un segundo modelo en el servidor de Pipelines.
 *
 * El archivo existe igual porque documenta la costura y la deja medida: cuando la Fase 3
 * esté, lo único que cambia es el cuerpo de `scan()`. La vista, el contrato y el resto
 * de la extensión quedan como están.
 *
 * La forma prevista de la llamada:
 *
 *     POST {ragUrl}/v1/chat/completions
 *     { "model": "pipeline_dependencias",
 *       "messages": [{ "role": "user", "content": "<contenido del requirements.txt>" }] }
 *
 * y la respuesta trae el mismo `ScanResult` de types.ts, con `explanation` y `citations`
 * completos. Los campos duros los sigue armando Python: el modelo sólo redacta.
 */

import { ScanError, ScanProvider, ScanResult } from "./types";

export interface RagOptions {
  url: string;
  model: string;
}

export class RagProvider implements ScanProvider {
  readonly label = "pipeline RAG";

  constructor(private readonly options: RagOptions) {}

  async scan(_manifestPath: string): Promise<ScanResult> {
    throw new ScanError(
      "El proveedor RAG todavía no está implementado.",
      `Requiere la Fase 3 (pipeline_dependencias en ${this.options.url}, modelo ` +
        `"${this.options.model}"). Mientras tanto, poné cibersec.provider en "local".`,
    );
  }
}
