/**
 * Cuánto de la ventana de contexto del LLM ocupa lo que el chat tiene cargado ahora,
 * estimado del lado de la extensión. Es el estado actual, no una proyección: no cuenta lo
 * que se está escribiendo ni el lugar que se reserva para la próxima respuesta.
 *
 * Es una **estimación**: el prompt lo arma el pipeline y Ollama no expone un tokenizer.
 * Para que el número se parezca al real se imita lo que hace el servidor al armarlo.
 *
 * - Lo que es valve (ventana, tope de respuesta, documentos, historial) se lee del
 *   pipeline en vivo (`PipelineLimits`, vía `GET /<pipeline>/valves`), así que sigue a
 *   los cambios hechos desde OpenWebUI.
 * - Lo que son constantes de módulo del pipeline (el reparto de advisories de
 *   src/pipeline/chat/context.py, el largo de PROMPT_TEMPLATE, el tamaño de chunk) está
 *   copiado acá abajo. Si cambian allá, hay que cambiarlos acá. Pendiente: que el
 *   pipeline los informe (docs/arquitectura.md §4.4).
 *
 * Lógica pura, sin `vscode`: la usa ChatView y se puede probar con node.
 */

/** Caracteres por token. El texto mezcla inglés, español e identificadores: 3.5 sobreestima un poco, que es lo prudente. */
const CHARS_PER_TOKEN = 3.5;

/** Los valves de pipeline_ciberseguridad que deciden el tamaño del prompt (docs/arquitectura.md §4.3). */
export interface PipelineLimits {
  /** Ventana de Ollama: prompt y respuesta la comparten. */
  num_ctx: number;
  /** Tope de largo de la respuesta. */
  max_tokens: number;
  /** Documentos del corpus que llegan al prompt. */
  ranker_top_k: number;
  history_max_messages: number;
  history_max_chars: number;
}

/** Los defaults del código del pipeline: se usan si no se pudieron leer sus valves. */
export const DEFAULT_LIMITS: PipelineLimits = {
  num_ctx: 8192,
  max_tokens: 512,
  ranker_top_k: 4,
  history_max_messages: 6,
  history_max_chars: 600,
};

/** Toma de la respuesta de `/valves` sólo los números válidos; el resto queda en su default. */
export function parseLimits(valves: unknown): PipelineLimits {
  const source = (valves ?? {}) as Record<string, unknown>;
  const limits = { ...DEFAULT_LIMITS };
  for (const key of Object.keys(limits) as (keyof PipelineLimits)[]) {
    const value = source[key];
    if (typeof value === "number" && Number.isFinite(value) && value >= 0) limits[key] = value;
  }
  return limits;
}

/** Tokens por documento del corpus: chunks de 200 palabras, ~280 tokens (docs/data_splitting.md). */
const TOKENS_PER_DOC = 280;
/** Reglas de PROMPT_TEMPLATE más los encabezados de cada sección. */
const TEMPLATE_TOKENS = 450;

// Espejo de format_findings (src/pipeline/chat/context.py).
const MAX_ADVISORY_CHARS = 1200;
const MAX_ADVISORY_CHARS_TOTAL = 3000;
const MIN_ADVISORY_CHARS = 300;
/** Identificador, paquete, scores y CWE de cada hallazgo, sin el advisory. */
const FINDING_FIELDS_CHARS = 220;

export interface MeterPart {
  label: string;
  tokens: number;
}

/** Lo que el webview necesita para dibujar el medidor. */
export interface MeterData {
  window: number;
  parts: MeterPart[];
  /** Si los límites salen de los valves del pipeline o de los defaults (no respondió). */
  fromPipeline: boolean;
}

export interface MeterInput {
  limits: PipelineLimits;
  fromPipeline: boolean;
  /** summary + details de cada hallazgo adjunto. */
  advisories: string[];
  /** La conversación tal como la guarda el chat. */
  history: { content: string }[];
}

export function estimateTokens(text: string): number {
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}

export function meterData({ limits, fromPipeline, advisories, history }: MeterInput): MeterData {
  const parts: MeterPart[] = [];
  // Instrucciones y documentos entran al contexto recién con la primera pregunta.
  if (history.length) {
    parts.push({ label: "Instrucciones del sistema", tokens: TEMPLATE_TOKENS });
    if (limits.ranker_top_k > 0) {
      parts.push({ label: `Documentos del corpus (${limits.ranker_top_k})`, tokens: limits.ranker_top_k * TOKENS_PER_DOC });
    }
  }

  if (advisories.length) {
    const budget =
      advisories.length === 1
        ? MAX_ADVISORY_CHARS
        : Math.max(MIN_ADVISORY_CHARS, Math.floor(MAX_ADVISORY_CHARS_TOTAL / advisories.length));
    const chars = advisories.reduce(
      (sum, advisory) => sum + FINDING_FIELDS_CHARS + Math.min(advisory.trim().length, budget),
      0,
    );
    parts.push({ label: `Hallazgos adjuntos (${advisories.length})`, tokens: Math.ceil(chars / CHARS_PER_TOKEN) });
  }

  // Igual que select_history: los últimos N mensajes, cada uno recortado.
  const turns =
    limits.history_max_messages > 0
      ? history.slice(-limits.history_max_messages).filter((turn) => turn.content.trim())
      : [];
  if (turns.length) {
    const chars = turns.reduce((sum, turn) => sum + Math.min(turn.content.trim().length, limits.history_max_chars), 0);
    const dropped = history.length - turns.length;
    const label = dropped > 0 ? `Historial (últimos ${turns.length} de ${history.length})` : `Historial (${turns.length})`;
    parts.push({ label, tokens: Math.ceil(chars / CHARS_PER_TOKEN) });
  }

  return { window: limits.num_ctx, parts, fromPipeline };
}

/**
 * Si la respuesta parece cortada por el tope de `max_tokens`.
 *
 * El servidor de Pipelines cierra siempre con `finish_reason: "stop"`, así que no hay
 * señal explícita: se infiere. Tiene que estar cerca del tope (una respuesta corta que
 * termina en un ítem de lista sin punto es normal) y no terminar como termina un texto
 * completo.
 */
export function looksTruncated(answer: string, maxTokens: number): boolean {
  if (estimateTokens(answer) < maxTokens * 0.7) return false;
  const text = answer.trimEnd();
  const fences = text.match(/^\s*```/gm)?.length ?? 0;
  if (fences % 2 === 1) return true;
  return !/[.!?:;)\]"'»…*`|]$/.test(text);
}
