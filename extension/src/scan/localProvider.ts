/**
 * Proveedor local: corre el escáner determinístico y parsea su JSON.
 *
 * No necesita Docker, ni pgvector, ni LLM, ni red. Es el camino de la demo y también
 * el fallback razonable cuando el stack no está levantado.
 */

import { spawn } from "node:child_process";
import * as path from "node:path";
import { ScanError, ScanProvider, ScanResult } from "./types";

export interface LocalScannerOptions {
  /** Intérprete de Python (ruta absoluta o comando en el PATH). */
  pythonPath: string;
  /** Directorio desde el que se importa `deps` (habitualmente <workspace>/src/pipeline). */
  scannerRoot: string;
  /** Directorio con los dumps de OSV, EPSS y KEV. */
  dataDir: string;
}

/** Un escaneo grande no debería pasar de unos pocos segundos; más que esto es un cuelgue. */
const TIMEOUT_MS = 60_000;

export class LocalScannerProvider implements ScanProvider {
  readonly label = "escáner local";

  constructor(private readonly options: LocalScannerOptions) {}

  async scan(manifestPath: string): Promise<ScanResult> {
    const { pythonPath, scannerRoot, dataDir } = this.options;
    // El escáner corre con cwd en scannerRoot, así que una ruta relativa se resolvería
    // contra ese directorio y no contra el del usuario.
    const manifest = path.resolve(manifestPath);
    const args = ["-m", "deps.cli", manifest, "--data", dataDir, "--json"];

    const { stdout, stderr, code } = await run(pythonPath, args, scannerRoot);

    if (code !== 0) {
      throw new ScanError(
        `El escáner terminó con código ${code}.`,
        stderr.trim() || "sin salida de error",
      );
    }

    // La CLI escribe sólo el JSON en stdout; los avisos van a stderr. Aun así se busca
    // el primer '{' por si el intérprete imprime algún warning antes de tiempo.
    const start = stdout.indexOf("{");
    if (start === -1) {
      throw new ScanError("El escáner no devolvió JSON.", stdout.trim() || stderr.trim());
    }

    try {
      return JSON.parse(stdout.slice(start)) as ScanResult;
    } catch (error) {
      throw new ScanError(
        "No pude interpretar la salida del escáner.",
        `${(error as Error).message}\n\n${stdout.slice(0, 800)}`,
      );
    }
  }
}

function run(
  command: string,
  args: string[],
  cwd: string,
): Promise<{ stdout: string; stderr: string; code: number }> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      // `deps` se importa desde scannerRoot; PYTHONPATH lo hace explícito para que no
      // dependa de que el cwd esté en sys.path.
      env: { ...process.env, PYTHONPATH: prependPath(cwd, process.env.PYTHONPATH) },
    });

    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill();
      reject(new ScanError(`El escáner no respondió en ${TIMEOUT_MS / 1000} s.`));
    }, TIMEOUT_MS);

    child.stdout.on("data", (chunk) => (stdout += chunk));
    child.stderr.on("data", (chunk) => (stderr += chunk));

    child.on("error", (error) => {
      clearTimeout(timer);
      const detail = (error as NodeJS.ErrnoException).code === "ENOENT"
        ? `No encontré el intérprete "${command}". Revisá el ajuste cibersec.pythonPath.`
        : error.message;
      reject(new ScanError("No pude ejecutar el escáner.", detail));
    });

    child.on("close", (code) => {
      clearTimeout(timer);
      resolve({ stdout, stderr, code: code ?? -1 });
    });
  });
}

function prependPath(entry: string, existing: string | undefined): string {
  return existing ? `${entry}${path.delimiter}${existing}` : entry;
}
