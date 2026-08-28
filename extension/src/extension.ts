/**
 * Punto de entrada de la extensión.
 *
 * Un solo comando: escanear un requirements.txt y mostrar los hallazgos priorizados.
 * La extensión no sabe cómo se resuelve el escaneo — eso lo decide el `ScanProvider`
 * que elige `buildProvider()` según el ajuste `cibersec.provider`. Cuando exista la
 * Fase 3, ahí se devuelve un RagProvider y no cambia nada más.
 */

import * as path from "node:path";
import * as vscode from "vscode";
import { ResultsPanel } from "./panel";
import { LocalScannerProvider } from "./scan/localProvider";
import { RagProvider } from "./scan/ragProvider";
import { ScanError, ScanProvider } from "./scan/types";

export function activate(context: vscode.ExtensionContext): void {
  context.subscriptions.push(
    vscode.commands.registerCommand("cibersec.scanRequirements", scanCommand),
  );
}

export function deactivate(): void {
  // Sin recursos de larga vida: el panel se limpia solo al cerrarse.
}

async function scanCommand(resource?: vscode.Uri): Promise<void> {
  const manifest = resource ?? vscode.window.activeTextEditor?.document.uri;
  if (!manifest || manifest.scheme !== "file") {
    vscode.window.showWarningMessage(
      "Abrí un requirements.txt (o hacé clic derecho sobre él) para escanearlo.",
    );
    return;
  }

  const workspace = vscode.workspace.getWorkspaceFolder(manifest);
  if (!workspace) {
    vscode.window.showWarningMessage(
      "El archivo tiene que estar dentro de un workspace abierto.",
    );
    return;
  }

  const provider = buildProvider(workspace.uri.fsPath);
  const panel = ResultsPanel.show();
  panel.loading(path.basename(manifest.fsPath), provider.label);

  try {
    const result = await vscode.window.withProgress(
      { location: vscode.ProgressLocation.Notification, title: "Escaneando dependencias…" },
      () => provider.scan(manifest.fsPath),
    );
    panel.render(result);
  } catch (error) {
    if (error instanceof ScanError) {
      panel.error(error.message, error.detail);
    } else {
      panel.error("Error inesperado.", (error as Error).message);
    }
  }
}

/**
 * Elige quién resuelve el escaneo.
 *
 * Es la costura de la que habla docs/escaneo_dependencias.md: hoy sólo `local` funciona;
 * `rag` está declarado para que la integración de la Fase 3 sea cambiar este `switch`
 * y el cuerpo de RagProvider.scan(), sin tocar la vista ni el contrato.
 */
function buildProvider(workspaceRoot: string): ScanProvider {
  const config = vscode.workspace.getConfiguration("cibersec");

  if (config.get<string>("provider") === "rag") {
    return new RagProvider({
      url: config.get<string>("ragUrl", "http://localhost:9099"),
      model: config.get<string>("ragModel", "pipeline_dependencias"),
    });
  }

  return new LocalScannerProvider({
    pythonPath: config.get<string>("pythonPath", "python3"),
    scannerRoot: resolve(workspaceRoot, config.get<string>("scannerRoot", "src/pipeline")),
    dataDir: resolve(workspaceRoot, config.get<string>("dataDir", "data/raw")),
  });
}

/** Los ajustes de rutas se interpretan relativos al workspace, salvo que sean absolutos. */
function resolve(root: string, configured: string): string {
  return path.isAbsolute(configured) ? configured : path.join(root, configured);
}
