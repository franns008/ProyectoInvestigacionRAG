/**
 * Punto de entrada de la extensión.
 *
 * Un solo comando: escanear un requirements.txt y mostrar los hallazgos priorizados.
 * La extensión no sabe cómo se resuelve el escaneo — eso lo decide el `ScanProvider`
 * que arma `buildProvider()` a partir de los ajustes. Las explicaciones del LLM entran
 * como un decorador sobre ese proveedor, no como un proveedor distinto.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import * as vscode from "vscode";
import { ExplainedProvider } from "./scan/explainedProvider";
import { ResultsPanel } from "./panel";
import { LocalScannerProvider } from "./scan/localProvider";
import { RagProvider } from "./scan/ragProvider";
import { ScanError, ScanProvider } from "./scan/types";

export function activate(context: vscode.ExtensionContext): void {
  context.subscriptions.push(
    vscode.commands.registerCommand("cibersec.scanRequirements", scanCommand),
  );

  // Señal visible de que la extensión está cargada. Sin esto, la ventana de prueba se ve
  // idéntica a una normal y no hay forma de distinguir "cargó y espera" de "no cargó".
  const status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  status.command = "cibersec.scanRequirements";
  status.text = "$(shield) Escanear dependencias";
  status.tooltip = "Buscar dependencias vulnerables en este requirements.txt";
  context.subscriptions.push(status);

  const refresh = () => {
    const file = vscode.window.activeTextEditor?.document.uri.fsPath;
    if (file && IS_MANIFEST.test(path.basename(file))) {
      status.show();
    } else {
      status.hide();
    }
  };
  context.subscriptions.push(vscode.window.onDidChangeActiveTextEditor(refresh));
  refresh();
}

export function deactivate(): void {
  // Sin recursos de larga vida: el panel se limpia solo al cerrarse.
}

/** Mismo criterio que el `when` de los menús en package.json. */
const IS_MANIFEST = /requirements.*\.txt$/;

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
 * Es la costura de la que habla docs/escaneo_dependencias.md. `local` resuelve el escaneo
 * offline; encima va el decorador que le pide las explicaciones al pipeline (ajuste
 * `cibersec.explain`, ver docs/explicacion_hallazgos.md). `rag` —que resolvería TODO el
 * escaneo en el servidor— sigue declarado y sin implementar.
 */
function buildProvider(workspaceRoot: string): ScanProvider {
  const config = vscode.workspace.getConfiguration("cibersec");

  if (config.get<string>("provider") === "rag") {
    return new RagProvider({
      url: config.get<string>("ragUrl", "http://localhost:9099"),
      model: config.get<string>("ragModel", "pipeline_dependencias"),
    });
  }

  const local = new LocalScannerProvider({
    pythonPath: pythonFor(workspaceRoot, config.get<string>("pythonPath", "")),
    scannerRoot: resolve(workspaceRoot, config.get<string>("scannerRoot", "src/pipeline")),
    dataDir: resolve(workspaceRoot, config.get<string>("dataDir", "data/raw")),
  });

  // El decorador no arriesga nada: si el servidor no está, devuelve el escaneo tal cual.
  if (!config.get<boolean>("explain", true)) {
    return local;
  }
  return new ExplainedProvider(local, {
    url: config.get<string>("ragUrl", "http://localhost:9099"),
    model: config.get<string>("explainModel", "pipeline_dependencias"),
    apiKey: config.get<string>("ragApiKey", "0p3n-w3bui"),
    topN: config.get<number>("explainTopN", 3),
    timeoutMs: config.get<number>("explainTimeoutSeconds", 240) * 1000,
  });
}

/**
 * Intérprete a usar: el configurado, o el `.venv` del workspace si existe.
 *
 * El escáner necesita `packaging`, que rara vez está en el Python del sistema. Detectar
 * el entorno del proyecto evita que la extensión falle en el primer intento por un
 * ajuste que el usuario todavía no sabe que tiene que tocar.
 */
function pythonFor(workspaceRoot: string, configured: string): string {
  if (configured.trim()) {
    return resolve(workspaceRoot, configured.trim());
  }
  const candidates = [
    path.join(workspaceRoot, ".venv", "bin", "python"),
    path.join(workspaceRoot, ".venv", "Scripts", "python.exe"),
    path.join(workspaceRoot, "venv", "bin", "python"),
    path.join(workspaceRoot, "venv", "Scripts", "python.exe"),
  ];
  return candidates.find((candidate) => fs.existsSync(candidate)) ?? "python3";
}

/** Los ajustes de rutas se interpretan relativos al workspace, salvo que sean absolutos. */
function resolve(root: string, configured: string): string {
  return path.isAbsolute(configured) ? configured : path.join(root, configured);
}
