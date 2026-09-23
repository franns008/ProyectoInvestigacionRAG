/**
 * Punto de entrada de la extensión.
 *
 * Dos superficies: el comando que escanea un requirements.txt y muestra los hallazgos
 * priorizados, y el chat de la barra lateral, al que se le adjuntan hallazgos desde ese
 * informe (ver chatView.ts). La extensión no sabe cómo se resuelve el escaneo — eso lo decide el `ScanProvider`
 * que arma `buildProvider()` a partir de los ajustes. Las explicaciones del LLM entran
 * como un decorador sobre ese proveedor, no como un proveedor distinto.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import * as vscode from 'vscode';
import { ChatOptions, ChatView } from './chatView';
import { ExplainedProvider } from './scan/explainedProvider';
import { ResultsPanel } from './panel';
import { LocalScannerProvider } from './scan/localProvider';
import { previewPackages } from './scan/manifest';
import { RagProvider } from './scan/ragProvider';
import { ScanError, ScanProvider } from './scan/types';

export function activate(context: vscode.ExtensionContext): void {
    const chat = new ChatView(chatOptions);
    context.subscriptions.push(
        chat,
        // retainContextWhenHidden: el hilo y el input a medio escribir sobreviven a
        // cambiar de vista en la barra lateral, igual que en Copilot.
        vscode.window.registerWebviewViewProvider(ChatView.viewType, chat, {
            webviewOptions: { retainContextWhenHidden: true },
        }),
        vscode.commands.registerCommand('cibersec.newChat', () => chat.newChat()),
        vscode.commands.registerCommand('cibersec.clearChat', () => chat.clear()),
        vscode.workspace.onDidChangeConfiguration((event) => {
            if (event.affectsConfiguration('cibersec')) void chat.loadLimits();
        }),
        vscode.commands.registerCommand(
            'cibersec.scanRequirements',
            (resource?: vscode.Uri) => scanCommand(chat, resource),
        ),
    );

    // Señal visible de que la extensión está cargada. Sin esto, la ventana de prueba se ve
    // idéntica a una normal y no hay forma de distinguir "cargó y espera" de "no cargó".
    const status = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        100,
    );
    status.command = 'cibersec.scanRequirements';
    // Etiqueta corta al lado del ícono; el detalle va al tooltip, igual que el botón
    // de la barra de título del editor.
    status.text = '$(shield) Escanear';
    status.tooltip = 'Escanear dependencias vulnerables en este requirements.txt';
    context.subscriptions.push(status);

    const refresh = () => {
        const file = vscode.window.activeTextEditor?.document.uri.fsPath;
        if (file && IS_MANIFEST.test(path.basename(file))) {
            status.show();
        } else {
            status.hide();
        }
    };
    context.subscriptions.push(
        vscode.window.onDidChangeActiveTextEditor(refresh),
    );
    refresh();
}

export function deactivate(): void {
    // El chat se libera con context.subscriptions; el panel se limpia solo al cerrarse.
}

/** Se leen en cada pregunta: cambiar un ajuste no obliga a recargar la ventana. */
function chatOptions(): ChatOptions {
    const config = vscode.workspace.getConfiguration('cibersec');
    return {
        url: config.get<string>('ragUrl', 'http://localhost:9099'),
        model: config.get<string>('chatModel', 'pipeline_ciberseguridad'),
        apiKey: config.get<string>('ragApiKey', '0p3n-w3bui'),
        timeoutMs: config.get<number>('chatTimeoutSeconds', 240) * 1000,
    };
}

/** Mismo criterio que el `when` de los menús en package.json. */
const IS_MANIFEST = /requirements.*\.txt$/;

async function scanCommand(chat: ChatView, resource?: vscode.Uri): Promise<void> {
    const manifest = resource ?? vscode.window.activeTextEditor?.document.uri;
    if (!manifest || manifest.scheme !== 'file') {
        vscode.window.showWarningMessage(
            'Abrí un requirements.txt (o hacé clic derecho sobre él) para escanearlo.',
        );
        return;
    }

    const workspace = vscode.workspace.getWorkspaceFolder(manifest);
    if (!workspace) {
        vscode.window.showWarningMessage(
            'El archivo tiene que estar dentro de un workspace abierto.',
        );
        return;
    }

    const provider = buildProvider(
        workspace.uri.fsPath,
        path.dirname(manifest.fsPath),
    );
    const panel = ResultsPanel.show(chat);
    panel.loading(
        path.basename(manifest.fsPath),
        await readPreview(manifest.fsPath),
    );

    try {
        const result = await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: 'Escaneando dependencias…',
            },
            () => provider.scan(manifest.fsPath),
        );
        panel.render(result);
    } catch (error) {
        if (error instanceof ScanError) {
            panel.error(error.message, error.detail);
        } else {
            panel.error('Error inesperado.', (error as Error).message);
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
function buildProvider(
    workspaceRoot: string,
    manifestDirectory: string,
): ScanProvider {
    const config = vscode.workspace.getConfiguration('cibersec');

    if (config.get<string>('provider') === 'rag') {
        return new RagProvider({
            url: config.get<string>('ragUrl', 'http://localhost:9099'),
            model: config.get<string>('ragModel', 'pipeline_dependencias'),
        });
    }

    const projectRoot = findProjectRoot(
        workspaceRoot,
        manifestDirectory,
        config.get<string>('scannerRoot', 'src/pipeline'),
        config.get<string>('dataDir', 'data/raw'),
    );
    const local = new LocalScannerProvider({
        pythonPath: pythonFor(
            projectRoot,
            config.get<string>('pythonPath', ''),
        ),
        scannerRoot: resolve(
            projectRoot,
            config.get<string>('scannerRoot', 'src/pipeline'),
        ),
        dataDir: resolve(
            projectRoot,
            config.get<string>('dataDir', 'data/raw'),
        ),
    });

    // El decorador no arriesga nada: si el servidor no está, devuelve el escaneo tal cual.
    if (!config.get<boolean>('explain', true)) {
        return local;
    }
    return new ExplainedProvider(local, {
        url: config.get<string>('ragUrl', 'http://localhost:9099'),
        model: config.get<string>('explainModel', 'pipeline_dependencias'),
        apiKey: config.get<string>('ragApiKey', '0p3n-w3bui'),
        topN: config.get<number>('explainTopN', 10),
        timeoutMs: config.get<number>('explainTimeoutSeconds', 240) * 1000,
    });
}

/**
 * Los nombres que se muestran mientras se escanea. Es decorado: si el archivo no se
 * puede leer acá, el escáner lo va a leer igual y el error sale por su camino normal.
 */
async function readPreview(fsPath: string) {
    try {
        return previewPackages(await fs.promises.readFile(fsPath, 'utf8'));
    } catch {
        return [];
    }
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
        path.join(workspaceRoot, '.venv', 'bin', 'python'),
        path.join(workspaceRoot, '.venv', 'Scripts', 'python.exe'),
        path.join(workspaceRoot, 'venv', 'bin', 'python'),
        path.join(workspaceRoot, 'venv', 'Scripts', 'python.exe'),
    ];
    return (
        candidates.find((candidate) => fs.existsSync(candidate)) ??
        (process.platform === 'win32' ? 'python' : 'python3')
    );
}

function findProjectRoot(
    workspaceRoot: string,
    manifestDirectory: string,
    scannerRoot: string,
    dataDir: string,
): string {
    const candidates = [
        workspaceRoot,
        manifestDirectory,
        path.dirname(workspaceRoot),
    ];
    return (
        candidates.find(
            (candidate) =>
                fs.existsSync(resolve(candidate, scannerRoot)) &&
                fs.existsSync(resolve(candidate, dataDir)),
        ) ?? workspaceRoot
    );
}

/** Los ajustes de rutas se interpretan relativos al workspace, salvo que sean absolutos. */
function resolve(root: string, configured: string): string {
    return path.isAbsolute(configured)
        ? configured
        : path.join(root, configured);
}
