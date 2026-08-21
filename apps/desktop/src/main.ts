import { app, BrowserWindow, shell, Tray, Menu, Notification } from "electron";

let mainWindow: BrowserWindow | undefined;
let tray: Tray | undefined;
export function createWindow(url: string): BrowserWindow {
  const window = new BrowserWindow({
    width: 1440, height: 900, show: false,
    webPreferences: { sandbox: true, contextIsolation: true, nodeIntegration: false },
  });
  window.webContents.setWindowOpenHandler(({ url: external }) => { void shell.openExternal(external); return { action: "deny" }; });
  window.loadURL(url).catch(() => undefined);
  window.once("ready-to-show", () => window.show());
  window.on("close", (event) => { if (!(app as typeof app & { isQuiting?: boolean }).isQuiting) { event.preventDefault(); window.hide(); } });
  return window;
}
app.whenReady().then(() => {
  mainWindow = createWindow(process.env.JUSTIN_WEB_URL ?? "http://127.0.0.1:5173");
  tray = new Tray(process.env.JUSTIN_TRAY_ICON ?? "");
  tray.setContextMenu(Menu.buildFromTemplate([{ label: "Open Justin Agent", click: () => mainWindow?.show() }, { label: "Quit", click: () => { (app as typeof app & { isQuiting?: boolean }).isQuiting = true; app.quit(); } }]));
  new Notification({ title: "Justin Agent", body: "Desktop shell is ready" }).show();
});
