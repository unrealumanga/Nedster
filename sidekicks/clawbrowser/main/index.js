const { app, BrowserWindow, ipcMain } = require("electron");
const path = require('path');

function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false, // Eased for local agent injection
      preload: path.join(__dirname, '../agent/post_reddit.js')
    },
  });

  win.loadURL("https://www.reddit.com/r/LocalLLaMA/submit?type=text");
  
  // Expose API for external scripts to run JS inside the loaded page
  ipcMain.on("execute-script", (event, script) => {
    win.webContents.executeJavaScript(script)
      .then(result => event.reply("script-result", result))
      .catch(err => event.reply("script-error", err.message));
  });
}

app.whenReady().then(() => {
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
