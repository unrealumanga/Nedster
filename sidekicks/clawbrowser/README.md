# 🦞 ClawBrowser (Nedster Sidekick)

ClawBrowser is an Electron-based, headless-optional, scriptable browser designed to give local AI agents (like **Nedster**) the ability to physically interact with web pages, bypass CAPTCHAs, and navigate complex React/Shadow DOM applications that simple API/curl tools cannot handle.

Instead of trying to parse raw HTML, Nedster can write a custom JavaScript payload (`agent/payload.js`), inject it into ClawBrowser, and command the browser to load a specific URL and execute clicks or keystrokes automatically.

## Usage
1. Make sure you have Node.js installed.
2. Install dependencies:
   ```bash
   cd sidekicks/clawbrowser
   npm install
   ```
3. Run the browser:
   ```bash
   npm run dev
   ```

*(See Nedster's global skills for automation examples).*
