// Preload scripts can contain node APIs exposed to the renderer.
// By default, contextIsolation is true, so we can expose APIs safely.
window.addEventListener("DOMContentLoaded", () => {
  // Example: expose a simple API to the renderer
  // window.myAPI = { /* ... */ };
});
