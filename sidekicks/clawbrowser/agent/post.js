const { ipcRenderer } = require('electron');

async function postToHN() {
  // First we need to make sure the user logs in manually
  console.log("Waiting for manual login to Hacker News...");
  
  // Checking login state (does the 'submit' form exist?)
  const checkLogin = setInterval(() => {
    if (document.querySelector('input[name="title"]')) {
      clearInterval(checkLogin);
      console.log("Logged in! Filling form...");
      fillForm();
    }
  }, 2000);
}

function fillForm() {
  const title = "Show HN: Nedster – An open-source, local-first coding agent that verifies its own work";
  const text = `Hey HN,\n\nI built Nedster, a highly autonomous, CLI-based AI software engineer designed to run entirely locally. It’s optimized specifically for consumer hardware (tested extensively on an 8GB RTX 3060 Ti).\n\nThe biggest problem I faced with running local agents (especially mid-size models like Qwen) is what I call "Execution Theater"—the model hallucinates that it created a file.\n\nTo solve this, Nedster uses a heavily customized Ollama Modelfile (aria-qwen) and an aggressive python orchestrator:\n\n1. Anti-Execution-Theater: Physically checks disk sizes after writes.\n2. OpenClaw-Style Precision: Bypasses XML for exact-string replacement.\n3. Emergency Context Reset: Auto-summarizes at 85% context usage to preserve tool instructions.\n\nRepo: https://github.com/unrealumanga/Nedster`;

  const titleInput = document.querySelector('input[name="title"]');
  const textInput = document.querySelector('textarea[name="text"]');

  if (titleInput && textInput) {
    titleInput.value = title;
    textInput.value = text;
    console.log("Form filled. Ready to submit.");
    // document.querySelector('input[type="submit"]').click(); // Commented out to prevent accidental auto-posting without review
  }
}

// Automatically start monitoring when injected
window.onload = postToHN;
