const { ipcRenderer } = require('electron');

async function postToReddit() {
  console.log("Waiting for Reddit page load...");
  
  const checkLogin = setInterval(() => {
    try {
      // 1. The old Reddit design (Old.Reddit)
      const oldTitle = document.querySelector('textarea[name="title"]');
      const oldText = document.querySelector('textarea[name="text"]');
      
      // 2. The standard new Reddit design
      const stdTitle = document.querySelector('textarea[placeholder="Title"]');
      const stdText = document.querySelector('div[role="textbox"]');
      
      // 3. The newest Reddit "Shreddit" web components (Shadow DOM)
      let shredditTitle = null;
      let shredditText = null;
      
      const composers = document.querySelectorAll('shreddit-composer');
      for (const composer of composers) {
        if (composer.shadowRoot) {
          const t = composer.shadowRoot.querySelector('textarea[name="title"]');
          if (t) shredditTitle = t;
          
          // The rich text editor inside the composer is often deep
          const r = composer.shadowRoot.querySelector('div[contenteditable="true"]');
          if (r) shredditText = r;
        }
      }

      const titleInput = shredditTitle || stdTitle || oldTitle;
      const textInput = shredditText || stdText || oldText;

      if (titleInput) {
        clearInterval(checkLogin);
        console.log("Found post fields! Filling form...");
        fillRedditForm(titleInput, textInput);
      }
    } catch (e) {
      console.log("Waiting for fields...", e);
    }
  }, 1000);
}

function fillRedditForm(titleInput, textInput) {
  const title = "I built Nedster: An autonomous, 100% local coding agent optimized for 8GB VRAM (RTX 3060/4060)";
  const text = `I was tired of local coding agents suffering from "Execution Theater"—where they hallucinate that they created a file, or write bash scripts in markdown blocks instead of using their tools properly.\n\nI wanted Devin/Claude Code-level autonomy without paying API credits or sending my proprietary codebase to the cloud. So, I built **Nedster**, a highly autonomous, CLI-based AI software engineer designed to run entirely locally.\n\n**The biggest problem it solves is "Tool Amnesia" and "Execution Theater":**\n\n*   **Anti-Execution-Theater:** Nedster doesn't just trust the LLM. If the model uses the \`write_file\` tool, the python orchestrator physically checks the disk size of the target path. If the file isn't there, it silently intercepts the model's success message, injects a failure warning into the context, and forces the model to try a different approach.\n*   **Emergency Context Reset:** At 85% context usage, it automatically summarizes the session and flushes raw messages to preserve its critical tool instructions. The model never forgets it has filesystem access.\n*   **OpenClaw-Style Precision:** It bypasses brittle XML \`<edit>\` blocks and uses exact-string replacement (\`edit_file\`), rapid file discovery (\`glob_search\`), and regex content searching (\`grep_search\` via \`ripgrep\`).\n\nIt runs on a custom Qwen3.5:9b Modelfile (\`aria-qwen\`) and easily fits in 8GB VRAM. It uses Python, ChromaDB for long-term memory, and Ollama.\n\nI engineered this specifically to run on an average gamer's GPU (tested on an RTX 3060 Ti).\n\nI'd love for you guys to try it out, tear it apart, and let me know what you think of the approach.\n\n**Repo:** https://github.com/unrealumanga/Nedster`;

  if (titleInput) {
    try {
      titleInput.focus();
      titleInput.value = title;
      titleInput.dispatchEvent(new Event('input', { bubbles: true, composed: true }));
      titleInput.dispatchEvent(new Event('change', { bubbles: true, composed: true }));
      
      // Sometimes we need React's internal setter
      let nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
      nativeInputValueSetter.call(titleInput, title);
      titleInput.dispatchEvent(new Event('input', { bubbles: true }));
    } catch(e) {
      console.log("Failed to set title:", e);
    }
    
    if (textInput) {
      try {
        textInput.focus();
        if (textInput.tagName === "TEXTAREA") {
            textInput.value = text;
            let nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
            nativeInputValueSetter.call(textInput, text);
        } else {
            // It's a contenteditable div
            textInput.innerHTML = `<p>${text.replace(/\n/g, '</p><p>')}</p>`;
        }
        textInput.dispatchEvent(new Event('input', { bubbles: true, composed: true }));
        textInput.dispatchEvent(new Event('change', { bubbles: true, composed: true }));
      } catch(e) {
        console.log("Failed to set text:", e);
      }
    }

    console.log("Reddit form filled. Ready to submit.");
    
    // As a fallback, copy it to the clipboard!
    const { clipboard } = require('electron');
    clipboard.writeText(`[TITLE]\n${title}\n\n[BODY]\n${text}`);
    console.log("Fallback: Copied to clipboard just in case!");
  }
}

// Automatically start monitoring when injected
window.onload = postToReddit;
