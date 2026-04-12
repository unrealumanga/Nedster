"""
Ralph supervisor loop for Nedster.
Uses qwen3-4b (2.5GB) to supervise aria-qwen (6.6GB).
Models run SEQUENTIALLY — not simultaneously — to fit 8GB VRAM.

Supervisor runs BETWEEN turns, not during streaming.
Cost: ~200 tokens per check. Worth it.

Ralph signals (from steipete/agent-scripts):
  CONTINUE  → response looks good, proceed normally
  INJECT    → response has issue, inject correction message
  RESTART   → response shows tool amnesia or is completely broken
"""

import ollama, re

SUPERVISOR_MODEL = "qwen3-4b-q4_k_m:latest"

SUPERVISOR_PROMPT = """You are a brief quality checker for an AI agent.
Read the agent's response and output ONE of:
  CONTINUE — response is fine, proceed
  INJECT: <short correction> — response has a minor issue
  RESTART — response shows tool amnesia or is completely broken

Tool amnesia signs: "I cannot create files", "I don't have filesystem
access", "You would need to run this yourself", "as an AI".
Loop signs: exact same sentence repeated 3+ times.
Broken signs: the response is mostly empty or nonsensical.

Response to check:
{response}

Output ONLY: CONTINUE or INJECT: <msg> or RESTART
"""

def ralph_check(response: str, supervisor_model: str = SUPERVISOR_MODEL) -> dict:
    """
    Run supervisor check. Returns:
      {"action": "CONTINUE"|"INJECT"|"RESTART",
       "message": str}  # for INJECT
    """
    if len(response) < 20:
        return {"action": "CONTINUE", "message": ""}
    
    # Fast local check first (no LLM cost)
    AMNESIA_PATTERNS = [
        r"I cannot (?:directly )?create files",
        r"I don'?t have (?:direct )?filesystem",
        r"as an AI(?:,| I)",
        r"You (?:can|would need to) run this",
    ]
    for pat in AMNESIA_PATTERNS:
        if re.search(pat, response, re.I):
            return {
                "action": "INJECT",
                "message": (
                    "[CORRECTION] You DO have filesystem access. "
                    "Use write_file or run_bash. Execute NOW."
                )
            }
    
    # Repetition check (no LLM needed)
    sentences = [s.strip() for s in response.split('.') if len(s.strip()) > 20]
    if len(sentences) >= 3:
        if len(set(sentences[:5])) <= 2:  # mostly duplicates
            return {"action": "RESTART", "message": ""}
    
    # Only if response is long enough to warrant it
    if len(response) < 500:
        return {"action": "CONTINUE", "message": ""}
    
    try:
        r = ollama.generate(
            model=supervisor_model,
            prompt=SUPERVISOR_PROMPT.format(response=response[:800]),
            options={"num_ctx": 1024, "num_predict": 50,
                     "temperature": 0.0, "think": False}
        )
        result = r["response"].strip()
        if result.startswith("INJECT:"):
            return {"action": "INJECT", "message": result[7:].strip()}
        elif result.startswith("RESTART"):
            return {"action": "RESTART", "message": ""}
        else:
            return {"action": "CONTINUE", "message": ""}
    except Exception:
        return {"action": "CONTINUE", "message": ""}
