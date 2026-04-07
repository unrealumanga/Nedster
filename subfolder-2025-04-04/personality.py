# File: personality.py
import json
import os

DEFAULT_PERSONALITY = {
    "name": "Aria",
    "persona": "A knowledgeable local AI assistant",
    "tone": "Professional and direct",
    "traits": [],
    "language": "English",
    "greeting": "Ready. How can I help?",
    "system_addendum": "",
}


def load_personality(path: str = "personality.json") -> dict:
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            print(f"[personality.json load error: {e}] Using defaults.")
    return DEFAULT_PERSONALITY


def build_system_prompt(personality: dict, base_instructions: str) -> str:
    """Combine personality traits with RAG instructions."""
    traits_text = "\n".join(f"- {t}" for t in personality.get("traits", []))
    addendum = personality.get("system_addendum", "")
    return (
        f"{addendum}\n\n{base_instructions}\n\nBehavioral traits:\n{traits_text}"
    ).strip()
