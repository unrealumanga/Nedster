import json
import requests
from pathlib import Path
import importlib.util

SKILLS_DIR = Path(__file__).parent / "skills"
TOOL_REGISTRY = {}


def install_skill(skill_url: str):
    """Downloads and installs a skill from a URL."""
    try:
        response = requests.get(skill_url)
        response.raise_for_status()
        skill_data = response.json()

        skill_name = skill_data["name"]
        skill_dir = SKILLS_DIR / skill_name
        skill_dir.mkdir(exist_ok=True)

        # Save the manifest
        with open(skill_dir / "skill.json", "w") as f:
            json.dump(skill_data, f, indent=2)

        # Download files
        print(f"Installing skill '{skill_name}'...")
        for rel_path, url in skill_data["files"].items():
            file_path = skill_dir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)

            file_content = requests.get(url).text
            with open(file_path, "w") as f:
                f.write(file_content)
            print(f"  - Downloaded {rel_path}")

        print(f"Skill '{skill_name}' installed successfully.")

    except Exception as e:
        print(f"Error installing skill from {skill_url}: {e}")


def load_skills():
    """Loads all installed skills and registers their tools."""
    if not SKILLS_DIR.exists():
        return

    for skill_dir in SKILLS_DIR.iterdir():
        if skill_dir.is_dir() and (skill_dir / "skill.json").exists():
            with open(skill_dir / "skill.json") as f:
                skill_data = json.load(f)

            try:
                entry_point_path = skill_dir / skill_data["entry_point"]
                spec = importlib.util.spec_from_file_location(
                    skill_data["name"], entry_point_path
                )
                skill_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(skill_module)

                # A skill module should have a register() function that returns a dict of tools
                if hasattr(skill_module, "register"):
                    tools = skill_module.register()
                    for tool_name, tool_func in tools.items():
                        full_tool_name = f"{skill_data['name']}__{tool_name}"
                        TOOL_REGISTRY[full_tool_name] = tool_func
                        print(f"[Skills] Registered tool: {full_tool_name}")

            except Exception as e:
                print(f"Error loading skill '{skill_data['name']}': {e}")


# Load skills on import
load_skills()
