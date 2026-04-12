"""
Nedster 3D Workflow Engine
Runs structured 3D pipeline workflows from JSON definitions.

Inspired by getdesign.md's workflow approach but for 3D pipelines:
SketchUp -> D5 Renderer / Unreal Engine 5
"""

import json, subprocess, os
from pathlib import Path
from datetime import datetime

WORKFLOWS_DIR = Path.home() / ".aria" / "workflows"
DEFAULT_WORKFLOWS = {
    "sketchup_to_d5": {
        "name": "SketchUp -> D5 Renderer",
        "description": "Export SketchUp model, import to D5, render",
        "stages": [
            {
                "id": "export_skp",
                "name": "Export from SketchUp",
                "tool": "run_bash",
                "cmd_template": (
                    "SketchUpViewer.exe "
                    "--export-to-3ds \"{input}\" "
                    "\"{output}\""),
                "sket_paths": [
                    "C:\\Program Files\\SketchUp\\SketchUp 2024",
                    "C:\\Program Files\\SketchUp\\SketchUp 2025",
                ],
                "output_format": ".3ds",
                "notes": ("Use File > Export > 3D Model > .3DS "
                            "in SketchUp for best D5 compatibility. "
                            "Or export as .fbx for UE5."),
            },
            {
                "id": "d5_import",
                "name": "Import to D5",
                "tool": "run_bash",
                "notes": ("D5 Renderer: File > Import > "
                            "choose .3ds or .fbx. "
                            "Alternatively use D5 Sync plugin "
                            "in SketchUp for live sync."),
                "d5_paths": [
                    "C:\\Program Files\\D5 Render",
                    "C:\\Program Files (x86)\\D5 Render",
                ],
            },
            {
                "id": "d5_render",
                "name": "Render in D5",
                "tool": "run_bash",
                "notes": ("D5 batch render via CLI: "
                            "D5Render.exe --batch "
                            "--scene \"{scene}\" "
                            "--output \"{output}\""),
            },
        ],
        "ai_prompts": {
            "scene_setup": (
                "You are an architecture visualization expert. "
                "Analyze this SketchUp model and suggest: "
                "1. Camera positions for best architectural impact "
                "2. Lighting setup (sun angle, HDRI) "
                "3. Material suggestions for realistic rendering "
                "4. Any geometry issues to fix before rendering"
            ),
            "render_review": (
                "Review this architectural render and suggest "
                "improvements to: composition, lighting, materials, "
                "atmosphere, and rendering settings."
            ),
        }
    },

    "sketchup_to_ue5": {
        "name": "SketchUp -> Unreal Engine 5",
        "description": "Export SketchUp, import to UE5 via Datasmith",
        "stages": [
            {
                "id": "export_fbx",
                "name": "Export FBX from SketchUp",
                "notes": ("SketchUp: File > Export > 3D Model > .fbx "
                            "Settings: Export Two-Sided Faces ON, "
                            "Swap YZ coordinates ON for UE5"),
            },
            {
                "id": "datasmith_import",
                "name": "Import via Datasmith",
                "notes": ("UE5 Datasmith plugin: "
                            "Import > Datasmith > choose .fbx "
                            "Or use SketchUp Datasmith Exporter plugin "
                            "for direct .udatasmith export"),
            },
            {
                "id": "ue5_setup",
                "name": "UE5 Scene Setup",
                "tool": "run_bash",
                "cmd_template": (
                    "UnrealEditor-Cmd.exe "
                    "\"{project}.uproject\" "
                    "-run=pythonscript "
                    "-script=\"{setup_script}\""),
                "ue5_paths": [
                    "C:\\Program Files\\Epic Games\\UE_5.3",
                    "C:\\Program Files\\Epic Games\\UE_5.4",
                ],
                "notes": ("UE5 Python script for batch setup: "
                            "set lighting, materials, camera paths"),
            },
        ],
        "ai_prompts": {
            "ue5_material_setup": (
                "Generate Unreal Engine 5 Python API script to: "
                "1. Set up realistic architectural materials "
                "2. Configure Lumen global illumination "
                "3. Set HDRI sky lighting "
                "4. Create cinematic camera sequence "
            ),
        }
    },

    "arch_viz_prompt_pack": {
        "name": "Architecture Viz Prompt Pack",
        "description": "AI prompts for architectural visualization",
        "prompts": {
            "d5_scene_description": (
                "Architectural visualization prompt for D5 Renderer: "
                "{building_type} with {style} aesthetic. "
                "Time: {time_of_day}. Weather: {weather}. "
                "Camera: {camera_angle} at {elevation}. "
                "Materials: {material_style}. "
                "Atmosphere: {mood}."
            ),
            "sketchup_optimization": (
                "Review this SketchUp model structure and identify: "
                "1. Unnecessary geometry to purge (Hidden > Purge) "
                "2. Materials to consolidate "
                "3. Component instances to optimize "
                "4. Geometry that will cause render artifacts"
            ),
            "render_composition": (
                "Suggest 5 camera compositions for this architecture: "
                "{description}. "
                "For each: position, focal length, height, angle, "
                "what to emphasize, what to hide."
            ),
        }
    },
    
    "arch_concept": {
        "name": "Architecture Concept Generator",
        "description": "Generate early architectural concepts and briefs",
        "stages": [
            {
                "id": "brief_analysis",
                "name": "Brief Analysis",
                "notes": "Parse project requirements",
            },
            {
                "id": "style_references",
                "name": "Style References",
                "notes": "Suggest visual references",
            },
            {
                "id": "space_program",
                "name": "Space Program",
                "notes": "Generate space planning",
            },
            {
                "id": "material_palette",
                "name": "Material Palette",
                "notes": "Suggest materials",
            },
            {
                "id": "render_brief",
                "name": "Render Brief",
                "notes": "Create render brief for D5/UE5",
            }
        ],
        "ai_prompts": {
            "brief_analysis": (
                "You are a senior architect. Analyze this project brief "
                "and extract: building type, program requirements, "
                "site constraints, budget tier, client personality. "
                "Brief: {brief}"
            ),
            "render_brief": (
                "Create a D5 Renderer scene brief for: {project_name}. "
                "Include: lighting scenario, time of day, weather mood, "
                "camera positions (minimum 3), material highlights, "
                "landscape elements, and suggested HDRI environment."
            ),
        }
    }
}

def get_workflow(name: str) -> dict:
    """Get a workflow by name from registry or file."""
    WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)

    wf_path = WORKFLOWS_DIR / f"{name}.json"
    if wf_path.exists():
        with open(wf_path) as f:
            return json.load(f)

    if name in DEFAULT_WORKFLOWS:
        return DEFAULT_WORKFLOWS[name]

    return None

def list_workflows() -> list:
    WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)
    builtin = list(DEFAULT_WORKFLOWS.keys())
    file_based = [f.stem for f in WORKFLOWS_DIR.glob("*.json")]
    return list(set(builtin + file_based))

def run_workflow_stage(workflow: dict, stage_id: str, params: dict = None) -> str:
    """Execute a single workflow stage."""
    stage = next((s for s in workflow.get("stages", []) if s["id"] == stage_id), None)
    if not stage:
        return f"[Error] Stage '{stage_id}' not found"

    notes = stage.get("notes", "")
    cmd_template = stage.get("cmd_template", "")

    if cmd_template and params:
        cmd = cmd_template.format(**params)
        from tools import run_bash
        return run_bash(cmd=cmd)
    else:
        return (f"Stage: {stage['name']}\\n"
                f"Manual steps: {notes}\\n"
                f"AI prompt available: "
                f"{'Yes' if workflow.get('ai_prompts') else 'No'}")

def workflow_tool(action: str = "list", name: str = "", stage: str = "", params: str = "{}", **kwargs) -> str:
    """3D workflow runner tool."""
    import json as _json
    if action == "list":
        wfs = list_workflows()
        return "Available workflows:\\n" + "\\n".join(f"  {w}" for w in wfs)
    elif action == "get":
        wf = get_workflow(name)
        if not wf: return f"Workflow '{name}' not found"
        stages = [s["id"] for s in wf.get("stages", [])]
        return (f"{wf['name']}: {wf['description']}\\n"
                f"Stages: {', '.join(stages)}")
    elif action == "run":
        wf = get_workflow(name)
        if not wf: return f"Workflow '{name}' not found"
        try:
            params_dict = _json.loads(params) if params else {}
        except Exception:
            params_dict = {}
        return run_workflow_stage(wf, stage, params_dict)
    elif action == "prompt":
        wf = get_workflow(name)
        if not wf: return f"Workflow '{name}' not found"
        prompts = wf.get("ai_prompts", {})
        if stage and stage in prompts:
            return prompts[stage]
        return "Prompts: " + ", ".join(prompts.keys())
    return f"Unknown action: {action}"
