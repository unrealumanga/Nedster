# skills/self_tune.py
import json, time
from pathlib import Path
from skills.base import NedsterSkill

class JournalScorer:
    """Heuristic quality score for a completed task, range 0..1."""
    def score(self, entry: dict) -> float:
        score = 0.5  # baseline
        if entry.get("tests_passed") is True:
            score += 0.3
        if entry.get("rolled_back") is True:
            score -= 0.4
        # Penalize tasks that needed a correction turn shortly after
        if entry.get("corrected_within_n_turns", 99) <= 2:
            score -= 0.2
        # Reward concise, single-pass completions (no retries)
        if entry.get("retry_count", 0) == 0:
            score += 0.1
        return max(0.0, min(1.0, score))


class CurateFineTuneSet(NedsterSkill):
    name = "curate_finetune_set"
    description = "Score journal entries and export the highest-quality ones as a LoRA instruction set."
    parameters = {
        "type": "object",
        "properties": {
            "journal_path": {"type": "string"},
            "min_score": {"type": "number", "default": 0.75},
            "output_path": {"type": "string", "default": "finetune/dataset.jsonl"},
        },
        "required": ["journal_path"],
    }

    async def run(self, journal_path: str, min_score: float = 0.75,
                  output_path: str = "finetune/dataset.jsonl") -> dict:
        scorer = JournalScorer()
        kept, skipped = 0, 0
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(journal_path) as src, open(output_path, "w") as dst:
            for line in src:
                entry = json.loads(line)
                s = scorer.score(entry)
                if s >= min_score:
                    dst.write(json.dumps({
                        "instruction": entry["user_input"],
                        "input": entry.get("context_summary", ""),
                        "output": entry["agent_response"],
                        "quality_score": s,
                    }) + "\n")
                    kept += 1
                else:
                    skipped += 1

        return {"kept": kept, "skipped": skipped, "dataset_path": output_path}


class RunLoRATraining(NedsterSkill):
    name = "run_lora_finetune"
    description = "Kick off a local LoRA fine-tune of the base model on the curated dataset. Requires explicit confirmation."
    parameters = {
        "type": "object",
        "properties": {
            "dataset_path": {"type": "string"},
            "base_model": {"type": "string", "default": "nedster"},
            "confirmed": {"type": "boolean", "description": "Must be true; never auto-trains."},
        },
        "required": ["dataset_path", "confirmed"],
    }

    async def run(self, dataset_path: str, base_model: str = "nedster", confirmed: bool = False) -> dict:
        if not confirmed:
            return {"status": "blocked", "msg": "Fine-tuning requires explicit user confirmation."}
        # Real training invocation left to your local peft/transformers pipeline;
        # this skill's job is the gate + bookkeeping, not reimplementing peft.
        run_id = f"lora_{int(time.time())}"
        Path(f"finetune/runs/{run_id}.json").parent.mkdir(parents=True, exist_ok=True)
        Path(f"finetune/runs/{run_id}.json").write_text(json.dumps({
            "dataset": dataset_path, "base_model": base_model, "started": time.time()
        }))
        return {"status": "started", "run_id": run_id}
