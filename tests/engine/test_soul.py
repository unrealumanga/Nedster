from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from ironclad.config import resolve_config
from ironclad.engine.loop import Engine
from ironclad.engine.prompt import (
    DEFAULT_SOUL_TEMPLATE,
    INSTRUCTIONS,
    MAX_SOUL_BYTES,
    MAX_SOUL_PREFERENCES,
    SOUL_HEADER,
    SOUL_PREFERENCES_HEADER,
    ProjectInstructions,
    SoulText,
    build_system_prompt,
    load_soul,
    session_soul,
    with_preference,
)

from .fakes import FakeRuntime, MemoryStore, ScriptedTransport, Turn, ok


def runtime() -> FakeRuntime:
    return FakeRuntime({"read_file": lambda c: ok(c, "contents")})


@pytest.fixture(autouse=True)
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in list(os.environ):
        if name.startswith("NEDSTER_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("NEDSTER_CONFIG_DIR", str(tmp_path / "cfg"))
    return tmp_path / "cfg"


def _instructions() -> ProjectInstructions:
    return ProjectInstructions(name="AGENTS.md", text="Use tabs.", sha256="x", size=9)


# --- template and loader -----------------------------------------------------------------------


def test_default_template_is_terse_and_has_a_preferences_section():
    assert len(DEFAULT_SOUL_TEMPLATE.encode("utf-8")) <= 600
    assert SOUL_PREFERENCES_HEADER in DEFAULT_SOUL_TEMPLATE


def test_load_soul_truncates_with_visible_marker(tmp_path: Path):
    path = tmp_path / "soul.md"
    body = b"x" * (MAX_SOUL_BYTES + 50)
    path.write_bytes(body)
    soul = load_soul(path, mode="full")
    assert soul.truncated and soul.text.endswith("[truncated]")
    assert soul.size == len(body)
    assert soul.sha256 == hashlib.sha256(body).hexdigest()


def test_non_full_mode_carries_no_text(tmp_path: Path):
    path = tmp_path / "soul.md"
    path.write_text("be nice", encoding="utf-8")
    assert load_soul(path, mode="off").text == ""
    assert load_soul(path, mode="ui").present is False


# --- prompt ------------------------------------------------------------------------------------


def test_off_mode_prompt_is_byte_identical_to_today():
    facts = {"os": "Linux"}
    baseline = build_system_prompt(facts, _instructions())
    assert build_system_prompt(facts, _instructions(), SoulText()) == baseline
    assert build_system_prompt(facts, _instructions(), None) == baseline


def test_full_mode_prompt_appends_soul_last():
    soul = SoulText(mode="full", text="Be terse.", sha256="x", size=9)
    prompt = build_system_prompt({"os": "Linux"}, _instructions(), soul)
    assert prompt.startswith(INSTRUCTIONS)
    assert prompt.rstrip().endswith(f"{SOUL_HEADER}\nBe terse.")


# --- per-session snapshot ----------------------------------------------------------------------


def test_full_mode_writes_the_template_on_first_use(config_dir: Path):
    store = MemoryStore()
    sid = store.create("proj")
    soul = session_soul(store, sid, "full")
    path = config_dir / "soul.md"
    assert path.is_file() and path.read_text(encoding="utf-8") == DEFAULT_SOUL_TEMPLATE
    assert soul.mode == "full" and soul.text == DEFAULT_SOUL_TEMPLATE.rstrip("\n")


async def test_snapshot_is_stable_across_resume(config_dir: Path, tmp_path: Path):
    store = MemoryStore()
    sid = store.create(str(tmp_path))
    engine = Engine(
        ScriptedTransport([Turn("ok")]), runtime(), model="m", store=store, project_root=tmp_path, soul="full"
    )
    await engine.run(sid, "hello")
    first = engine.messages(sid)[0].content
    assert SOUL_HEADER in first

    (config_dir / "soul.md").write_text("# Voice\n\n- edited line\n", encoding="utf-8")
    resumed = Engine(
        ScriptedTransport([Turn("again")]), runtime(), model="m", store=store, project_root=tmp_path, soul="full"
    )
    await resumed.run(sid, "again")
    prefix = resumed.messages(sid)[0].content
    assert prefix == first and "edited line" not in prefix
    assert store.get_state(sid, "soul")["mode"] == "full"


async def test_ui_mode_leaves_the_prompt_unchanged_and_never_reaches_the_answer(config_dir: Path, tmp_path: Path):
    from nedster_agent import personality

    store = MemoryStore()
    sid = store.create(str(tmp_path))
    engine = Engine(
        ScriptedTransport([Turn("hello")]), runtime(), model="m", store=store, project_root=tmp_path, soul="ui"
    )
    result = await engine.run(sid, "hi")
    assert SOUL_HEADER not in engine.messages(sid)[0].content
    for moment in ("thinking", "done_ok", "done_unvalidated", "done_failed", "idle"):
        for line in personality.lines(moment):
            assert line not in result.answer


# --- preferences -------------------------------------------------------------------------------


def test_with_preference_adds_a_bullet_and_dedupes():
    edit = with_preference(DEFAULT_SOUL_TEMPLATE, "Use metric units")
    assert edit.ok and edit.bullet == "- Use metric units" and "- Use metric units" in edit.text
    again = with_preference(edit.text, "use metric units")
    assert not again.ok and again.action == "duplicate"


def test_with_preference_keeps_at_most_five_and_drops_the_oldest():
    text = DEFAULT_SOUL_TEMPLATE
    for index in range(MAX_SOUL_PREFERENCES):
        edit = with_preference(text, f"pref {index}")
        assert edit.ok
        text = edit.text
    sixth = with_preference(text, "pref new")
    assert sixth.ok and sixth.dropped == "- pref 0"
    assert "- pref 0" not in sixth.text and "- pref new" in sixth.text
    assert sixth.text.count("- pref") == MAX_SOUL_PREFERENCES


def test_with_preference_rejects_empty_text():
    assert with_preference(DEFAULT_SOUL_TEMPLATE, "   ").action == "invalid"


# --- config validation -------------------------------------------------------------------------


def test_soul_defaults_to_off(tmp_path: Path):
    config, provenance = resolve_config({}, tmp_path)
    assert config.runtime.soul == "off"
    assert provenance["runtime.soul"] == "default"


def test_soul_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("NEDSTER_RUNTIME__SOUL", "ui")
    config, provenance = resolve_config({}, tmp_path)
    assert config.runtime.soul == "ui" and provenance["runtime.soul"] == "env"


def test_soul_bad_value_warns_and_falls_back_to_off(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("NEDSTER_RUNTIME__SOUL", "snarky")
    config, _ = resolve_config({}, tmp_path)
    assert config.runtime.soul == "off"
    assert any("runtime.soul" in warning for warning in config.warnings)


def test_project_config_cannot_set_soul(tmp_path: Path):
    project = tmp_path / "proj"
    (project / ".nedster").mkdir(parents=True)
    (project / ".nedster" / "config.toml").write_text('[runtime]\nsoul = "full"\n', encoding="utf-8")
    config, _ = resolve_config({}, project)
    assert config.runtime.soul == "off"
    assert any("runtime.soul" in warning for warning in config.warnings)
