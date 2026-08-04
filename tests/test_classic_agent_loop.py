from core.agent_loop import COMPLETE_MARKER, AgentLoop


def test_classic_loop_chains_dependent_tool_calls():
    calls = []

    def chat(messages, tools):
        tool_results = sum(message["role"] == "tool" for message in messages)
        if tool_results == 0:
            return {
                "content": "",
                "tool_calls": [
                    {"id": "read", "function": {"name": "read_file", "arguments": {"path": "x.py"}}}
                ],
            }
        if tool_results == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "edit",
                        "function": {
                            "name": "edit_file",
                            "arguments": {"path": "x.py", "old": "a", "new": "b"},
                        },
                    }
                ],
            }
        return {"content": f"{COMPLETE_MARKER} Patched x.py."}

    def execute(name, _arguments):
        calls.append(name)
        return "ok"

    result = AgentLoop(chat, execute, tools_schema=[{"type": "function"}]).run("fix x.py")
    assert calls == ["read_file", "edit_file"]
    assert result.outcome == "success"


def test_classic_loop_caps_runaway_and_detects_repeats():
    counter = 0

    def forever(_messages, _tools):
        nonlocal counter
        counter += 1
        return {
            "content": "",
            "tool_calls": [
                {
                    "id": str(counter),
                    "function": {"name": "run_bash", "arguments": {"cmd": f"echo {counter}"}},
                }
            ],
        }

    capped = AgentLoop(forever, lambda *_: "ok", max_iterations=3).run("loop")
    assert capped.outcome == "budget_exhausted"
    assert capped.iterations == 3

    def repeat(_messages, _tools):
        return {
            "content": "",
            "tool_calls": [
                {"id": "same", "function": {"name": "run_bash", "arguments": {"cmd": "git status"}}}
            ],
        }

    stalled = AgentLoop(repeat, lambda *_: "ok", max_iterations=8).run("repeat")
    assert stalled.outcome == "stalled"
    assert stalled.iterations < 8


def test_failed_tool_cannot_be_recorded_as_success():
    def chat(messages, _tools):
        if not any(message["role"] == "tool" for message in messages):
            return {
                "content": "",
                "tool_calls": [
                    {"id": "bad", "function": {"name": "run_bash", "arguments": {"cmd": "false"}}}
                ],
            }
        return {"content": f"{COMPLETE_MARKER} done"}

    result = AgentLoop(chat, lambda *_: "[ERROR] failed").run("do it")
    assert result.outcome == "blocked"
    assert result.steps_for_replay()[0]["ok"] is False
