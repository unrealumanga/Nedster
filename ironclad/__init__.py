"""Compatibility package for the Nedster local runtime.

Public names are resolved lazily (PEP 562) so that importing a submodule,
or running ``nedster --help``, does not load the orchestrator and httpx.
"""

from importlib import import_module

_EXPORTS = {
    "IroncladConfig": ".config",
    "NedsterConfig": ".config",
    "load_config": ".config",
    "MythosOrchestrator": ".orchestrator",
    "NedsterRuntime": ".orchestrator",
}

__all__ = sorted(_EXPORTS)
__version__ = "4.0.0a1"


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
