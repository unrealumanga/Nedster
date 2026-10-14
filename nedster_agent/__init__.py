"""Nedster: a local-first AI coding and automation agent."""

from importlib import import_module

_EXPORTS = {
    "NedsterConfig": "ironclad.config",
    "load_config": "ironclad.config",
}

__all__ = sorted(_EXPORTS)
__version__ = "4.0.0a1"


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value
