"""Nedster: a local-first AI coding and automation agent."""

from ironclad.config import NedsterConfig, load_config
from ironclad.orchestrator import NedsterRuntime

__all__ = ["NedsterConfig", "NedsterRuntime", "load_config"]
__version__ = "3.1.0a2"
