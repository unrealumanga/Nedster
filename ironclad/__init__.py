"""Nedster Ironclad: local-first router/worker agent runtime."""

from .config import IroncladConfig, load_config
from .orchestrator import MythosOrchestrator

__all__ = ["IroncladConfig", "MythosOrchestrator", "load_config"]
__version__ = "3.0.0a1"
