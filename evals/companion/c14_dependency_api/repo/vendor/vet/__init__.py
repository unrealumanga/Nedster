"""vet: tiny record checker.  Vendored from the platform team's toolkit (v0.4).

Public names live in vet.core and are re-exported here.
"""

from .core import Checker, Problem, Verdict, expect

__all__ = ["Checker", "Problem", "Verdict", "expect"]
