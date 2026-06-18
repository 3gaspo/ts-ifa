"""Compatibility imports for foundation model wrappers.

New code should import Chronos from ``chronos_model`` and PatchTST from
``patchtst``. This module is kept so older local notebooks keep working.
"""

from __future__ import annotations

try:
    from .chronos_model import Chronos
    from .patchtst import PatchTST
except ImportError:  # pragma: no cover - direct script execution
    from chronos_model import Chronos
    from patchtst import PatchTST

__all__ = ["Chronos", "PatchTST"]
