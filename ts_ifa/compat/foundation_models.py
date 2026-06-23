"""Compatibility imports for foundation model wrappers.

New code should import from ``ts_ifa.models``. This module is kept so older
local notebooks can migrate without changing both imports at once.
"""

from __future__ import annotations

from ..models.chronos_model import Chronos
from ..models.patchtst import PatchTST

__all__ = ["Chronos", "PatchTST"]
