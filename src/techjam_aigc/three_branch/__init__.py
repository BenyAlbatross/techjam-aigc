"""Three-branch AIGC detector.

This package is intentionally isolated from :mod:`techjam_aigc.trace_rx_m` so
the historical TRACE-RX-M implementation and its checkpoints remain intact.
"""

from .config import ThreeBranchConfig
from .memory import DualPrototypeMemory
from .model import ThreeBranchDetector, ThreeBranchOutput

__all__ = [
    "DualPrototypeMemory",
    "ThreeBranchConfig",
    "ThreeBranchDetector",
    "ThreeBranchOutput",
]
