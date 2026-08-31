"""Parallel global-classifier and authentic-memory TRACE-RX components."""

from .config import ParallelHeadConfig, TraceRXParallelConfig
from .model import TraceRXParallel, TraceRXParallelOutput

__all__ = (
    "ParallelHeadConfig",
    "TraceRXParallel",
    "TraceRXParallelConfig",
    "TraceRXParallelOutput",
)
