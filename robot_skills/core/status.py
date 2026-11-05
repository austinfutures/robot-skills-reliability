from __future__ import annotations

from enum import Enum


class Status(str, Enum):
    """Tick result shared by skills and behavior-tree nodes."""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"
