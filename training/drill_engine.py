"""Manages repetition sequence scheduling, timing delays, and target cues."""
from __future__ import annotations

import random
import time
from typing import Optional

from config import Direction, DrillConfig


class DrillEngine:
    """Orchestrates random parameters and tracking state for individual reps."""

    def __init__(self, config: DrillConfig) -> None:
        self.cfg = config
        self.current_repetition: int = 0
        self._target_direction: Optional[Direction] = None
        self._cue_start_time: float = 0.0
        self._wait_start_time: float = 0.0
        self._target_delay: float = 0.0

    def start_new_session(self) -> None:
        self.current_repetition = 0
        self._target_direction = None

    def prepare_next_repetition(self) -> None:
        self.current_repetition += 1
        self._target_direction = random.choice(list(Direction))
        self._wait_start_time = time.time()
        self._target_delay = random.uniform(self.cfg.min_delay_s, self.cfg.max_delay_s)

    def is_session_complete(self) -> bool:
        return self.current_repetition >= self.cfg.repetitions

    def has_delay_expired(self) -> bool:
        return (time.time() - self._wait_start_time) >= self._target_delay

    def trigger_cue(self) -> Direction:
        self._cue_start_time = time.time()
        if self._target_direction is None:
            self._target_direction = random.choice(list(Direction))
        return self._target_direction

    def get_reaction_time(self) -> float:
        return time.time() - self._cue_start_time

    def has_timeout_expired(self) -> bool:
        return self.get_reaction_time() >= self.cfg.reaction_window_s

    @property
    def target_direction(self) -> Direction:
        if self._target_direction is None:
            raise ValueError("No active repetition target direction has been set.")
        return self._target_direction