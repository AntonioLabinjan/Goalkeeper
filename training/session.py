"""Structured finite state machine governing the active training loop."""
from __future__ import annotations

import time
from enum import Enum, auto
from typing import Optional

from config import AppConfig, Direction
from training.drill_engine import DrillEngine
from training.scoring import RepetitionResult, Scoreboard


class TrainingState(Enum):
    INITIALIZING = auto()
    CALIBRATING = auto()
    READY = auto()
    WAITING = auto()
    CUE_ACTIVE = auto()
    FEEDBACK = auto()
    SESSION_COMPLETE = auto()


class TrainingSession:
    """State machine coordinator decoupled from the rendering and tracking loops."""

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        self.drill = DrillEngine(config.drill)
        self.scoreboard = Scoreboard()
        self._state = TrainingState.INITIALIZING

        # Repetition feedback storage
        self.last_result: Optional[RepetitionResult] = None
        self.feedback_display_until: float = 0.0
        self._feedback_duration_s: float = 1.5

    @property
    def state(self) -> TrainingState:
        return self._state

    def change_state(self, to_state: TrainingState) -> None:
        self._state = to_state

    def handle_frame(self, pose_found: bool, detected_dir: Optional[Direction]) -> None:
        """Processes time ticks and transitions based on the context of the current state."""
        if self._state == TrainingState.INITIALIZING:
            # External execution loop transitions this after system checks pass.
            return

        elif self._state == TrainingState.CALIBRATING:
            # Calibrated successfully externally via builder constraints.
            return

        elif self._state == TrainingState.READY:
            if self.drill.is_session_complete():
                self.finalize_session()
            else:
                self.drill.prepare_next_repetition()
                self.change_state(TrainingState.WAITING)

        elif self._state == TrainingState.WAITING:
            if self.drill.has_delay_expired():
                self.drill.trigger_cue()
                self.change_state(TrainingState.CUE_ACTIVE)

        elif self._state == TrainingState.CUE_ACTIVE:
            # Evaluate reaction windows or timeouts
            if detected_dir is not None:
                success = detected_dir == self.drill.target_direction
                rt = self.drill.get_reaction_time()
                self._record_rep(detected_dir, rt, success)
            elif self.drill.has_timeout_expired():
                self._record_rep(None, None, False)

        elif self._state == TrainingState.FEEDBACK:
            if time.time() >= self.feedback_display_until:
                self.change_state(TrainingState.READY)

    def _record_rep(self, detected: Optional[Direction], rt: Optional[float], success: bool) -> None:
        self.last_result = self.scoreboard.record_attempt(
            index=self.drill.current_repetition,
            target=self.drill.target_direction,
            detected=detected,
            reaction_time=rt,
            success=success,
        )
        self.feedback_display_until = time.time() + self._feedback_duration_s
        self.change_state(TrainingState.FEEDBACK)

    def finalize_session(self) -> str:
        self.change_state(TrainingState.SESSION_COMPLETE)
        meta = {
            "repetitions_total": self.cfg.drill.repetitions,
            "reaction_window_limit_s": self.cfg.drill.reaction_window_s,
        }
        return self.scoreboard.export_json(self.cfg.sessions_dir, meta)

    def reset_entire_drill(self) -> None:
        self.drill.start_new_session()
        self.scoreboard = Scoreboard()
        self.last_result = None
        self.change_state(TrainingState.READY)