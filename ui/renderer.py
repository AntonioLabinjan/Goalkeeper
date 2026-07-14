"""High-performance HUD and visual telemetry drawing system using OpenCV."""
from __future__ import annotations

import cv2
import numpy as np
import mediapipe as mp

from config import Direction
from training.session import TrainingSession, TrainingState
from vision.pose_tracker import PoseSnapshot

# Root level extraction for drawing utilities
_MP_DRAWING = getattr(mp, "solutions", None).drawing_utils if hasattr(mp, "solutions") else getattr(mp, "drawing_utils", None)
_MP_POSE = getattr(mp, "solutions", None).pose if hasattr(mp, "solutions") else getattr(mp, "pose", None)

# Direct fallback configuration just in case attributes are hidden deeper
if _MP_DRAWING is None or _MP_POSE is None:
    import mediapipe.python.solutions.drawing_utils as _MP_DRAWING
    import mediapipe.python.solutions.pose as _MP_POSE

# Semantic Color palette definitions (BGR)
COLOR_WHITE = (245, 245, 245)
COLOR_BLACK = (20, 20, 20)
COLOR_GRAY = (120, 120, 120)
COLOR_GREEN = (40, 220, 40)
COLOR_RED = (40, 40, 255)
COLOR_AMBER = (30, 160, 240)
COLOR_BLUE = (255, 140, 0)


class Renderer:
    """Renders overlays, vector directions, telemetry feeds, and results dashboards."""

    def __init__(self) -> None:
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render(self, frame: np.ndarray, session: TrainingSession, pose: Optional[PoseSnapshot]) -> None:
        # 1. Overlay Skeleton Context if tracking holds
        if pose is not None and pose.raw_landmarks is not None:
            _MP_DRAWING.draw_landmarks(
                frame,
                pose.raw_landmarks,
                _MP_POSE.POSE_CONNECTIONS,
                _MP_DRAWING.DrawingSpec(color=(220, 100, 100), thickness=2, circle_radius=2),
                _MP_DRAWING.DrawingSpec(color=(100, 220, 100), thickness=2, circle_radius=2),
            )

        # 2. Layout Top Status Banner Bar
        self._draw_header(frame, session)

        # 3. Handle Overlay Injection based on the current state
        st = session.state
        if st == TrainingState.INITIALIZING:
            self._draw_centered_box_text(frame, "INITIALIZING SYSTEM...", "Checking webcam and tracking models", COLOR_AMBER)

        elif st == TrainingState.CALIBRATING:
            # Handled dynamically contextually inside primary orchestration loop
            pass

        elif st == TrainingState.WAITING:
            self._draw_centered_box_text(frame, "HOLD POSITION", "Waiting for server delivery / shot release...", COLOR_GRAY)

        elif st == TrainingState.CUE_ACTIVE:
            self._draw_cue_arrow(frame, session.drill.target_direction)

        elif st == TrainingState.FEEDBACK:
            self._draw_feedback_overlay(frame, session)

        elif st == TrainingState.SESSION_COMPLETE:
            self._draw_summary_dashboard(frame, session)

    def draw_calibration_progress(self, frame: np.ndarray, current: int, total: int, stability: float) -> None:
        h, w, _ = frame.shape
        progress_pct = min(1.0, current / max(1, total))

        # Status window container
        cv2.rectangle(frame, (w // 4, h // 3), (3 * w // 4, 2 * h // 3), COLOR_BLACK, -1)
        cv2.rectangle(frame, (w // 4, h // 3), (3 * w // 4, 2 * h // 3), COLOR_BLUE, 2)

        title = "CALIBRATING BASELINE"
        subtitle = "Stand in center holding your ready stance"
        cv2.putText(frame, title, (w // 4 + 30, h // 3 + 50), self.font, 0.9, COLOR_WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, subtitle, (w // 4 + 30, h // 3 + 90), self.font, 0.5, COLOR_GRAY, 1, cv2.LINE_AA)

        # Progress bar geometry tracks across samples
        bar_x1, bar_y = w // 4 + 30, h // 2 + 10
        bar_x2 = 3 * w // 4 - 30
        bar_w = bar_x2 - bar_x1
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + 20), COLOR_GRAY, -1)
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + int(bar_w * progress_pct), bar_y + 20), COLOR_GREEN, -1)

        metrics = f"Samples: {current}/{total}  |  Jitter: {stability:.2f}"
        cv2.putText(frame, metrics, (bar_x1, bar_y + 50), self.font, 0.5, COLOR_AMBER, 1, cv2.LINE_AA)

    def _draw_header(self, frame: np.ndarray, session: TrainingSession) -> None:
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, 60), COLOR_BLACK, -1)
        cv2.line(frame, (0, 60), (w, 60), COLOR_BLUE, 2)

        state_str = f"STATE: {session.state.name}"
        rep_str = f"REP: {session.drill.current_repetition}/{session.cfg.drill.repetitions}"
        
        summary = session.scoreboard.compute_summary()
        score_str = f"SAVES: {summary.successful_saves}/{summary.total_attempts}"

        cv2.putText(frame, state_str, (20, 40), self.font, 0.7, COLOR_AMBER, 2, cv2.LINE_AA)
        cv2.putText(frame, rep_str, (w // 2 - 80, 40), self.font, 0.7, COLOR_WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, score_str, (w - 220, 40), self.font, 0.7, COLOR_GREEN, 2, cv2.LINE_AA)

    def _draw_centered_box_text(self, frame: np.ndarray, line1: str, line2: str, border_color: tuple) -> None:
        h, w, _ = frame.shape
        cv2.rectangle(frame, (w // 4, h // 2 - 60), (3 * w // 4, h // 2 + 40), COLOR_BLACK, -1)
        cv2.rectangle(frame, (w // 4, h // 2 - 60), (3 * w // 4, h // 2 + 40), border_color, 2)
        cv2.putText(frame, line1, (w // 4 + 30, h // 2 - 15), self.font, 0.9, COLOR_WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, line2, (w // 4 + 30, h // 2 + 15), self.font, 0.5, COLOR_GRAY, 1, cv2.LINE_AA)

    def _draw_cue_arrow(self, frame: np.ndarray, direction: Direction) -> None:
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2 + 40
        length = 120

        vectors = {
            Direction.LEFT: (-length, 0),
            Direction.RIGHT: (length, 0),
            Direction.UP_LEFT: (-int(length * 0.7), -int(length * 0.7)),
            Direction.UP_RIGHT: (int(length * 0.7), -int(length * 0.7)),
            Direction.DOWN_LEFT: (-int(length * 0.7), int(length * 0.7)),
            Direction.DOWN_RIGHT: (int(length * 0.7), int(length * 0.7)),
        }

        dx, dy = vectors[direction]
        target_pt = (cx + dx, cy + dy)

        cv2.arrowedLine(frame, (cx, cy), target_pt, COLOR_GREEN, 15, tipLength=0.35)
        cv2.putText(frame, direction.name, (cx - 100, cy - length - 20), self.font, 1.5, COLOR_GREEN, 4, cv2.LINE_AA)

    def _draw_feedback_overlay(self, frame: np.ndarray, session: TrainingSession) -> None:
        res = session.last_result
        if not res:
            return

        if res.success:
            msg = "SAVE!"
            sub = f"Time: {res.reaction_time_s:.3f}s"
            color = COLOR_GREEN
        else:
            if res.detected_direction is None:
                msg = "TOO SLOW!"
                sub = "No definitive reaction detected inside window"
            else:
                msg = "WRONG DIRECTION!"
                sub = f"Expected {res.target_direction.name}, Read {res.detected_direction.name}"
            color = COLOR_RED

        self._draw_centered_box_text(frame, msg, sub, color)

    def _draw_summary_dashboard(self, frame: np.ndarray, session: TrainingSession) -> None:
        h, w, _ = frame.shape
        summary = session.scoreboard.compute_summary()

        cv2.rectangle(frame, (w // 6, h // 5), (5 * w // 6, 4 * h // 5), COLOR_BLACK, -1)
        cv2.rectangle(frame, (w // 6, h // 5), (5 * w // 6, 4 * h // 5), COLOR_GREEN, 3)

        start_y = h // 5 + 50
        cv2.putText(frame, "SESSION DRILL COMPLETE", (w // 6 + 40, start_y), self.font, 1.1, COLOR_GREEN, 3, cv2.LINE_AA)

        rows = [
            f"Total Shots: {summary.total_attempts}",
            f"Saves Logged: {summary.successful_saves} ({summary.accuracy_percentage:.1f}%)",
            f"Avg Reaction Time: {f'{summary.avg_reaction_time_s:.3f}s' if summary.avg_reaction_time_s else 'N/A'}",
            f"Fastest Breakout: {f'{summary.fastest_reaction_time_s:.3f}s' if summary.fastest_reaction_time_s else 'N/A'}",
            f"Longest Safe Hold: {f'{summary.slowest_save_time_s:.3f}s' if summary.slowest_save_time_s else 'N/A'}",
            f"Best Save Streak: {summary.max_streak}",
        ]

        for i, text in enumerate(rows):
            cv2.putText(frame, text, (w // 6 + 50, start_y + 50 + (i * 40)), self.font, 0.65, COLOR_WHITE, 2, cv2.LINE_AA)

        cv2.putText(
            frame,
            "Press 'R' to Restart Session | 'Q' or 'ESC' to Quit",
            (w // 6 + 40, 4 * h // 5 - 30),
            self.font,
            0.55,
            COLOR_AMBER,
            1,
            cv2.LINE_AA,
        )