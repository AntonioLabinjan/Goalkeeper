"""Heuristic movement detection based on a normalized body coordinate system.

Design summary:

* During calibration we record the trainee's neutral stance: torso center,
  a body-scale estimate (shoulder width vs. torso height, whichever is more
  reliable), baseline shoulder height, and baseline hip height.
* During a repetition, each frame's torso center is compared against the
  baseline center and divided by body scale, giving a scale-invariant
  displacement (dx, dy) that works whether the trainee is close to or far
  from the camera.
* Horizontal displacement selects LEFT/RIGHT. Vertical displacement, arm
  raise (wrist rising well above baseline shoulder height), and crouch
  (hips dropping below baseline) combine to add an UP/DOWN component,
  producing the six supported `Direction` values.
* A direction is only "confirmed" once the same direction has been read on
  several consecutive frames, which filters out normal pose jitter.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np

from config import Direction, MovementThresholds
from vision.pose_tracker import PoseSnapshot

_REQUIRED_LANDMARKS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


@dataclass(frozen=True)
class BodyBaseline:
    """Neutral stance reference computed during calibration."""

    center: np.ndarray  # normalized (x, y) torso center
    scale: float  # normalized body scale, used to make thresholds size-invariant
    shoulder_y: float  # baseline shoulder height, for arm-raise detection
    hip_y: float  # baseline hip height, for crouch detection


class BaselineBuilder:
    """Accumulates pose samples during calibration and builds a BodyBaseline."""

    def __init__(self) -> None:
        self._centers: List[np.ndarray] = []
        self._scales: List[float] = []
        self._shoulder_ys: List[float] = []
        self._hip_ys: List[float] = []

    def add_sample(self, pose: PoseSnapshot) -> bool:
        """Attempts to add a calibration sample. Returns True if usable."""
        if not pose.has(*_REQUIRED_LANDMARKS):
            return False

        left_shoulder = pose.get("left_shoulder")
        right_shoulder = pose.get("right_shoulder")
        left_hip = pose.get("left_hip")
        right_hip = pose.get("right_hip")

        shoulder_mid = _midpoint(left_shoulder, right_shoulder)
        hip_mid = _midpoint(left_hip, right_hip)
        center = _midpoint(shoulder_mid, hip_mid)

        shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))
        torso_height = float(np.linalg.norm(shoulder_mid - hip_mid))
        # Shoulder width is usually the more stable scale reference; torso
        # height is used as a fallback/blend for robustness.
        scale = max(shoulder_width, torso_height * 0.6, 1e-3)

        self._centers.append(center)
        self._scales.append(scale)
        self._shoulder_ys.append(float(shoulder_mid[1]))
        self._hip_ys.append(float(hip_mid[1]))
        return True

    @property
    def sample_count(self) -> int:
        return len(self._centers)

    def center_stability(self) -> float:
        """Std-dev of collected centers, normalized by mean scale.

        Lower is better. Used to detect that the user was not actually
        holding still during calibration.
        """
        if len(self._centers) < 2:
            return float("inf")
        centers = np.stack(self._centers)
        mean_scale = float(np.mean(self._scales))
        std = float(np.mean(np.std(centers, axis=0)))
        return std / max(mean_scale, 1e-3)

    def build(self) -> BodyBaseline:
        if not self._centers:
            raise ValueError("Cannot build baseline: no calibration samples collected.")
        return BodyBaseline(
            center=np.mean(np.stack(self._centers), axis=0),
            scale=float(np.mean(self._scales)),
            shoulder_y=float(np.mean(self._shoulder_ys)),
            hip_y=float(np.mean(self._hip_ys)),
        )


@dataclass(frozen=True)
class MovementReading:
    """A single frame's movement analysis relative to the baseline."""

    direction: Optional[Direction]  # confirmed direction, or None
    dx: float  # normalized horizontal displacement (negative = trainee's left)
    dy: float  # normalized vertical displacement (negative = up)
    magnitude: float  # overall displacement magnitude


class MovementDetector:
    """Classifies movement direction relative to a calibrated baseline."""

    def __init__(
        self,
        baseline: BodyBaseline,
        thresholds: MovementThresholds,
        consistency_frames: int = 3,
    ) -> None:
        self._baseline = baseline
        self._thresholds = thresholds
        self._consistency_frames = max(1, consistency_frames)
        self._recent: Deque[Optional[Direction]] = deque(maxlen=self._consistency_frames)

    def reset(self) -> None:
        """Clears the consistency buffer. Call at the start of each repetition."""
        self._recent.clear()

    def analyze(self, pose: PoseSnapshot) -> MovementReading:
        if not pose.has(*_REQUIRED_LANDMARKS):
            return MovementReading(direction=None, dx=0.0, dy=0.0, magnitude=0.0)

        left_shoulder = pose.get("left_shoulder")
        right_shoulder = pose.get("right_shoulder")
        left_hip = pose.get("left_hip")
        right_hip = pose.get("right_hip")

        shoulder_mid = _midpoint(left_shoulder, right_shoulder)
        hip_mid = _midpoint(left_hip, right_hip)
        center = _midpoint(shoulder_mid, hip_mid)

        scale = max(self._baseline.scale, 1e-3)
        delta = (center - self._baseline.center) / scale
        dx, dy = float(delta[0]), float(delta[1])
        magnitude = float(np.linalg.norm([dx, dy]))

        # Arm raise: either wrist rising well above the baseline shoulder line.
        arm_raised = False
        for wrist_name in ("left_wrist", "right_wrist"):
            wrist = pose.get(wrist_name)
            if wrist is None:
                continue
            raise_amount = (self._baseline.shoulder_y - float(wrist[1])) / scale
            if raise_amount > self._thresholds.arm_raise_threshold:
                arm_raised = True
                break

        # Crouch: hips dropping notably below the baseline hip line.
        hip_drop = (float(hip_mid[1]) - self._baseline.hip_y) / scale
        crouched = hip_drop > self._thresholds.crouch_threshold

        horizontal: Optional[str] = None
        if dx <= -self._thresholds.horizontal_threshold:
            horizontal = "LEFT"
        elif dx >= self._thresholds.horizontal_threshold:
            horizontal = "RIGHT"

        vertical_up = (dy <= -self._thresholds.vertical_threshold) or arm_raised
        vertical_down = (dy >= self._thresholds.vertical_threshold) or crouched

        direction: Optional[Direction] = None
        if horizontal is not None:
            if vertical_up and not vertical_down:
                direction = Direction.UP_LEFT if horizontal == "LEFT" else Direction.UP_RIGHT
            elif vertical_down and not vertical_up:
                direction = Direction.DOWN_LEFT if horizontal == "LEFT" else Direction.DOWN_RIGHT
            else:
                direction = Direction.LEFT if horizontal == "LEFT" else Direction.RIGHT

        self._recent.append(direction)

        confirmed: Optional[Direction] = None
        if len(self._recent) == self._consistency_frames and all(d is not None for d in self._recent):
            if len(set(self._recent)) == 1:
                confirmed = self._recent[-1]

        return MovementReading(direction=confirmed, dx=dx, dy=dy, magnitude=magnitude)