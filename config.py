"""Central configuration and shared types for the Goalkeeper Trainer app.

Keeping the `Direction` enum here (rather than inside `training/`) avoids a
circular import between `vision.movement_detector` (which classifies
movement into a direction) and `training.drill_engine` (which picks target
directions). Both layers depend on this module instead of on each other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Direction(Enum):
    """Goalkeeper reaction directions supported by the MVP."""

    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP_LEFT = "UP_LEFT"
    UP_RIGHT = "UP_RIGHT"
    DOWN_LEFT = "DOWN_LEFT"
    DOWN_RIGHT = "DOWN_RIGHT"


@dataclass
class CameraConfig:
    device_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    # Mirrors the feed so the trainee sees themselves as in a mirror.
    # This also means "LEFT"/"RIGHT" in the app refer to the trainee's own
    # left/right, not the camera's raw left/right.
    flip_horizontal: bool = True


@dataclass
class CalibrationConfig:
    required_samples: int = 30
    # Max allowed normalized std-dev of the torso center across calibration
    # samples. If exceeded, calibration restarts (the user likely moved).
    max_center_std: float = 0.06


@dataclass
class DrillConfig:
    repetitions: int = 10
    min_delay_s: float = 1.0
    max_delay_s: float = 3.0
    reaction_window_s: float = 2.0
    # Number of consecutive frames that must agree on a direction before a
    # reaction is confirmed. Reduces false positives from pose jitter.
    consistency_frames: int = 3


@dataclass
class MovementThresholds:
    """Normalized thresholds (fractions of body scale) for movement detection."""

    horizontal_threshold: float = 0.35
    vertical_threshold: float = 0.30
    arm_raise_threshold: float = 0.25
    crouch_threshold: float = 0.20


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    drill: DrillConfig = field(default_factory=DrillConfig)
    thresholds: MovementThresholds = field(default_factory=MovementThresholds)
    sessions_dir: str = "data/sessions"