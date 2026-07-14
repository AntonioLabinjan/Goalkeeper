"""MediaPipe-based pose tracking with a simplified landmark interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

# Import only the top-level module to bypass broken solution namespaces
import mediapipe as mp

# Access the tracking engine directly from the root module mapping
try:
    _PoseEngine = mp.Pose
    _POSE_LANDMARK = mp.PoseLandmark
except AttributeError:
    # Universal backup configuration mapping
    _PoseEngine = getattr(mp, "Pose", None) or mp.solutions.pose.Pose
    _POSE_LANDMARK = getattr(mp, "PoseLandmark", None) or mp.solutions.pose.PoseLandmark

# Landmarks we care about, named for readability elsewhere in the codebase.
LANDMARK_NAMES: Dict[str, int] = {
    "nose": _POSE_LANDMARK.NOSE.value,
    "left_shoulder": _POSE_LANDMARK.LEFT_SHOULDER.value,
    "right_shoulder": _POSE_LANDMARK.RIGHT_SHOULDER.value,
    "left_hip": _POSE_LANDMARK.LEFT_HIP.value,
    "right_hip": _POSE_LANDMARK.RIGHT_HIP.value,
    "left_wrist": _POSE_LANDMARK.LEFT_WRIST.value,
    "right_wrist": _POSE_LANDMARK.RIGHT_WRIST.value,
    "left_knee": _POSE_LANDMARK.LEFT_KNEE.value,
    "right_knee": _POSE_LANDMARK.RIGHT_KNEE.value,
    "left_ankle": _POSE_LANDMARK.LEFT_ANKLE.value,
    "right_ankle": _POSE_LANDMARK.RIGHT_ANKLE.value,
}

MIN_VISIBILITY = 0.5


@dataclass(frozen=True)
class PoseSnapshot:
    """A simplified set of body landmarks in normalized image coordinates."""

    points: Dict[str, np.ndarray]
    raw_landmarks: object  # original MediaPipe landmark list, for drawing

    def get(self, name: str) -> Optional[np.ndarray]:
        return self.points.get(name)

    def has(self, *names: str) -> bool:
        return all(name in self.points for name in names)


class PoseTracker:
    """Wraps MediaPipe Pose and exposes PoseSnapshots."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._pose = _PoseEngine(
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> Optional[PoseSnapshot]:
        frame_rgb = frame_bgr[:, :, ::-1]
        results = self._pose.process(frame_rgb)
        if not results.pose_landmarks:
            return None

        points: Dict[str, np.ndarray] = {}
        landmarks = results.pose_landmarks.landmark
        for name, idx in LANDMARK_NAMES.items():
            lm = landmarks[idx]
            if lm.visibility is not None and lm.visibility < MIN_VISIBILITY:
                continue
            points[name] = np.array([lm.x, lm.y], dtype=np.float32)

        if not points:
            return None

        return PoseSnapshot(points=points, raw_landmarks=results.pose_landmarks)

    def close(self) -> None:
        try:
            self._pose.close()
        except Exception:
            pass

    def __enter__(self) -> "PoseTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()