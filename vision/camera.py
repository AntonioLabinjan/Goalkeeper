"""Webcam capture wrapper with defensive error handling."""
from __future__ import annotations

import cv2
import numpy as np


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or a frame cannot be read."""


class CameraStream:
    """Thin wrapper around cv2.VideoCapture with clear failure modes."""

    def __init__(
        self,
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        flip_horizontal: bool = True,
    ) -> None:
        self._flip = flip_horizontal
        self._cap = cv2.VideoCapture(device_index)
        if not self._cap.isOpened():
            raise CameraError(
                f"Could not open camera at index {device_index}. "
                "Check that a webcam is connected, that no other application "
                "is using it, and that OS camera permissions are granted."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> np.ndarray:
        """Reads a single frame. Raises CameraError on failure."""
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CameraError("Failed to read a frame from the camera.")
        if self._flip:
            frame = cv2.flip(frame, 1)
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()

    def __enter__(self) -> "CameraStream":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()