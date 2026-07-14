"""Application orchestration coordinating capture processing loops and lifecycle transitions."""
from __future__ import annotations

import sys
import time

import cv2

from config import AppConfig
from training.session import TrainingState
from training import TrainingSession
from ui import Renderer
from vision import BaselineBuilder, CameraError, CameraStream, MovementDetector, PoseTracker


def main() -> None:
    # Initialize standard configuration settings
    config = AppConfig()
    
    print("Initializing components...")
    try:
        camera = CameraStream(
            device_index=config.camera.device_index,
            width=config.camera.frame_width,
            height=config.camera.frame_height,
            flip_horizontal=config.camera.flip_horizontal,
        )
    except CameraError as e:
        print(f"\n[FATAL CAMERA ERROR]: {e}")
        sys.exit(1)

    pose_tracker = PoseTracker()
    renderer = Renderer()
    session = TrainingSession(config)

    # Calibration helper variables
    baseline_builder = BaselineBuilder()
    movement_detector = None

    print("\nSystem online. Launching display loop view...")
    session.change_state(TrainingState.CALIBRATING)

    try:
        while True:
            try:
                frame = camera.read()
            except CameraError as e:
                print(f"\n[FRAME CAPTURE WARNING]: {e}")
                continue

            # Process the frame using the pose estimation tracking models
            pose_snapshot = pose_tracker.process(frame)
            pose_found = pose_snapshot is not None

            # Route localized pose evaluation profiles contextually depending on the active state machine configuration
            current_state = session.state
            detected_direction = None

            if current_state == TrainingState.CALIBRATING:
                if pose_found and pose_snapshot is not None:
                    baseline_builder.add_sample(pose_snapshot)
                
                stability = baseline_builder.center_stability()
                renderer.draw_calibration_progress(
                    frame,
                    current=baseline_builder.sample_count,
                    total=config.calibration.required_samples,
                    stability=stability,
                )

                if baseline_builder.sample_count >= config.calibration.required_samples:
                    if stability <= config.calibration.max_center_std:
                        # Calibration holds within strict tolerance thresholds. Instantiating detector profile
                        base_profile = baseline_builder.build()
                        movement_detector = MovementDetector(
                            baseline=base_profile,
                            thresholds=config.thresholds,
                            consistency_frames=config.drill.consistency_frames,
                        )
                        print("[CALIBRATION MATCH SUCCESS]: Baseline stabilized. Drill ready.")
                        session.change_state(TrainingState.READY)
                    else:
                        print(f"[CALIBRATION FLUIDITY WARNING]: High jitter ({stability:.3f}). Resetting window.")
                        baseline_builder = BaselineBuilder()

            else:
                # Standard training run processing tracking metrics
                if pose_found and pose_snapshot is not None and movement_detector is not None:
                    analysis = movement_detector.analyze(pose_snapshot)
                    detected_direction = analysis.direction
                    
                    # Reset consistency tracking historical context array to enforce valid reactions across states
                    if current_state != TrainingState.CUE_ACTIVE:
                        movement_detector.reset()
                
                # Advance core state engine frames forward
                session.handle_frame(pose_found, detected_direction)
                renderer.render(frame, session, pose_snapshot)

            # Display composite window stream
            cv2.imshow("Goalkeeper Reaction Trainer MVP", frame)

            # Route input polling events
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # ESC or Q splits processing
                print("Closing workout session application...")
                break
            elif key == ord("r"):
                print("Session reset requested.")
                baseline_builder = BaselineBuilder()
                movement_detector = None
                session.reset_entire_drill()
                session.change_state(TrainingState.CALIBRATING)

    finally:
        camera.release()
        pose_tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()