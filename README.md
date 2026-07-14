# Goalkeeper Reaction & Position Trainer 🧤⚽

An interactive, computer-vision-powered training workspace designed to benchmark and train a goalkeeper's breakout reaction time, stance stability, and dive-direction accuracy using a standard webcam feed.

Built with **Python**, **OpenCV**, and **MediaPipe**.

---

## 🚀 Features

*   **Real-Time Pose Tracking:** Maps key goalkeeper mechanics (shoulders, hips, wrists, knees, and ankles) seamlessly via a robust landmark interface.
*   **Dynamic Stance Calibration:** Automatically builds an environment-specific baseline to calculate center-of-mass and resting hand levels.
*   **Visual Cue Engine:** Injects randomized directional targets (`LEFT`, `RIGHT`, `UP_LEFT`, etc.) to trigger reactive positioning.
*   **High-Performance HUD & Telemetry:** Overlays live bone tracking structures, reaction speeds, metrics, and save histories directly onto the stream.
*   **Session Summary Dashboard:** Displays advanced analytics at the end of every drill, showcasing save accuracy percentages, fastest breakout speeds, and streak tracking.

---

## 📂 Project Structure

```text
goalkeeper_trainer/
├── data/
│   └── sessions/         # Logged performance sessions
├── training/
│   ├── __init__.py
│   └── session.py        # Core session logic and state machine
├── ui/
│   ├── __init__.py
│   └── renderer.py       # OpenCV HUD drawing engine
├── vision/
│   ├── __init__.py
│   ├── baseline.py       # Calibration math
│   ├── camera.py         # Capture streams
│   ├── movement.py       # Dive & breakout detection
│   └── pose_tracker.py   # Safe-bound MediaPipe tracking adapter
├── config.py             # Global constants and app settings
├── main.py               # Application orchestrator launcher
├── requirements.txt      # Project dependencies
└── README.md

```

---

## 🛠️ Installation & Setup

### 1. Prerequisite Environment

Ensure you are using Linux (Ubuntu tested) or macOS with Python 3.8+ installed.

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate

```

### 3. Dependencies Configuration

Install the required system packages. Due to potential namespace parsing quirks with specific distributions, install the core dependencies via:

```bash
pip install opencv-python numpy
pip install mediapipe==0.10.14

```

---

## 🎮 How to Run

Fire up the system orchestrator loop from the project root:

```bash
python main.py

```

### Workout Workflow:

1. **System Initialization:** The application checks the integrity of your video device interface.
2. **Calibration Phase:** Stand centered in your webcam's field of view in your primary "ready stance." Hold completely still during the short sample countdown while it records your resting metrics and calculates jitter values.
3. **Hold Position:** Wait out the variable pre-shot delay.
4. **React:** As soon as the green target direction arrow strikes the HUD, burst to that side to register your save latency metrics!
5. **Review Dashboard:** Press `R` anytime to reset the drill statistics or `Q` / `ESC` to quit out of the workspace safely.
