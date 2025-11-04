# Detection-Triggered Recorder (Void-DTR)

An **intelligent, event-driven surveillance system** that automatically captures video and screenshots when humans are detected in real-time using **AI vision processing**.

---

## Overview

**Detection-Triggered Recorder** is a sophisticated **security monitoring application**, designed for **research** and **autonomous surveillance**. It continuously monitors a USB-connected camera feed, performs real-time human detection using **YOLOv8**, and automatically initiates recording only when humans are detected in the scene. This event-driven approach significantly reduces storage requirements compared to traditional 24/7 recording while ensuring comprehensive event documentation.

---

## Key Features

### Core Functionality
- **Real-Time Human Detection**: YOLOv8 Convolutional Neural Network (CNN) processes video frames at 30 FPS for instant detection.
- **Automatic Recording**: Only records when humans are detected, saving storage space.
- **Smart Cooldown Logic**: 5-second cooldown after detection to prevent fragmented clips.
- **Timestamped Snapshots**: Automatically captures screenshots every 2 seconds during detection events.
- **Low CPU Usage**: Operates in standby mode, consuming minimal resources while remaining responsive to detections.

### Recording Features
- **High-Quality Video**: 1920×1080 resolution at 30 FPS in MP4 format with H.264 encoding.
- **Automatic Timestamping**: Video files are automatically timestamped in `YYYYMMDD_HHMMSS` format.
- **Efficient Storage**: Video files are stored in a `recordings/` directory.
- **H.264 Compression**: Balances video quality and file size.

### Screenshot Features
- **Event-Triggered Snapshots**: Captures images every 2 seconds during detection.
- **Timestamp Overlay**: Displays detection timestamps on each captured image.
- **Organized Storage**: Screenshots stored in the `snapshots/` directory with millisecond precision.
- **Smart Snapshot Frequency**: Ensures no duplicate images are taken during continuous detection events.

### User Interface
- **PyQt5 GUI**: Professional, easy-to-use desktop interface.
- **Live Video Feed**: Real-time camera feed with detection bounding boxes.
- **Status Indicators**:
  - **Person Detection Status**: Displays whether a person is detected (Not Detected / DETECTED).
  - **Recording Status**: Indicates whether the system is recording, in standby, or in cooldown.
  - **System Health**: Shows the current system status (Monitoring / Stopped / Error).
- **Camera Selection**: Dropdown to select from available USB or built-in cameras.
- **Resizable Window**: UI that adapts to different screen sizes.

---

## System Architecture

### Threading Model
- **CameraThread**: Continuously captures frames from the USB camera at 30 FPS.
- **DetectionThread**: Performs real-time YOLOv8 inference on captured frames.
- **RecordingManager**: Manages recording state transitions (Standby → Detected → Cooldown).
- **ScreenshotManager**: Handles automatic screenshot capture and timestamping during detection.
- **Main UI Thread**: Handles GUI rendering and user interaction.

### Data Flow
```text
USB Camera (1920×1080, 30 FPS)
    ↓
CameraThread (frame capture)
    ↓
DetectionThread (YOLOv8 inference)
    ↓ (detection_result signal)
    ├→ RecordingManager (starts/stops recording)
    ├→ ScreenshotManager (captures snapshots)
    └→ Main UI (updates status indicators)
    ↓
Video Output: recordings/*.mp4
Image Output: snapshots/*.jpg
```

---

## Installation

### Prerequisites
- Python 3.8+.
- `pip` package manager.
- USB camera or built-in webcam.

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/detection-triggered-recorder.git
cd detection-triggered-recorder
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
python security_monitor.py
```

---

## Usage

### Starting the Application
1. Connect the USB camera to the computer.
2. Run the application: `python security_monitor.py`.
3. The system will automatically detect available cameras.
4. Select the desired camera from the dropdown (USB camera is pre-selected).
5. The system will enter monitoring mode and begin detection.

### During Operation
- **Live Video Feed**: Displays the real-time camera stream with detection bounding boxes.
- **Status Indicators**:
  - **Person Detection**: 🟢 Not Detected / 🔴 DETECTED.
  - **Recording**: 🟢 Standby / 🔴 RECORDING / 🟡 COOLDOWN.
  - **System Health**: 🟢 Monitoring / 🟡 Stopped / 🔴 Error.
- **Fully Automated**: No manual control needed; system automatically starts and stops recording.
- **Graceful Exit**: Press EXIT to stop all threads and close the application properly.

### Output Files
After running the system, the generated files will be stored as follows:
```
project-root/
├── recordings/
│   ├── security_20251103_143500.mp4
│   ├── security_20251103_143530.mp4
└── snapshots/
    ├── snapshot_20251103_143502_045.jpg  [with timestamp overlay]
    ├── snapshot_20251103_143505_123.jpg  [with timestamp overlay]
```

---

## Configuration

Edit `security_monitor.py` to customize the following parameters:

### Detection Sensitivity
```python
self.confidence_threshold = 0.5  # Range: 0.0-1.0 (higher values = stricter detection)
```

### Recording Cooldown
```python
self.cooldown_seconds = 5  # Seconds after last detection before stopping recording
```

### Screenshot Frequency
```python
self.screenshot_cooldown = 2  # Seconds between snapshots during detection
```

---

## Performance Characteristics

- **CPU Usage**: ~15-25% during monitoring (idle), ~30-40% during detection/recording.
- **Memory Usage**: ~200-300 MB without models loaded, ~400-500 MB with models.
- **GPU Acceleration**: Optional, for faster detection (requires CUDA support).
- **Detection Latency**: ~30-50ms per frame.
- **Storage Efficiency**:
  - **Event-Driven Recording**: ~5-15 GB for 24 hours.
  - **Snapshots**: ~1-3 GB for 24 hours.

---

## Directory Structure

```
detection-triggered-recorder/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── security_monitor.py          # Main application script
├── recordings/                  # Video output directory (created automatically)
├── snapshots/                   # Screenshot output directory (created automatically)
└── docs/
    ├── INSTALLATION.md          # Detailed installation guide
    ├── USAGE.md                 # Advanced usage scenarios
    ├── TROUBLESHOOTING.md       # Common issues and solutions
    └── API.md                   # Code documentation
```

---

## Dependencies

```
PyQt5==5.15.9              # GUI framework
opencv-python==4.8.1.78    # Video capture and processing
ultralytics==8.0.196       # YOLOv8 detection model
numpy==1.24.3              # Numerical computing
Pillow==10.0.1             # Image processing
torch==2.0.1               # Deep learning backend
```

---

## System Requirements

### Minimum
- Python 3.8+.
- 4 GB RAM.
- 2 GB free disk space.
- USB camera (USB 2.0+).
- Processor: Intel i5 or equivalent.

### Recommended
- Python 3.10+.
- 8 GB RAM.
- 20 GB free disk space (for video recordings).
- USB 3.0 camera connection.
- Processor: Intel i7 or higher.
- GPU: NVIDIA GPU with CUDA support (optional but recommended).

### Supported Operating Systems
- ✅ **Windows 10/11**.
- ✅ **macOS 10.14+**.
- ✅ **Linux** (Ubuntu 18.04+, Debian, Fedora).

---

## License

MIT License — See [LICENSE](./LICENSE) for full details.

---

## Acknowledgments

- **YOLOv8**: Thanks to Ultralytics for providing cutting-edge object detection models.
- **PyQt5**: For the powerful and efficient GUI framework.
- **OpenCV**: For video capture and image processing functionalities.
- **OBSBOT**: For high-quality USB cameras that support this system.

---

## Contributing

Contributions are welcome! Some areas for improvement include:
- Multi-camera support.
- Advanced filtering (reducing false positives).
- Cloud integration for video storage.
- Mobile app companion.
- Performance optimizations.

For contributions, please open a pull request on GitHub or create an issue.

---

## Troubleshooting

### Camera Not Detected
- Ensure the camera is properly connected and recognized by the operating system.
- Check the camera driver or try a different USB port.

### Low Detection Performance
- Lower the `confidence_threshold` if the model is not detecting people.
- Ensure good lighting and camera focus.

### High CPU Usage
- Use a lighter YOLO model (e.g., `yolov8n.pt`).
- Enable
