# OBSBOT Tiny2 Intelligent Security Monitor

**Professional event-driven surveillance system with real-time human detection, automatic recording, and persistent snapshot evidence.**

An AI-powered security monitoring application purpose-built for OBSBOT Tiny2 USB cameras. Automatically captures timestamped video and snapshots when humans are detected, providing comprehensive evidence even if the camera connection is interrupted.

---

## 🎯 Key Features

### Smart Detection & Recording
- **Real-Time Detection**: YOLOv8 neural network processes 30 FPS video stream for instant human detection
- **Automatic Recording**: Records only when humans detected
- **Storage Effieceint**: it saves only videos when someone detected saves you tons of GBs.
- **Intelligent Cooldown**: 6-second recording buffer after person leaves frame prevents fragmented clips
- **Continuous Monitoring**: System stays alert 24/7 in standby mode with minimal CPU usage

### Persistent Evidence Collection
- **Automatic Snapshots**: Captures timestamped image every 15 seconds during detection
- **Timestamp Overlay**: Detection time written directly on each snapshot (format: YYYY-MM-DD HH:MM:SS)
- **Backup Protection**: If camera disconnects mid-recording, you still have photo evidence every 15 seconds
- **Organized Storage**: Separate folders for videos (`recordings/`) and images (`snapshots/`)

### User Interface
- **Professional Dark Theme**: Easy on the eyes, perfect for 24/7 monitoring setups
- **Live Status Display**:
  - 🟢 **Person Detection** - Shows when humans are detected (🔴 when active)
  - 🟢 **Recording Status** - Standby / Recording / Cooldown states
  - 🟢 **System Status** - Real-time system health monitoring
- **Live Video Feed**: Real-time camera stream with detection bounding boxes
- **Maximizable Window**: Resize to any screen size
- **OBSBOT Tiny2 Optimized**: Perfectly calibrated for OBSBOT Tiny2 USB camera

---

## 🎬 How It Works

### Detection & Recording Flow

```
Person Enters Frame
        ↓
[Detection: 🔴 DETECTED]
        ↓
Recording Starts (Video begins)
[Recording: 🔴 RECORDING]
        ↓
Person Still In Frame
        ↓
Video Continues + Snapshot Every 15 Sec
        ↓
Person Leaves Frame
        ↓
[Recording: 🟡 COOLDOWN] (6 seconds)
        ↓
If Person Returns → Resume Recording
If No Person After 6 Sec → Stop Recording
[Recording: 🟢 Standby]
```

### Why Snapshots Every 15 Seconds?

**Scenario: Camera Tampering or Disconnection**

| Time | Event | Video | Snapshot |
|------|-------|-------|----------|
| 14:30:00 | Person detected | ▶️ Recording starts | 📸 Image 1 |
| 14:30:15 | Person in frame | ▶️ Recording continues | 📸 Image 2 |
| 14:30:30 | Person in frame | ▶️ Recording continues | 📸 Image 3 |
| 14:30:45 | **Camera disconnected** | ❌ Recording stops | 📸 Image 4 |
| 14:31:00 | Camera offline | ❌ No video | 📸 Image 5 (last capture) |

**Result**: Even if camera goes offline, you have photographic evidence every 15 seconds starting from detection.

---

## 📥 Installation & Setup

### Prerequisites
- Windows 10/11, macOS, or Linux
- OBSBOT Tiny2 USB camera (or compatible USB webcam)
- 4GB RAM (8GB recommended)
- 20GB free disk space (for recordings)

### Quick Start

**Option A: Standalone Executable** (Recommended)
```bash
# Just download and run:
OBSBOT Tiny2 Intelligent Security Monitor.exe
```
No Python or additional setup needed!

**Option B: From Source**
```bash
pip install -r requirements.txt
python OBSBOT_Tiny2_Intelligent_Security_Monitor.py
```

---

## 🚀 Usage

1. **Connect OBSBOT Tiny2**: USB connection to computer
2. **Launch Application**: Run the executable or Python script
3. **Auto Detection**: System automatically detects and selects the camera
4. **Start Monitoring**: System enters continuous monitoring mode immediately
5. **Monitor Indicators**:
   - Green = Safe/Normal
   - Red = Active/Alert
   - Orange = Transitional (Cooldown)
6. **Exit**: Click "Exit" button to cleanly stop all recording and close application

### Output Files

After monitoring, check:
- **`recordings/`** - Video files: `security_YYYYMMDD_HHMMSS.mp4`
- **`snapshots/`** - Images: `snapshot_YYYYMMDD_HHMMSS_mmm.jpg` (with timestamp overlay)

---

## ⚙️ Configuration

All settings editable in `OBSBOT_Tiny2_Intelligent_Security_Monitor.py`:

| Setting | Location | Value | Purpose |
|---------|----------|-------|---------|
| Detection Confidence | Line ~73 | 0.0-1.0 | Higher = stricter detection |
| Recording Cooldown | Line ~145 | Seconds | How long after person leaves |
| Snapshot Frequency | Line ~171 | Seconds | Interval between snapshots (15 recommended) |

### Recommended Configuration

```python
# Detection sensitivity - balanced for indoor environments
confidence_threshold = 0.5  # Good balance between sensitivity and accuracy

# Recording cooldown - 6 seconds
cooldown_seconds = 6  # Enough to capture context, minimal fragmentation

# Snapshot frequency - every 15 seconds
screenshot_cooldown = 15  # Provides clear timeline of detection event
```

---

## 💾 Storage & Performance

### Expected Storage Usage (24 hours)

| Scenario | Video Size | Snapshots | Total |
|----------|-----------|-----------|-------|
| Continuous (24/7 recording) | 150-200 GB | 5-10 GB | 160-210 GB |
| Event-Driven (typical office) | 10-20 GB | 2-5 GB | 12-25 GB |
| Sparse events (1-2 per hour) | 1-3 GB | 500MB-1GB | 1.5-4 GB |

### System Requirements

- **CPU**: 15-25% during monitoring, 30-40% during detection/recording
- **RAM**: ~300MB base, ~500MB with models loaded
- **Disk I/O**: Only during recording events (minimal impact)
- **Network**: Not required (works offline)

---

## 🔍 Typical Use Cases

**Office Security**
- Monitor entry/exit points
- Detect unauthorized personnel
- Automatic incident documentation

**Research & Academia**
- Long-duration monitoring experiments
- Behavioral analysis with computer vision
- Event-triggered data collection

**Facility Protection**
- After-hours monitoring
- Perimeter surveillance
- Evidence collection for incidents

---

## 🛠️ Technical Details

**Core Technologies**
- **AI Detection**: YOLOv8 nano (lightweight, accurate)
- **GUI Framework**: PyQt5 (professional, responsive)
- **Video Processing**: OpenCV with H.264 codec
- **Threading**: 4 concurrent threads (Camera, Detection, Recording, Screenshots)

**Supported Hardware**
- **Camera**: OBSBOT Tiny2, standard USB webcams, V4L2-compliant devices
- **Resolution**: 1920×1080 @ 30 FPS
- **Output Format**: MP4 video + JPEG snapshots

---

## 📋 System Architecture

**Multi-Threaded Design**:
- **CameraThread**: Continuous 30 FPS capture from USB camera
- **DetectionThread**: Real-time YOLOv8 inference on frames
- **RecordingManager**: State machine for video recording lifecycle
- **ScreenshotManager**: Event-triggered snapshot capture with timestamp

This architecture ensures:
- Responsive UI (no freezing)
- Real-time detection (minimal latency)
- Reliable recording (no frame drops)
- Concurrent capture of video + snapshots

---

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not detected | Reconnect OBSBOT Tiny2, check USB port, restart application |
| Low detection accuracy | Increase lighting, adjust confidence threshold (lower value) |
| High CPU usage | Verify single instance running, reduce detection frequency |
| Large file sizes | Normal with 1920×1080 @ 30 FPS; plan storage accordingly |
| Missing snapshots | Check `snapshots/` folder permissions, verify detection occurred |

---

## 📦 Deployment

### Standalone Executable
- Download `OBSBOT Tiny2 Intelligent Security Monitor.exe`
- No installation needed
- No Python required on target machine
- Icon in taskbar and window
- Works on any Windows PC with USB support

### Source Distribution
- Requires Python 3.8+
- Install dependencies: `pip install -r requirements.txt`
- Run: `python OBSBOT_Tiny2_Intelligent_Security_Monitor.py`

---

## 🔐 Data Privacy

- **Local Processing**: All detection and recording happens locally on your machine
- **No Cloud**: No data sent to external servers
- **No Tracking**: Application doesn't collect any usage data
- **Full Control**: You own all recordings and snapshots

---

## 📝 License & Attribution

MIT License - Open for academic and commercial use

**Built with:**
- YOLOv8 (Ultralytics) - Real-time object detection
- PyQt5 - Professional GUI framework
- OpenCV - Computer vision processing
- PyTorch - Deep learning backend

---

**Version**: 1.0.0 | **Status**: Production Ready ✅

For updates and source code: [[GitHub Repository](https://github.com/void0x11/OBSBOT-Intelligent-Security-Monitor)]

---

*OBSBOT Tiny2 Intelligent Security Monitor - Smart surveillance when you need it*
