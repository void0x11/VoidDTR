# OBSBOT Tiny2 Intelligent Security Monitor

**Professional event-driven surveillance system with real-time human detection, automatic recording, and persistent snapshot evidence.**

An AI-powered security monitoring application purpose-built for OBSBOT Tiny2 USB cameras. Automatically captures timestamped video and snapshots when humans are detected, providing comprehensive evidence even if the camera connection is interrupted.

---

## 🎯 Key Features

### Smart Detection & Recording
- **Real-Time Detection**: YOLOv8 neural network processes 30 FPS video stream for instant human detection
- **Automatic Recording**: Records only when humans detected (no wasted storage on empty scenes)
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

## 📸 Screenshots & Live Demonstration

### 1️⃣ Main Application GUI (Standby Mode)

![Main GUI](https://github.com/void0x11/Obsbot-Intelligent-Security-Monitor/raw/main/media/GUI.png)

**Application Status**: 
- ✅ System actively monitoring 24/7
- ✅ No person detected → Recording standby
- ✅ Dark theme UI for comfortable viewing
- ✅ Three status indicators clearly visible

**What You See**:
- Camera selector at top (USB Camera 1)
- Large live video feed (empty room)
- Three status boxes: Detection, Recording, System
- All indicators showing 🟢 (Green = Normal/Standby)

---

### 2️⃣ Person Detected - Recording Started Instantly

![Person Detected](https://github.com/void0x11/Obsbot-Intelligent-Security-Monitor/raw/main/media/snapshot.jpg)

**Instant Actions Triggered**:
- 🔴 **DETECTED** indicator turns red
- 🔴 **RECORDING** starts immediately
- 📸 **First snapshot captured** with timestamp
- 📹 **Video file created** in recordings folder

**What's Happening**:
- Person visible in video feed with green bounding box
- Detection count shows (1) person detected
- Recording status shows active 🔴
- Timestamp overlay on snapshot: `2025-11-04 14:30:15`

**Timeline Start**:
```
T=0s    → Person enters frame
T=0s    → YOLOv8 detection triggers
T=0s    → Recording starts
T=0s    → Snapshot #1 captured & timestamped
```

---

### 3️⃣ Evidence Backup - Last Snapshot Before Camera Disconnect

![Last Capture](https://github.com/void0x11/Obsbot-Intelligent-Security-Monitor/raw/main/media/still.png)

**Critical Feature Demonstration**:

This screenshot shows why **15-second snapshots are essential for evidence preservation**.

**Scenario: Person detected, then camera loses USB connection**

```
TRADITIONAL SYSTEM (Continuous Recording Only):
═════════════════════════════════════════════════
T=0s  → Recording starts ▶️
T=15s → Recording continues ▶️
T=30s → Recording continues ▶️
T=45s → CAMERA DISCONNECTED ❌
        Video file LOST ❌
RESULT: NO EVIDENCE ❌❌❌

OUR SYSTEM (Recording + 15-Second Snapshots):
═══════════════════════════════════════════════
T=0s  → Recording starts + Snapshot #1 📸
T=15s → Recording continues + Snapshot #2 📸
T=30s → Recording continues + Snapshot #3 📸
T=45s → CAMERA DISCONNECTED ❌ + Snapshot #4 📸 CAPTURED!
        Video may be lost, BUT 4 PHOTOS ARE SAFE ✅
RESULT: COMPLETE EVIDENCE ✅✅✅
```

**Each Snapshot Includes**:
- ✅ Exact timestamp (YYYY-MM-DD HH:MM:SS)
- ✅ High-quality person image
- ✅ Backup proof if video fails
- ✅ Cannot be disputed

---

## 🎬 How It Works - Complete Workflow

### Detection → Recording → Snapshots → Storage

**State Machine Flow**:

```
┌─────────────────────────────────────────────────────────┐
│ STANDBY (Monitoring)                                    │
│ 🟢 Detection: Not Detected                              │
│ 🟢 Recording: Standby                                   │
│ 🟢 System: Monitoring                                   │
└────────────────┬────────────────────────────────────────┘
                 │
        Person enters frame
                 ↓
┌─────────────────────────────────────────────────────────┐
│ DETECTION TRIGGERED                                     │
│ 🔴 Detection: DETECTED (1)                              │
│ [Green bounding box appears in video]                   │
└────────────────┬────────────────────────────────────────┘
                 │
        Detection threshold met
                 ↓
┌─────────────────────────────────────────────────────────┐
│ RECORDING ACTIVE                                        │
│ 🔴 Recording: RECORDING                                 │
│ 📹 Video file created: security_YYYYMMDD_HHMMSS.mp4    │
│ 📸 Snapshot taken every 15 seconds                      │
│ ⏱️ Timer counting up                                     │
└────────────────┬────────────────────────────────────────┘
                 │
     Person stays in frame or leaves
                 ↓
         ┌───────┴───────┐
         ↓               ↓
    Person In      Person Out
      Frame          Frame
         │               │
    Continue        Start Cooldown
     Recording       (6 seconds)
         │               │
         │          🟡 COOLDOWN
         │          [Waiting...]
         │               │
         └───────┬───────┘
                 │
        If returns within 6s
        OR cooldown expires
                 ↓
┌─────────────────────────────────────────────────────────┐
│ RECORDING STOPPED                                       │
│ 🟢 Recording: Standby                                   │
│ ✅ Video file saved                                      │
│ ✅ All snapshots saved with timestamps                  │
│ Ready for next detection                                │
└────────────────┬────────────────────────────────────────┘
                 │
        Return to STANDBY
                 ↓
         [Cycle Repeats]
```

---

## 📊 Real-World Example - Event Timeline

| Time | Detection | Recording | Event | Files Created |
|------|-----------|-----------|-------|---------------|
| 14:30:00 | 🔴 START | 🔴 START | Person enters frame | 📹 video_start.mp4<br>📸 snapshot_1.jpg |
| 14:30:15 | 🔴 ON | 🔴 REC | Person detected | 📸 snapshot_2.jpg |
| 14:30:30 | 🔴 ON | 🔴 REC | Continuous monitoring | 📸 snapshot_3.jpg |
| 14:30:45 | 🔴 ON | 🔴 REC | Still in frame | 📸 snapshot_4.jpg |
| 15:01:00 | 🟢 OFF | 🟡 COOL | Person left frame | 📸 snapshot_5.jpg |
| 15:01:06 | 🟢 OFF | 🟢 STOP | Cooldown expired | ✅ video_saved.mp4 |
| **RESULT** | **1 min event** | **1:06 duration** | **Complete evidence preserved** | **1 video + 5 photos** |

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

## 📥 Installation & Setup

### Prerequisites
- Windows 10/11, macOS, or Linux
- OBSBOT Tiny2 USB camera (or compatible USB webcam)
- 4GB RAM (8GB recommended)
- 20GB free disk space (for recordings)

### Quick Start

**Option A: Standalone Executable** (Recommended)
```bash
python cache_model.py        # First time only (2-3 min)
python build.py              # Build EXE (10-15 min)
# Run: dist\OBSBOT Tiny2 Intelligent Security Monitor.exe
```

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
   - 🟢 Green = Safe/Normal
   - 🔴 Red = Active/Alert
   - 🟡 Orange = Transitional (Cooldown)
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

---

*OBSBOT Tiny2 Intelligent Security Monitor - Smart surveillance when you need it*
