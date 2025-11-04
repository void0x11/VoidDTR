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
- **Live Status Display**: Detection, Recording, and System status indicators
- **Live Video Feed**: Real-time camera stream with detection bounding boxes
- **Maximizable Window**: Resize to any screen size

---

## 📸 Screenshots & Live Demonstration

### 1️⃣ Main Application GUI (Standby Mode)

![Main GUI](https://github.com/void0x11/Obsbot-Intelligent-Security-Monitor/raw/main/media/GUI.png)

System actively monitoring with no detection. Dark theme UI with three status indicators showing green (standby).

---

### 2️⃣ Person Detected - Recording Started Instantly

![Person Detected](https://github.com/void0x11/Obsbot-Intelligent-Security-Monitor/raw/main/media/snapshot.jpg)

Person detected with green bounding box. Recording indicator turns red, snapshot captured with timestamp overlay.

---

### 3️⃣ Evidence Backup - Camera Disconnect Scenario

![Last Capture](https://github.com/void0x11/Obsbot-Intelligent-Security-Monitor/raw/main/media/still.png)

**Why 15-second snapshots matter:**

```
WITHOUT Snapshots: Camera disconnects → Video LOST ❌
WITH Snapshots:   Camera disconnects → Photos SAVED ✅
```

Even if video fails, you have timestamped photo evidence every 15 seconds.

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

- **CPU**: 15-25% during monitoring, 30-40% during recording
- **RAM**: ~300MB base, ~500MB with models loaded
- **Network**: Not required (works offline)

---

## 📥 Installation & Setup

### Prerequisites
- Windows 10/11, macOS, or Linux
- OBSBOT Tiny2 USB camera
- 4GB RAM (8GB recommended)
- 20GB free disk space

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
4. **Start Monitoring**: System enters continuous monitoring immediately
5. **Monitor Indicators**: 🟢 Green (Normal) | 🔴 Red (Alert) | 🟡 Orange (Cooldown)
6. **Exit**: Click "Exit" button to stop recording and close

### Output Files

- **`recordings/`** - Video files: `security_YYYYMMDD_HHMMSS.mp4`
- **`snapshots/`** - Images: `snapshot_YYYYMMDD_HHMMSS_mmm.jpg` (with timestamp)

---

## ⚙️ Configuration & Customization

Edit these settings in `OBSBOT_Tiny2_Intelligent_Security_Monitor.py`:

### 🎯 Main Settings

| Setting | Location | Default | Range | What It Does |
|---------|----------|---------|-------|--------------|
| **Cooldown Duration** | Line ~145 | 6s | 3-30s | How long to record after person leaves |
| **Snapshot Frequency** | Line ~171 | 15s | 5-60s | Time between snapshot captures |
| **Detection Confidence** | Line ~73 | 0.5 | 0.0-1.0 | Stricter = higher confidence needed |
| **Video Resolution** | Line ~51-53 | 1920×1080 | 1280×720 to 2560×1440 | Output quality |

### 💡 Quick Presets

**Standard Monitoring** (RECOMMENDED):
```python
cooldown_seconds = 6
screenshot_cooldown = 15
confidence_threshold = 0.5
```

**Security Critical**:
```python
cooldown_seconds = 15
screenshot_cooldown = 10
confidence_threshold = 0.6
```

**Storage Saving**:
```python
cooldown_seconds = 3
screenshot_cooldown = 30
confidence_threshold = 0.7
```

### How to Edit

1. Open `OBSBOT_Tiny2_Intelligent_Security_Monitor.py`
2. Find the line number from table above
3. Change the value and save
4. Run: `python build.py` (for .exe) or restart app

---

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not detected | Reconnect USB, check port, restart application |
| Low detection accuracy | Increase lighting, lower confidence threshold |
| High CPU usage | Verify single instance running |
| Large file sizes | Normal with 1920×1080 @ 30 FPS - plan storage |
| Missing snapshots | Check folder permissions, verify detection occurred |

---

## 📦 Deployment

### Standalone Executable
- No installation needed
- No Python required on target machine
- Icon in taskbar and window
- Works on any Windows PC with USB support

### Source Distribution
- Requires Python 3.8+
- Install: `pip install -r requirements.txt`
- Run: `python OBSBOT_Tiny2_Intelligent_Security_Monitor.py`

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

*OBSBOT Tiny2 Intelligent Security Monitor - Smart surveillance when you need it*
