"""
Detection-Triggered Recorder
Intelligent event-driven surveillance system with real-time human detection
Dark Theme Version
"""

import sys
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QComboBox, QPushButton, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from ultralytics import YOLO
import queue
import time


def get_available_cameras():
    """Detect only USB cameras, skip built-in camera at index 0"""
    available_cameras = []
    
    # Start from index 1 to skip built-in camera (index 0)
    # Try camera indices 1-10 (USB cameras typically)
    for index in range(1, 11):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(index)
            cap.release()
    
    return available_cameras


def identify_camera_type(index):
    """Identify camera type"""
    # All detected cameras are USB (since we skip index 0 in get_available_cameras)
    return "USB Camera"


class CameraThread(QThread):
    """Thread for continuous camera capture from specific camera index"""
    frame_ready = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            self.error_signal.emit(f"Failed to open camera at index {self.camera_index}")
            return
        
        # Set camera properties for high quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while self.running:
            ret, frame = cap.read()
            if ret:
                self.frame_ready.emit(frame)
            else:
                time.sleep(0.1)

        cap.release()

    def stop(self):
        self.running = False


class DetectionThread(QThread):
    """Thread for YOLOv8 human detection"""
    detection_result = pyqtSignal(bool, list, np.ndarray)

    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        super().__init__()
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)

                # Run inference - detect only person class (0)
                results = self.model(frame, classes=[0], conf=self.confidence_threshold, verbose=False)

                # Extract detections
                boxes = []
                human_detected = False

                if len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        if box.conf[0] >= self.confidence_threshold:
                            human_detected = True
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            boxes.append((int(x1), int(y1), int(x2), int(y2)))

                # Emit both detection result AND the frame for recording
                self.detection_result.emit(human_detected, boxes, frame)

            except queue.Empty:
                continue

    def add_frame(self, frame):
        """Add frame to processing queue"""
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass

    def stop(self):
        self.running = False


class RecordingManager(QThread):
    """Manages recording state machine and video writing"""
    recording_status = pyqtSignal(str, bool)

    STATE_STANDBY = 0
    STATE_DETECTED = 1
    STATE_COOLDOWN = 2

    def __init__(self, save_dir='recordings', cooldown_seconds=6):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.cooldown_seconds = cooldown_seconds

        self.state = self.STATE_STANDBY
        self.frame_queue = queue.Queue(maxsize=30)
        self.video_writer = None
        self.cooldown_timer = 0
        self.running = False
        self.current_filename = None

    def run(self):
        self.running = True

        while self.running:
            try:
                frame, human_detected = self.frame_queue.get(timeout=1)

                # State machine logic
                if self.state == self.STATE_STANDBY:
                    if human_detected:
                        self._start_recording(frame)
                        self.state = self.STATE_DETECTED
                        self.recording_status.emit("RECORDING", True)

                elif self.state == self.STATE_DETECTED:
                    if human_detected:
                        # Continue recording
                        if self.video_writer is not None:
                            self.video_writer.write(frame)
                    else:
                        # Human left frame - start cooldown
                        self.state = self.STATE_COOLDOWN
                        self.cooldown_timer = time.time()
                        self.recording_status.emit("COOLDOWN", True)

                elif self.state == self.STATE_COOLDOWN:
                    if human_detected:
                        # Human returned - resume recording
                        self.state = self.STATE_DETECTED
                        self.recording_status.emit("RECORDING", True)
                    else:
                        # Continue cooldown
                        if self.video_writer is not None:
                            self.video_writer.write(frame)

                        # Check if cooldown expired
                        if time.time() - self.cooldown_timer >= self.cooldown_seconds:
                            self._stop_recording()
                            self.state = self.STATE_STANDBY
                            self.recording_status.emit("STANDBY", False)

            except queue.Empty:
                continue

    def _start_recording(self, frame):
        """Initialize video writer and start recording"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_filename = self.save_dir / f"security_{timestamp}.mp4"

        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(str(self.current_filename), fourcc, 30.0, (w, h))
        self.video_writer.write(frame)
        print(f"[RECORDING STARTED] {self.current_filename}")

    def _stop_recording(self):
        """Stop recording and save video file"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            print(f"[RECORDING STOPPED] Saved: {self.current_filename}")

    def add_frame(self, frame, human_detected):
        """Add frame and detection status to processing queue"""
        try:
            self.frame_queue.put_nowait((frame, human_detected))
        except queue.Full:
            pass

    def stop(self):
        self.running = False
        if self.video_writer is not None:
            self.video_writer.release()


class ScreenshotManager(QThread):
    """Manages screenshot capture on person detection"""
    screenshot_saved = pyqtSignal(str)

    def __init__(self, save_dir='snapshots'):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.frame_queue = queue.Queue(maxsize=5)
        self.running = False
        self.last_screenshot_time = 0
        self.screenshot_cooldown = 30  # Take screenshot at most every 30 seconds

    def run(self):
        self.running = True
        while self.running:
            try:
                frame, human_detected = self.frame_queue.get(timeout=1)

                if human_detected:
                    current_time = time.time()
                    # Only take screenshot if cooldown has passed
                    if current_time - self.last_screenshot_time >= self.screenshot_cooldown:
                        self._save_screenshot(frame)
                        self.last_screenshot_time = current_time

            except queue.Empty:
                continue

    def _save_screenshot(self, frame):
        """Save screenshot with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        filename = self.save_dir / f"snapshot_{timestamp}.jpg"
        
        # Create copy for annotation
        annotated_frame = frame.copy()
        
        # Add timestamp text to the frame
        datetime_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated_frame, datetime_text, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        # Save the frame
        cv2.imwrite(str(filename), annotated_frame)
        print(f"[SCREENSHOT SAVED] {filename}")
        self.screenshot_saved.emit(str(filename))

    def add_frame(self, frame, human_detected):
        """Add frame and detection status to processing queue"""
        try:
            self.frame_queue.put_nowait((frame, human_detected))
        except queue.Full:
            pass

    def stop(self):
        self.running = False


class MainWindow(QMainWindow):
    """Main application window - Maximizable with improved GUI and Dark Theme"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OBSBOT-Tiny2-Intelligent-Security-Monitor")
        self.setGeometry(100, 100, 1280, 900)

        # Set window icon
        icon_path = Path("icon.ico")
        if icon_path.exists():
            app_icon = QIcon(str(icon_path))
            self.setWindowIcon(app_icon)
            print("[SYSTEM] Application icon loaded successfully")
        else:
            print("[WARNING] icon.ico not found in project directory")

        # Initialize components
        self.camera_thread = None
        self.detection_thread = None
        self.recording_manager = None
        self.screenshot_manager = None
        self.current_boxes = []
        self.selected_camera_index = 0
        self.monitoring_active = False

        self._init_ui()
        self._detect_cameras()

    def _init_ui(self):
        """Initialize user interface with improved layout and dark theme"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        central_widget.setStyleSheet("background-color: #1e1e1e;")

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # Top section: Camera selection
        camera_layout = QHBoxLayout()
        camera_layout.setContentsMargins(0, 0, 0, 0)
        camera_layout.setSpacing(10)
        camera_label = QLabel("Select Camera:")
        camera_label.setFont(QFont("Arial", 12, QFont.Bold))
        camera_label.setStyleSheet("color: #e0e0e0;")
        self.camera_combo = QComboBox()
        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)
        self.camera_combo.setMinimumWidth(300)
        self.camera_combo.setFixedHeight(35)
        self.camera_combo.setStyleSheet("""
            QComboBox {
                background-color: #333333;
                color: #e0e0e0;
                border: 2px solid #444;
                border-radius: 4px;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
            }
        """)
        camera_layout.addWidget(camera_label)
        camera_layout.addWidget(self.camera_combo)
        camera_layout.addStretch()
        main_layout.addLayout(camera_layout)

        # Video display section
        self.video_label = QLabel("Detecting cameras...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 3px solid #555; background-color: #000; border-radius: 5px;")
        self.video_label.setMinimumHeight(550)
        main_layout.addWidget(self.video_label, 1)  # Expandable

        # Status indicators section - IMPROVED LAYOUT with Dark Theme
        status_container = QFrame()
        status_container.setStyleSheet("background-color: #2d2d2d; border: 2px solid #444; border-radius: 8px;")
        status_container.setFixedHeight(150)
        status_layout = QHBoxLayout(status_container)
        status_layout.setContentsMargins(20, 15, 20, 15)
        status_layout.setSpacing(25)

        # Left section: Person Detection Status
        detection_box = QFrame()
        detection_box.setStyleSheet("background-color: #333333; border: 2px solid #444; border-radius: 8px;")
        detection_layout = QVBoxLayout(detection_box)
        detection_layout.setContentsMargins(15, 12, 15, 12)
        detection_layout.setSpacing(5)
        
        detection_title = QLabel("Person Detection")
        detection_title.setFont(QFont("Arial", 12, QFont.Bold))
        detection_title.setStyleSheet("color: #e0e0e0;")
        detection_layout.addWidget(detection_title)
        
        self.detection_status = QLabel("🟢 Not Detected")
        self.detection_status.setFont(QFont("Arial", 13, QFont.Bold))
        self.detection_status.setStyleSheet("color: green;")
        detection_layout.addWidget(self.detection_status)
        
        status_layout.addWidget(detection_box, 1)

        # Middle section: Recording Status
        recording_box = QFrame()
        recording_box.setStyleSheet("background-color: #333333; border: 2px solid #444; border-radius: 8px;")
        recording_layout = QVBoxLayout(recording_box)
        recording_layout.setContentsMargins(15, 12, 15, 12)
        recording_layout.setSpacing(5)
        
        recording_title = QLabel("Recording Status")
        recording_title.setFont(QFont("Arial", 12, QFont.Bold))
        recording_title.setStyleSheet("color: #e0e0e0;")
        recording_layout.addWidget(recording_title)
        
        self.recording_status = QLabel("🟢 Standby")
        self.recording_status.setFont(QFont("Arial", 13, QFont.Bold))
        self.recording_status.setStyleSheet("color: green;")
        recording_layout.addWidget(self.recording_status)
        
        status_layout.addWidget(recording_box, 1)

        # Right section: System Status
        system_box = QFrame()
        system_box.setStyleSheet("background-color: #333333; border: 2px solid #444; border-radius: 8px;")
        system_layout = QVBoxLayout(system_box)
        system_layout.setContentsMargins(15, 12, 15, 12)
        system_layout.setSpacing(5)
        
        system_title = QLabel("System Status")
        system_title.setFont(QFont("Arial", 12, QFont.Bold))
        system_title.setStyleSheet("color: #e0e0e0;")
        system_layout.addWidget(system_title)
        
        self.system_status = QLabel("🟢 Monitoring")
        self.system_status.setFont(QFont("Arial", 13, QFont.Bold))
        self.system_status.setStyleSheet("color: green;")
        system_layout.addWidget(self.system_status)
        
        status_layout.addWidget(system_box, 1)

        main_layout.addWidget(status_container)

        # Bottom section: Control buttons
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(10)
        button_layout.addStretch()

        self.close_button = QPushButton("Exit")
        self.close_button.clicked.connect(self.close_application)
        self.close_button.setFixedSize(180, 50)
        self.close_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #d32f2f;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #b71c1c;
            }
            QPushButton:pressed {
                background-color: #9a0007;
            }
        """)
        button_layout.addWidget(self.close_button)
        
        main_layout.addLayout(button_layout)

    def _detect_cameras(self):
        """Detect available cameras"""
        available = get_available_cameras()
        
        if not available:
            self.video_label.setText("No USB cameras detected! Please connect a USB camera.")
            return

        self.camera_combo.clear()
        for idx in available:
            self.camera_combo.addItem(f"USB Camera {idx}", idx)

        # Automatically select first USB camera found
        self.camera_combo.setCurrentIndex(0)
        
        # Start monitoring immediately
        self.on_camera_changed()

    def on_camera_changed(self):
        """Handle camera selection change"""
        if self.monitoring_active:
            self.stop_monitoring()
        
        self.selected_camera_index = self.camera_combo.currentData()
        self.start_monitoring()

    def start_monitoring(self):
        """Start camera, detection, recording, and screenshot threads"""
        # Start camera thread
        self.camera_thread = CameraThread(camera_index=self.selected_camera_index)
        self.camera_thread.frame_ready.connect(self.on_frame_ready)
        self.camera_thread.error_signal.connect(self.on_camera_error)
        self.camera_thread.start()

        # Start detection thread
        self.detection_thread = DetectionThread()
        self.detection_thread.detection_result.connect(self.on_detection_result)
        self.detection_thread.start()

        # Start recording manager
        self.recording_manager = RecordingManager()
        self.recording_manager.recording_status.connect(self.on_recording_status)
        self.recording_manager.start()

        # Start screenshot manager
        self.screenshot_manager = ScreenshotManager()
        self.screenshot_manager.screenshot_saved.connect(self.on_screenshot_saved)
        self.screenshot_manager.start()

        self.monitoring_active = True
        self.system_status.setText("🟢 Monitoring")
        self.system_status.setStyleSheet("color: green; font-weight: bold;")
        print(f"[SYSTEM] Monitoring started on camera index {self.selected_camera_index}")

    def stop_monitoring(self):
        """Stop all threads"""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait()

        if self.detection_thread:
            self.detection_thread.stop()
            self.detection_thread.wait()

        if self.recording_manager:
            self.recording_manager.stop()
            self.recording_manager.wait()

        if self.screenshot_manager:
            self.screenshot_manager.stop()
            self.screenshot_manager.wait()

        self.monitoring_active = False
        self.system_status.setText("🟡 Stopped")
        self.system_status.setStyleSheet("color: orange; font-weight: bold;")
        self.recording_status.setText("🟢 Standby")
        self.recording_status.setStyleSheet("color: green; font-weight: bold;")
        self.detection_status.setText("🟢 Not Detected")
        self.detection_status.setStyleSheet("color: green; font-weight: bold;")

    def close_application(self):
        """Close the application"""
        self.stop_monitoring()
        self.close()

    @pyqtSlot(np.ndarray)
    def on_frame_ready(self, frame):
        """Handle new frame from camera"""
        if self.detection_thread:
            self.detection_thread.add_frame(frame)

        # Display frame with bounding boxes
        self._display_frame(frame)

    @pyqtSlot(bool, list, np.ndarray)
    def on_detection_result(self, human_detected, boxes, frame):
        """Handle detection results"""
        self.current_boxes = boxes

        # Update detection indicator
        if human_detected:
            self.detection_status.setText(f"🔴 DETECTED ({len(boxes)})")
            self.detection_status.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.detection_status.setText("🟢 Not Detected")
            self.detection_status.setStyleSheet("color: green; font-weight: bold;")

        # Send to recording manager
        if self.recording_manager:
            self.recording_manager.add_frame(frame, human_detected)

        # Send to screenshot manager
        if self.screenshot_manager:
            self.screenshot_manager.add_frame(frame, human_detected)

    @pyqtSlot(str, bool)
    def on_recording_status(self, status, is_recording):
        """Handle recording status updates"""
        if status == "RECORDING":
            self.recording_status.setText("🔴 RECORDING")
            self.recording_status.setStyleSheet("color: red; font-weight: bold;")
        elif status == "COOLDOWN":
            self.recording_status.setText("🟡 COOLDOWN")
            self.recording_status.setStyleSheet("color: orange; font-weight: bold;")
        else:  # STANDBY
            self.recording_status.setText("🟢 Standby")
            self.recording_status.setStyleSheet("color: green; font-weight: bold;")

    @pyqtSlot(str)
    def on_camera_error(self, error_msg):
        """Handle camera errors"""
        self.video_label.setText(f"Camera Error: {error_msg}")
        self.system_status.setText("🔴 Error")
        self.system_status.setStyleSheet("color: red; font-weight: bold;")

    @pyqtSlot(str)
    def on_screenshot_saved(self, filename):
        """Handle screenshot saved signal"""
        print(f"[SYSTEM] Screenshot captured: {filename}")

    def _display_frame(self, frame):
        """Convert and display frame in UI"""
        display_frame = frame.copy()
        
        # Draw bounding boxes
        for (x1, y1, x2, y2) in self.current_boxes:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, "Person", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert to QPixmap and display
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image).scaled(self.video_label.size(), 
                                                    Qt.KeepAspectRatio, 
                                                    Qt.SmoothTransformation)
        self.video_label.setPixmap(pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
