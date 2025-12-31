"""
Combined IV Monitoring System
- YOLOv8 Drop Detection & Tracking
- CNN Fluid Level Classification
- Time Remaining Estimation
- Rule-Based Anomaly Detection
"""

import cv2
import numpy as np
import csv
from collections import deque, defaultdict
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image as pil_image
import PIL.ImageOps
from datetime import datetime

# ==================== CONFIGURATION ====================
# Drop Detection
WINDOW_DURATION = 15   # seconds for drip-rate sample
RECHECK_INTERVAL = 30  # seconds between samples

# Level Detection
LEVEL_MODEL_PATH = "iv-fluids-level-detection-model.h5"
MODEL_IMG_SIZE = 32
CLASS_LABELS = ['sal_data_100', 'sal_data_80', 'sal_data_50', 'sal_data_empty']
LEVEL_PERCENT_MAP = {
    'sal_data_100': 100,
    'sal_data_80': 80,
    'sal_data_50': 50,
    'sal_data_empty': 0,
}
ALERT_THRESHOLD_PCT = 50

# ===== NEW: Time & Anomaly Configuration =====
# User-configurable parameters for the specific IV setup
TOTAL_BAG_VOLUME_ML = 500  # Total volume of the IV bag (e.g., 1000ml, 500ml)
DROP_FACTOR_GTT_PER_ML = 15 # Drops per mL (check IV tubing spec, common values are 10, 15, 20)

# Anomaly Detection Thresholds
FREE_FLOW_THRESHOLD_DPM = 200  # Rate above which a free-flow alert is triggered
FLOW_STOP_SECONDS = 60         # Seconds with no drops to trigger a stopped alert

# UI Layout
INFO_PANEL_WIDTH = 400
FONT = cv2.FONT_HERSHEY_SIMPLEX
# =======================================================

class DropTracker:
    """Tracks drops using YOLO detections"""
    
    def __init__(self):
        self.active_tracks = {}
        self.completed_tracks = set()
        self.track_history = defaultdict(list)
        
        self.min_track_duration = 3
        self.min_y_travel = 10
        
        self.drop_count = 0
        self.drop_times = deque()
    
    def _register_drop(self, track_id: int, end_time_s: float):
        self.drop_count += 1
        self.drop_times.append(end_time_s)
        self.completed_tracks.add(track_id)
        if track_id in self.active_tracks:
            del self.active_tracks[track_id]
        
        max_age = WINDOW_DURATION + RECHECK_INTERVAL
        while self.drop_times and end_time_s - self.drop_times[0] > max_age:
            self.drop_times.popleft()
    
    def update(self, detections, frame_idx: int, time_s: float):
        current_ids = set()
        for det in detections:
            tid = det['track_id']
            current_ids.add(tid)
            x, y = det['center']
            self.track_history[tid].append((x, y, frame_idx))
            
            if tid not in self.active_tracks and tid not in self.completed_tracks:
                self.active_tracks[tid] = {
                    'start_frame': frame_idx,
                    'last_frame': frame_idx,
                    'start_y': y,
                }
            elif tid in self.active_tracks:
                self.active_tracks[tid]['last_frame'] = frame_idx
        
        disappeared = set(self.active_tracks) - current_ids
        for tid in disappeared:
            if tid in self.active_tracks:
                track = self.active_tracks[tid]
                duration = track['last_frame'] - track['start_frame']
                positions = self.track_history[tid]
                if duration >= self.min_track_duration and len(positions) >= 2:
                    y_travel = positions[-1][1] - positions[0][1]
                    if y_travel >= self.min_y_travel:
                        self._register_drop(tid, time_s)
                else:
                    if tid in self.active_tracks:
                        del self.active_tracks[tid]
    
    def drops_in_window(self, start_s: float, end_s: float) -> int:
        return sum(1 for t in self.drop_times if start_s <= t < end_s)

    # ===== NEW: Helper method for anomaly detection =====
    def get_time_of_last_drop(self) -> float:
        """Returns the timestamp of the most recently counted drop."""
        if not self.drop_times:
            return 0.0
        return self.drop_times[-1]

def preprocess_for_level_model(frame, target_size=(MODEL_IMG_SIZE, MODEL_IMG_SIZE)):
    # ... (this function is unchanged)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = pil_image.fromarray(rgb)
    inverted_pil = PIL.ImageOps.invert(pil_img)
    preprocessed_pil = inverted_pil.resize(target_size, pil_image.LANCZOS)
    model_input = np.expand_dims(np.array(preprocessed_pil) / 255.0, axis=0)
    return model_input


def combined_iv_monitor(
    video_path: str,
    yolo_weights: str = "best.pt",
    level_model_path: str = LEVEL_MODEL_PATH,
    output_csv: str = "combined_iv_log.csv",
):
    # ... (model loading and video setup is unchanged) ...
    drop_model = YOLO(yolo_weights)
    level_model = tf.keras.models.load_model(level_model_path, compile=False)
    print("✓ Models loaded")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    combined_width = W + INFO_PANEL_WIDTH
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = video_path.replace(".mp4", "_combined_monitor.mp4")
    out_vid = cv2.VideoWriter(out_path, fourcc, fps, (combined_width, H))
    
    csv_file = open(output_csv, "w", newline="")
    csv_wr = csv.writer(csv_file)
    csv_wr.writerow(["time_s", "drop_count", "drip_rate_dpm", "level_pct", "level_class", "alert", "anomaly_status"])
    
    tracker = DropTracker()
    frame_idx = 0
    latest_rate_dpm = 0.0
    next_check_time = WINDOW_DURATION
    
    print("Processing video...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_idx += 1
        time_s = frame_idx / fps
        
        # ... (Drop Detection, Tracking, and Rate Calculation are unchanged) ...
        results = drop_model.track(frame, persist=True, conf=0.25, verbose=False)[0]
        detections = []
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy().astype(int)
            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                detections.append({"track_id": tid, "center": (int((x1+x2)/2), int((y1+y2)/2)), "box": (x1, y1, x2, y2)})
        tracker.update(detections, frame_idx, time_s)
        if time_s >= next_check_time:
            win_start = next_check_time - WINDOW_DURATION
            drops = tracker.drops_in_window(win_start, next_check_time)
            latest_rate_dpm = drops * (60 / WINDOW_DURATION)
            next_check_time += RECHECK_INTERVAL
        
        # ... (Level Detection and Alert Logic are unchanged) ...
        level_input = preprocess_for_level_model(frame)
        probs = level_model.predict(level_input, verbose=0)[0]
        pred_idx = np.argmax(probs)
        pred_label = CLASS_LABELS[pred_idx]
        confidence = float(probs[pred_idx])
        level_pct = LEVEL_PERCENT_MAP.get(pred_label, 0)
        
        alert_msg = "Normal"
        alert_color = (0, 150, 0)
        if level_pct == 0:
            alert_msg = "EMPTY!"
            alert_color = (0, 0, 255)
        elif level_pct <= ALERT_THRESHOLD_PCT:
            alert_msg = "LOW (<=50%)"
            alert_color = (0, 165, 255)

        # ===== NEW: Time Remaining Estimation Logic =====
        time_remaining_str = "Calculating..."
        if latest_rate_dpm > 0:
            total_drops_in_bag = TOTAL_BAG_VOLUME_ML * DROP_FACTOR_GTT_PER_ML
            drops_remaining = total_drops_in_bag * (level_pct / 100.0)
            minutes_remaining = drops_remaining / latest_rate_dpm
            if minutes_remaining > 60:
                hours = int(minutes_remaining / 60)
                mins = int(minutes_remaining % 60)
                time_remaining_str = f"~{hours}h {mins}m to empty"
            else:
                time_remaining_str = f"~{int(minutes_remaining)} min to empty"

        # ===== NEW: Anomaly Detection Logic =====
        anomaly_status = "Normal"
        anomaly_color = (0, 150, 0) # Green

        # Check for Free-Flow
        if latest_rate_dpm > FREE_FLOW_THRESHOLD_DPM:
            anomaly_status = "ANOMALY: FREE-FLOW"
            anomaly_color = (0, 0, 255) # Red
        
        # Check for Flow Stoppage (only after the first drop has been counted)
        time_since_last_drop = time_s - tracker.get_time_of_last_drop()
        if tracker.drop_count > 0 and time_since_last_drop > FLOW_STOP_SECONDS:
            anomaly_status = "ANOMALY: FLOW STOPPED"
            anomaly_color = (0, 100, 255) # Orange
        
        # Log to CSV
        csv_wr.writerow([f"{time_s:.2f}", tracker.drop_count, f"{latest_rate_dpm:.2f}", 
                        level_pct, pred_label, alert_msg, anomaly_status])
        
        # ===== VISUALIZATION (Updated) =====
        vis_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['box']
            tid = det['track_id']
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_frame, f"ID {tid}", (x1, y1-10), FONT, 0.5, (0, 255, 0), 2)
        
        info_panel = np.full((H, INFO_PANEL_WIDTH, 3), (220, 220, 220), dtype=np.uint8)
        
        y_pos = 50
        cv2.putText(info_panel, "IV Monitor", (20, y_pos), FONT, 1.2, (0, 0, 0), 3); y_pos += 20
        cv2.line(info_panel, (20, y_pos), (INFO_PANEL_WIDTH-20, y_pos), (0, 0, 0), 2); y_pos += 40
        
        cv2.putText(info_panel, "DROP DETECTION", (20, y_pos), FONT, 0.8, (0, 0, 0), 2); y_pos += 40
        cv2.putText(info_panel, f"Total: {tracker.drop_count}", (30, y_pos), FONT, 0.9, (0, 0, 0), 2); y_pos += 40
        cv2.putText(info_panel, f"Rate: {latest_rate_dpm:.1f} dpm", (30, y_pos), FONT, 0.9, (0, 0, 0), 2); y_pos += 40
        time_to_next = max(0, next_check_time - time_s)
        cv2.putText(info_panel, f"Next sample in: {time_to_next:.1f}s", (30, y_pos), FONT, 0.7, (100, 100, 100), 2); y_pos += 30
        
        cv2.line(info_panel, (20, y_pos), (INFO_PANEL_WIDTH-20, y_pos), (0, 0, 0), 1); y_pos += 40
        
        cv2.putText(info_panel, "LEVEL MONITOR", (20, y_pos), FONT, 0.8, (0, 0, 0), 2); y_pos += 40
        cv2.putText(info_panel, f"Level: {level_pct}%", (30, y_pos), FONT, 0.9, (0, 0, 0), 2); y_pos += 40
        cv2.putText(info_panel, f"Conf: {confidence:.2f}", (30, y_pos), FONT, 0.9, (0, 0, 0), 2); y_pos += 40
        # ===== NEW: Add time remaining to panel =====
        cv2.putText(info_panel, time_remaining_str, (30, y_pos), FONT, 0.8, (0, 0, 150), 2); y_pos += 30
        
        cv2.line(info_panel, (20, y_pos), (INFO_PANEL_WIDTH-20, y_pos), (0, 0, 0), 1); y_pos += 40
        cv2.putText(info_panel, "LEVEL STATUS:", (20, y_pos), FONT, 0.8, (0, 0, 0), 2); y_pos += 20
        cv2.rectangle(info_panel, (20, y_pos), (INFO_PANEL_WIDTH-20, y_pos+60), alert_color, -1); y_pos += 40
        cv2.putText(info_panel, alert_msg, (40, y_pos), FONT, 1.0, (255, 255, 255), 3); y_pos += 50
        
        # ===== NEW: Add anomaly status to panel =====
        cv2.line(info_panel, (20, y_pos), (INFO_PANEL_WIDTH-20, y_pos), (0, 0, 0), 1); y_pos += 40
        cv2.putText(info_panel, "ANOMALY STATUS:", (20, y_pos), FONT, 0.8, (0, 0, 0), 2); y_pos += 20
        cv2.rectangle(info_panel, (20, y_pos), (INFO_PANEL_WIDTH-20, y_pos+60), anomaly_color, -1); y_pos += 40
        cv2.putText(info_panel, anomaly_status, (40, y_pos), FONT, 0.9, (255, 255, 255), 2)
        
        combined_view = np.hstack((info_panel, vis_frame))
        
        out_vid.write(combined_view)
        cv2.imshow("Combined IV Monitor", combined_view)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release()
    out_vid.release()
    csv_file.close()
    cv2.destroyAllWindows()
    
    print(f"\n✓ Complete! Total drops: {tracker.drop_count}")
    print(f"✓ Output video: {out_path}")
    print(f"✓ Log saved: {output_csv}")


# ==================== RUN ====================
if __name__ == "__main__":
    VIDEO_PATH = "C:\\Users\\hegde\\Documents\\Github\\iv-fluids-level-monitor-and-drop-count\\IV-fluids-vids\\Dual model\\final.mp4"
    YOLO_WEIGHTS = "best.pt"       # CHANGE THIS
    
    combined_iv_monitor(
        video_path=VIDEO_PATH,
        yolo_weights=YOLO_WEIGHTS,
        level_model_path=LEVEL_MODEL_PATH
    )