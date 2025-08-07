import cv2
import numpy as np
import csv
from collections import deque, defaultdict
from ultralytics import YOLO

"""
Drip-rate monitor using YOLO-based drop detection/tracking.
"""

# ------------------ CONFIGURATION ------------------
WINDOW_DURATION = 15   # seconds over which each drip-rate sample is computed
RECHECK_INTERVAL = 30  # seconds between successive drip-rate samples
# ---------------------------------------------------

class DropTracker:
    """Tracks individual drops; records the wall-clock *second* at which
    each valid drop disappears so window-based drip-rate stats can be
    derived later with minimal memory."""

    def __init__(self):
        self.active_tracks = {}
        self.completed_tracks = set()
        self.track_history = defaultdict(list)

        # Validation thresholds
        self.min_track_duration = 3   # ≥ frames a detection must persist
        self.min_y_travel      = 10  # ≥ px downward travel required

        # Global counters / timestamps
        self.drop_count  = 0
        self.drop_times  = deque()

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
                    'last_frame':  frame_idx,
                    'start_y':     y,
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
                    del self.active_tracks[tid]

    def drops_in_window(self, start_s: float, end_s: float) -> int:
        return sum(1 for t in self.drop_times if start_s <= t < end_s)

# ==============================================================

def detect_drip_rate(
    video_path: str,
    yolo_weights: str = "best.pt",
    output_csv: str = "drip_rate_log.csv",
    show_video: bool = True,
):
    """Run YOLO-based detection on *video_path* and compute periodic
    drip-rate statistics with a side-by-side interface."""

    model = YOLO(yolo_weights)
    cap   = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Warning: Could not determine video FPS. Assuming 30.")
        fps = 30

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    panel_width = 400
    combined_width = W + panel_width

    tracker   = DropTracker()
    frame_idx = 0
    latest_rate_dpm = None

    if show_video:
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = video_path.replace(".mp4", "_rate_interface.mp4")
        out_vid  = cv2.VideoWriter(out_path, fourcc, fps, (combined_width, H))

    csv_file = open(output_csv, "w", newline="")
    csv_wr   = csv.writer(csv_file)
    csv_wr.writerow(["window_start_s", "window_end_s", "drops", "rate_dpm"])

    next_check_time = WINDOW_DURATION

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        time_s = frame_idx / fps

        results = model.track(frame, persist=True, conf=0.25, verbose=False)[0]
        detections = []
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids   = results.boxes.id.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()
            for box, tid, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = map(int, box)
                detections.append({
                    "track_id": tid,
                    "center": (int((x1+x2)/2), int((y1+y2)/2)),
                    "confidence": conf,
                    "box": (x1, y1, x2, y2),
                })

        tracker.update(detections, frame_idx, time_s)

        if time_s >= next_check_time:
            win_start = next_check_time - WINDOW_DURATION
            drops = tracker.drops_in_window(win_start, next_check_time)
            latest_rate_dpm = drops * (60 / WINDOW_DURATION)
            print(f"⏱️  {win_start:.1f}s–{next_check_time:.1f}s: {drops} drops  →  {latest_rate_dpm:.1f} dpm")
            csv_wr.writerow([f"{win_start:.2f}", f"{next_check_time:.2f}", drops, f"{latest_rate_dpm:.2f}"])
            csv_file.flush()
            next_check_time += RECHECK_INTERVAL

        if show_video:
            vis_video = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = det['box']
                tid = det['track_id']
                cv2.rectangle(vis_video, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID {tid}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(vis_video, (x1, y1 - h - 15), (x1 + w, y1-5), (0, 255, 0), -1)
                cv2.putText(vis_video, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Create the grey panel for displaying information
            info_panel = np.full((H, panel_width, 3), (200, 200, 200), dtype=np.uint8)

            # Display stats on the info panel
            cv2.putText(info_panel, f"Total Drops: {tracker.drop_count}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            rate_text = f"Rate: {latest_rate_dpm:.1f} dpm" if latest_rate_dpm is not None else "Rate: Calculating..."
            cv2.putText(info_panel, rate_text, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            time_to_next = max(0, next_check_time - time_s)
            cv2.putText(info_panel, f"Next sample in: {time_to_next:.1f}s", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            # Combine the info panel and the video frame
            combined_view = np.hstack((info_panel, vis_video))

            out_vid.write(combined_view)
            cv2.imshow("Drip-Rate Monitor", combined_view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    csv_file.close()
    if show_video:
        out_vid.release()
        cv2.destroyAllWindows()
    print("🚰  Finished. Total drops:", tracker.drop_count)
    if show_video:
        print("Processed video saved to:", out_path)

# ==============================================================
if __name__ == "__main__":
    video_file_path = "F:\\CCVR project\\drip3.mp4"
    yolo_weights_path = "F:\\CCVR project\\final_working\\drop_counting.pt"

    detect_drip_rate(
        video_path=video_file_path,
        yolo_weights=yolo_weights_path,
    )