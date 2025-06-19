import cv2
import numpy as np
import time
import csv
from collections import defaultdict, deque
from ultralytics import YOLO

class DropTracker:
    def __init__(self):
        self.active_tracks = {}  # Currently tracked drops
        self.completed_tracks = set()  # IDs of drops that have been counted
        self.drop_count = 0
        self.min_track_duration = 3  # Min frames to consider valid
        self.min_y_travel = 10  # Min vertical travel distance
        self.last_positions = {}  # Store last position of each track
        self.track_history = defaultdict(list)  # Store positions history

    def update(self, detections, frame_idx, frame):
        # Current tracked IDs
        current_ids = set()
        
        # Detect new and current tracks
        for det in detections:
            track_id = det['track_id']
            current_ids.add(track_id)
            x, y = det['center']
            
            # Record position history
            self.track_history[track_id].append((x, y, frame_idx))
            
            # Check if this is a new track
            if track_id not in self.active_tracks and track_id not in self.completed_tracks:
                self.active_tracks[track_id] = {
                    'start_frame': frame_idx,
                    'last_frame': frame_idx,
                    'start_y': y,
                    'positions': [(x, y)]
                }
            elif track_id in self.active_tracks:
                # Update existing track
                self.active_tracks[track_id]['last_frame'] = frame_idx
                self.active_tracks[track_id]['positions'].append((x, y))
                self.last_positions[track_id] = (x, y)
                
        # Check for disappeared tracks
        disappeared_ids = set(self.active_tracks.keys()) - current_ids
        for track_id in disappeared_ids:
            track = self.active_tracks[track_id]
            
            # Calculate track duration and distance
            duration = track['last_frame'] - track['start_frame']
            
            # Only count if track exists long enough and shows downward movement
            if duration >= self.min_track_duration:
                positions = self.track_history[track_id]
                if len(positions) >= 2:
                    first_y = positions[0][1]
                    last_y = positions[-1][1]
                    y_travel = last_y - first_y
                    
                    # Verify consistent downward movement
                    if y_travel >= self.min_y_travel:
                        self.drop_count += 1
                        print(f"Counted drop #{self.drop_count} (track_id: {track_id}, duration: {duration} frames)")
            
            # Move to completed tracks
            self.completed_tracks.add(track_id)
            del self.active_tracks[track_id]
            
        return current_ids

def detect_drops_by_tracking(video_path, yolo_weights='best.pt', output_csv='drops_count.csv', show_video=True):
    model = YOLO(yolo_weights)
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    tracker = DropTracker()
    frame_idx = 0
    
    # Setup CSV writer
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['drop_id', 'time', 'x', 'y'])
        
        # Setup video writer if needed
        if show_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video = video_path.replace('.mp4', '_tracking_count.mp4')
            out = cv2.VideoWriter(output_video, fourcc, fps, (W, H))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            current_time = frame_idx / fps
            
            # Run YOLO tracking
            results = model.track(frame, persist=True, conf=0.25, verbose=False)[0]
            
            # Extract detections
            detections = []
            if hasattr(results, 'boxes') and hasattr(results.boxes, 'id') and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                ids = results.boxes.id.cpu().numpy().astype(int)
                confs = results.boxes.conf.cpu().numpy()
                
                for box, track_id, conf in zip(boxes, ids, confs):
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    detections.append({
                        'track_id': track_id,
                        'center': (cx, cy),
                        'confidence': conf,
                        'box': (x1, y1, x2, y2)
                    })
            
            # Update tracker
            current_ids = tracker.update(detections, frame_idx, frame)
            
            # Visualization
            if show_video:
                vis = frame.copy()
                
                # Draw active tracks
                for det in detections:
                    track_id = det['track_id']
                    cx, cy = det['center']
                    x1, y1, x2, y2 = det['box']
                    
                    # Different color for new vs existing tracks
                    if track_id in tracker.active_tracks:
                        # Get track duration
                        duration = frame_idx - tracker.active_tracks[track_id]['start_frame']
                        # Color changes from red to green as track persists longer
                        if duration < tracker.min_track_duration:
                            color = (0, 0, 255)  # Red for new tracks
                            status = "TENT"
                        else:
                            color = (0, 255, 0)  # Green for established tracks
                            status = "CONF"
                    else:
                        color = (255, 0, 0)  # Blue for unknown status
                        status = "UNK"
                        
                    # Draw bounding box
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Draw ID and status
                    cv2.putText(vis, f"{track_id} {status}", (cx+5, cy-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Draw trail from history
                    if track_id in tracker.track_history:
                        history = tracker.track_history[track_id]
                        pts = [p[:2] for p in history[-10:]]  # Last 10 positions
                        if len(pts) > 1:
                            for i in range(len(pts) - 1):
                                cv2.line(vis, pts[i], pts[i+1], color, 2)
                
                # Show counter with red text for visibility
                cv2.putText(vis, f"Drops: {tracker.drop_count}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Write frame to video
                if show_video:
                    out.write(vis)
                    cv2.imshow('Drop Tracking', vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    # Cleanup
    cap.release()
    if show_video:
        out.release()
        cv2.destroyAllWindows()
        
    print(f"✅ Final count: {tracker.drop_count} drops")
    print(f"Output video saved to: {output_video}")
    return tracker.drop_count

if __name__ == "__main__":
    detect_drops_by_tracking("F:\\CCVR project\\drip3.mp4", yolo_weights="F:\\CCVR project\\final_working\\drop_counting.pt")
