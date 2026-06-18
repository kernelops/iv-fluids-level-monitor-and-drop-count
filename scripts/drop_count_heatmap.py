"""
IV Drop Detection - Simple Rising Edge Detection
================================================
Counts drops when heatmap appears (rising edge), no reference line needed
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
from pathlib import Path
import torchvision.models as models
import torchvision.transforms as transforms

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class HeatmapDropDetector(nn.Module):
    """Drop detector with heatmap output"""
    
    def __init__(self, grid_size=26, pretrained=False):
        super(HeatmapDropDetector, self).__init__()
        
        self.grid_size = grid_size
        
        resnet = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        backbone_out_channels = 512
        
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(backbone_out_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
    
    def forward(self, x):
        features = self.backbone(x)
        heatmap = self.heatmap_head(features)
        heatmap = self.adaptive_pool(heatmap)
        return heatmap


# ============================================================================
# SIMPLE DROP COUNTER (Rising Edge Detection)
# ============================================================================

class SimpleDropCounter:
    """
    Simple drop counter using rising edge detection
    Counts when heatmap appears (0 → detected), waits until it disappears
    """
    
    def __init__(self, detection_threshold=0.2, cooldown_frames=5):
        """
        Args:
            detection_threshold: Minimum heatmap value to consider as drop
            cooldown_frames: Minimum frames to wait after counting
        """
        self.detection_threshold = detection_threshold
        self.cooldown_frames = cooldown_frames
        
        self.drop_count = 0
        self.drop_times = deque(maxlen=100)
        
        self.previous_state = False  # Was drop present in last frame?
        self.frames_since_count = 999  # Frames since last count
        
        print(f"\n{'='*60}")
        print("Simple Drop Counter Initialized")
        print(f"{'='*60}")
        print(f"  Detection threshold: {detection_threshold:.2f}")
        print(f"  Cooldown frames: {cooldown_frames}")
        print(f"  Strategy: Count when heatmap appears (rising edge)")
        print(f"{'='*60}\n")
    
    def update(self, heatmap, frame_idx):
        """
        Update counter with new heatmap
        
        Args:
            heatmap: (grid_size, grid_size) heatmap from model
            frame_idx: Current frame number
            
        Returns:
            dict with detection info
        """
        if torch.is_tensor(heatmap):
            heatmap = heatmap.cpu().numpy()
        
        # Find maximum value in heatmap
        max_value = np.max(heatmap)
        max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        
        # Current state: Is drop detected?
        current_state = max_value > self.detection_threshold
        
        # RISING EDGE DETECTION: Count when state changes from False → True
        if current_state and not self.previous_state:
            # Check cooldown to avoid double-counting
            if self.frames_since_count >= self.cooldown_frames:
                self.drop_count += 1
                self.frames_since_count = 0
                self.drop_times.append(time.time())
                print(f"  ✓ DROP #{self.drop_count} detected! (Confidence: {max_value:.3f})")
        
        # Update state
        self.previous_state = current_state
        self.frames_since_count += 1
        
        # Debug output every 30 frames
        if frame_idx % 30 == 0:
            state_str = "DROP PRESENT" if current_state else "no drop"
            print(f"[Frame {frame_idx:4d}] Max: {max_value:.3f} | State: {state_str}")
        
        return {
            'has_drop': current_state,
            'confidence': float(max_value),
            'position': max_pos,
            'drop_count': self.drop_count
        }
    
    def get_drop_rate(self, window_seconds=15):
        """Calculate drops per minute over recent window"""
        if len(self.drop_times) < 2:
            return 0.0
        
        current_time = time.time()
        recent_drops = [t for t in self.drop_times if current_time - t <= window_seconds]
        
        if len(recent_drops) < 2:
            return 0.0
        
        time_span = current_time - recent_drops[0]
        if time_span > 0:
            return (len(recent_drops) / time_span) * 60
        return 0.0


# ============================================================================
# VIDEO PROCESSOR
# ============================================================================

class VideoDropCounter:
    """Process video and count drops using simple heatmap detection"""
    
    def __init__(self, model_path, device='cuda', img_size=416, grid_size=26):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.grid_size = grid_size
        
        # Load model
        print(f"\n{'='*60}")
        print("LOADING MODEL")
        print(f"{'='*60}")
        print(f"Model path: {model_path}")
        print(f"Device: {self.device}")
        
        self.model = HeatmapDropDetector(grid_size=grid_size, pretrained=False)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ Model loaded successfully!")
        print(f"{'='*60}\n")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.counter = None
    
    def process_frame(self, frame):
        """Process frame and get heatmap"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            heatmap = self.model(input_tensor)[0, 0]
        
        return heatmap.cpu().numpy()
    
    def visualize_frame(self, frame, heatmap, detection_info):
        """Create visualization with heatmap overlay"""
        display = frame.copy()
        h, w = frame.shape[:2]
        
        # Resize heatmap to frame size
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Create colored heatmap
        heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3] * 255
        heatmap_colored = heatmap_colored.astype(np.uint8)
        
        # Blend with frame
        overlay = cv2.addWeighted(display, 0.6, heatmap_colored, 0.4, 0)
        
        # Draw detection marker if drop present
        if detection_info['has_drop'] and detection_info['position'] is not None:
            grid_y, grid_x = detection_info['position']
            marker_x = int(grid_x * w / self.grid_size)
            marker_y = int(grid_y * h / self.grid_size)
            
            # Pulsing circle
            cv2.circle(overlay, (marker_x, marker_y), 25, (0, 255, 0), 3)
            cv2.circle(overlay, (marker_x, marker_y), 8, (0, 0, 255), -1)
        
        # Info panel
        info_bg = overlay.copy()
        cv2.rectangle(info_bg, (0, 0), (w, 100), (0, 0, 0), -1)
        overlay = cv2.addWeighted(overlay, 0.7, info_bg, 0.3, 0)
        
        # Stats
        drop_count = detection_info['drop_count']
        drop_rate = self.counter.get_drop_rate() if self.counter else 0.0
        confidence = detection_info['confidence']
        
        cv2.putText(overlay, f"DROP COUNT: {drop_count}", (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(overlay, f"DROP RATE: {drop_rate:.1f} dpm", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Status indicator
        if detection_info['has_drop']:
            status_color = (0, 255, 0)
            status_text = f"DETECTING ({confidence:.2f})"
        else:
            status_color = (128, 128, 128)
            status_text = "WAITING"
        
        cv2.putText(overlay, status_text, (w-300, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        return overlay
    
    def process_video(self, video_path, output_path=None, 
                     detection_threshold=0.2, cooldown_frames=5,
                     display=True, save_video=True):
        """
        Process video with simple drop counting
        
        Args:
            video_path: Input video path
            output_path: Output video path (None = auto-generate)
            detection_threshold: Heatmap threshold (0.2 recommended)
            cooldown_frames: Frames between counts (5-10 recommended)
            display: Show real-time display
            save_video: Save output video
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {video_path}")
            return
        
        # Get properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"{'='*60}")
        print("VIDEO INFORMATION")
        print(f"{'='*60}")
        print(f"Path: {video_path}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Frames: {total_frames}")
        print(f"Duration: {duration:.1f} seconds")
        print(f"{'='*60}\n")
        
        # Initialize counter
        self.counter = SimpleDropCounter(
            detection_threshold=detection_threshold,
            cooldown_frames=cooldown_frames
        )
        
        # Video writer
        if save_video:
            if output_path is None:
                output_path = str(Path(video_path).stem) + "_counted.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output will be saved to: {output_path}\n")
        
        # Processing
        frame_idx = 0
        start_time = time.time()
        
        print(f"{'='*60}")
        print("PROCESSING VIDEO")
        print(f"{'='*60}\n")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get heatmap
                heatmap = self.process_frame(frame)
                
                # Update counter (simple rising edge detection)
                detection_info = self.counter.update(heatmap, frame_idx)
                
                # Visualize
                display_frame = self.visualize_frame(frame, heatmap, detection_info)
                
                # Save
                if save_video:
                    out.write(display_frame)
                
                # Display
                if display:
                    cv2.imshow('Drop Counter - Simple Detection', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n[INFO] Stopped by user")
                        break
                
                frame_idx += 1
        
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if save_video:
                out.release()
            if display:
                cv2.destroyAllWindows()
            
            # Final stats
            elapsed = time.time() - start_time
            
            print(f"\n{'='*60}")
            print("PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"Frames processed: {frame_idx}")
            print(f"Total drops counted: {self.counter.drop_count}")
            print(f"Average drop rate: {self.counter.get_drop_rate():.1f} dpm")
            print(f"Processing time: {elapsed:.1f} seconds")
            print(f"Average FPS: {frame_idx/elapsed:.1f}")
            if save_video:
                print(f"Output saved: {output_path}")
            print(f"{'='*60}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # CONFIGURATION
    MODEL_PATH = "<your_path>/models/drop_detector_heatmap.pth"
    VIDEO_PATH = "<your_path>/testing/drop_count/test3.mp4"
    OUTPUT_PATH = "output_simple_counted.mp4"
    
    # DETECTION PARAMETERS
    DETECTION_THRESHOLD = 0.2  # Adjust 0.15-0.30 based on sensitivity
    COOLDOWN_FRAMES = 10        # Frames to wait between counts (higher = safer)
    
    # Create processor
    processor = VideoDropCounter(
        model_path=MODEL_PATH,
        device='cuda',
        img_size=416,
        grid_size=26
    )
    
    # Process video
    processor.process_video(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        detection_threshold=DETECTION_THRESHOLD,
        cooldown_frames=COOLDOWN_FRAMES,
        display=True,
        save_video=True
    )
    
    print("\n✅ Done! Check the output video.")
