"""
Combined IV Monitoring System - Approach B  [v4 - Live Oscilloscope + Threshold Tuner]
========================================================================================
NEW: Live confidence oscilloscope drawn in the info panel.
     Shows the last 150 frames of heatmap max-confidence as a scrolling line.
     Horizontal lines show current HIGH (blue) and LOW (grey) thresholds.
     Use keyboard to tune thresholds live without restarting:

  Threshold controls:
    +  /  =   Raise HIGH threshold by 0.05
    -         Lower HIGH threshold by 0.05
    ]         Raise LOW  threshold by 0.05
    [         Lower LOW  threshold by 0.05
    r         Re-draw ROI
    q         Quit

Watch the oscilloscope: each drop should appear as a sharp spike.
Set HIGH just below the spike peaks and LOW below the trough baseline.
"""

import cv2
import numpy as np
import csv
import time
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import tensorflow as tf
from PIL import Image as pil_image
import PIL.ImageOps
from collections import deque

# ==================== CONFIGURATION ====================
VIDEO_PATH         = r"<your_path>/testing/integrated/final.mp4"
HEATMAP_MODEL_PATH = r"<your_path>/models/drop_detector_heatmap.pth"
LEVEL_MODEL_PATH   = r"<your_path>/models/iv-fluids-level-detection-model.h5"
OUTPUT_CSV         = "combined_heatmap_iv_log.csv"

# Starting thresholds — tune live with +/-/[/] keys
HIGH_THRESHOLD_INIT = 0.12
LOW_THRESHOLD_INIT  = 0.04
THRESHOLD_STEP      = 0.02

COOLDOWN_FRAMES      = 5
IMG_SIZE             = 416
GRID_SIZE            = 26
WINDOW_DURATION      = 15
HEATMAP_ALPHA        = 0.45
LEVEL_RECHECK_FRAMES = 10
OSCILLOSCOPE_LEN     = 150    # number of frames shown in scrolling plot

MODEL_IMG_SIZE    = 32
CLASS_LABELS      = ['sal_data_100', 'sal_data_80', 'sal_data_50', 'sal_data_empty']
LEVEL_PERCENT_MAP = {'sal_data_100': 100, 'sal_data_80': 80,
                     'sal_data_50': 50,   'sal_data_empty': 0}
ALERT_THRESHOLD_PCT     = 50
TOTAL_BAG_VOLUME_ML     = 500
DROP_FACTOR_GTT_PER_ML  = 15
FREE_FLOW_THRESHOLD_DPM = 200
FLOW_STOP_SECONDS       = 60

INFO_PANEL_WIDTH = 420
FONT             = cv2.FONT_HERSHEY_SIMPLEX
# =======================================================


# ── Model
class HeatmapDropDetector(nn.Module):
    def __init__(self, grid_size=26, pretrained=False):
        super().__init__()
        self.grid_size = grid_size
        resnet = models.resnet18(pretrained=pretrained)
        self.backbone     = nn.Sequential(*list(resnet.children())[:-2])
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(512,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(True),nn.Dropout2d(0.2),
            nn.Conv2d(256,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(True),nn.Dropout2d(0.2),
            nn.Conv2d(128, 64,3,padding=1),nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64,   1,1), nn.Sigmoid(),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))

    def forward(self, x):
        return self.adaptive_pool(self.heatmap_head(self.backbone(x)))


# ── Hysteresis counter (thresholds mutable at runtime)
class HeatmapDropCounter:
    def __init__(self, high=HIGH_THRESHOLD_INIT, low=LOW_THRESHOLD_INIT,
                 cooldown=COOLDOWN_FRAMES):
        self.high  = high
        self.low   = low
        self.cd    = cooldown
        self.drop_count       = 0
        self.drop_video_times = deque()
        self.state            = False
        self.frames_since     = 999

    def update(self, heatmap_np, frame_idx, time_s):
        max_val = float(np.max(heatmap_np))
        max_pos = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
        if not self.state:
            if max_val > self.high and self.frames_since >= self.cd:
                self.state = True
                self.drop_count += 1
                self.drop_video_times.append(time_s)
                self.frames_since = 0
                print(f"  DROP #{self.drop_count:3d}  t={time_s:6.2f}s  conf={max_val:.3f}  HIGH={self.high:.2f}")
        else:
            if max_val < self.low:
                self.state = False
        self.frames_since += 1
        return {'has_drop': self.state, 'confidence': max_val, 'position': max_pos}

    def get_rate_dpm(self, time_s, window=WINDOW_DURATION):
        recent = [t for t in self.drop_video_times if time_s - t <= window]
        return len(recent) * (60.0 / window)

    def time_of_last_drop(self):
        return self.drop_video_times[-1] if self.drop_video_times else 0.0


# ── Helpers
def preprocess_for_level_model(frame, size=(MODEL_IMG_SIZE, MODEL_IMG_SIZE)):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inv = PIL.ImageOps.invert(pil_image.fromarray(rgb)).resize(size, pil_image.LANCZOS)
    return np.expand_dims(np.array(inv) / 255.0, axis=0).astype(np.float32)


def apply_heatmap_to_roi(frame, heatmap_np, roi, alpha=HEATMAP_ALPHA):
    x, y, w, h = roi
    vis  = frame.copy()
    crop = frame[y:y+h, x:x+w]
    hm   = (cv2.resize(heatmap_np, (w, h)) * 255).astype(np.uint8)
    vis[y:y+h, x:x+w] = cv2.addWeighted(
        crop, 1-alpha, cv2.applyColorMap(hm, cv2.COLORMAP_JET), alpha, 0)
    cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,255), 2)
    cv2.putText(vis, "DRIP CHAMBER ROI", (x, y-8), FONT, 0.52, (0,255,255), 1)
    return vis


def draw_oscilloscope(conf_history, high_thresh, low_thresh,
                      width, height):
    """
    Returns a BGR image of shape (height, width, 3) showing:
      - scrolling confidence signal (white line)
      - HIGH threshold (blue dashed line)
      - LOW  threshold (grey dashed line)
      - count-event markers (green triangles)
      - shaded region between HIGH and LOW
    """
    osc = np.zeros((height, width, 3), dtype=np.uint8)
    osc[:] = (30, 30, 30)   # dark background

    n = len(conf_history)
    if n < 2:
        cv2.putText(osc, "Collecting signal...", (10, height//2),
                    FONT, 0.5, (150,150,150), 1)
        return osc

    # Y mapping: conf 0→bottom, 1→top  with 4px margin
    M = 4
    def cy(v):
        return int((height - M) - v * (height - 2*M))

    # Shaded dead-zone between LOW and HIGH
    y_hi = cy(high_thresh)
    y_lo = cy(low_thresh)
    cv2.rectangle(osc, (0, y_hi), (width, y_lo), (20, 40, 20), -1)

    # HIGH threshold line (blue)
    for x in range(0, width, 8):
        cv2.line(osc, (x, y_hi), (min(x+4, width), y_hi), (220, 80, 80), 1)
    cv2.putText(osc, f"H={high_thresh:.2f}", (2, y_hi-3), FONT, 0.38, (220,80,80), 1)

    # LOW threshold line (grey)
    for x in range(0, width, 8):
        cv2.line(osc, (x, y_lo), (min(x+4, width), y_lo), (120,120,120), 1)
    cv2.putText(osc, f"L={low_thresh:.2f}", (2, y_lo+10), FONT, 0.38, (120,120,120), 1)

    # Signal line
    vals    = list(conf_history)
    x_step  = width / OSCILLOSCOPE_LEN
    pts     = []
    for i, v in enumerate(vals):
        px = int(i * x_step)
        py = cy(v)
        pts.append((px, py))

    for i in range(1, len(pts)):
        cv2.line(osc, pts[i-1], pts[i], (220, 220, 220), 1)

    # Current value dot
    if pts:
        cv2.circle(osc, pts[-1], 3, (0, 255, 255), -1)

    # Y-axis ticks at 0.25 intervals
    for tick in [0.25, 0.50, 0.75, 1.0]:
        ty = cy(tick)
        cv2.line(osc, (0, ty), (4, ty), (80,80,80), 1)
        cv2.putText(osc, f"{tick:.2f}", (width-32, ty+4), FONT, 0.3, (80,80,80), 1)

    return osc


def draw_info_panel(H, drop_count, rate_dpm,
                    conf_history, has_drop, high_thresh, low_thresh,
                    level_pct, pred_label, level_conf, time_remaining_str,
                    alert_msg, alert_color, anomaly_status, anomaly_color,
                    display_fps):

    panel = np.full((H, INFO_PANEL_WIDTH, 3), (220,220,220), dtype=np.uint8)
    y = 40

    # Title
    cv2.putText(panel, "IV Monitor [Heatmap+CNN]", (8,y), FONT,0.72,(0,0,0),2); y+=16
    cv2.line(panel,(8,y),(INFO_PANEL_WIDTH-8,y),(0,0,0),2);                      y+=28

    # Drop section
    cv2.putText(panel,"DROP DETECTION",(8,y),FONT,0.72,(30,30,150),2);           y+=30
    cv2.putText(panel,f"Total Drops: {drop_count}",(16,y),FONT,0.82,(0,0,0),2); y+=34
    cv2.putText(panel,f"Rate:  {rate_dpm:.1f} dpm",(16,y),FONT,0.82,(0,0,0),2); y+=34

    # State badge
    state_txt   = "  DROP PRESENT " if has_drop else "  waiting...   "
    state_color = (0,160,0) if has_drop else (90,90,90)
    cv2.rectangle(panel,(16,y),(INFO_PANEL_WIDTH-16,y+26),state_color,-1)
    cv2.putText(panel, state_txt,(20,y+18),FONT,0.62,(255,255,255),2);           y+=34

    # Threshold readout + tuning hint
    cv2.putText(panel,f"HIGH={high_thresh:.2f}  LOW={low_thresh:.2f}",
                (16,y),FONT,0.6,(60,60,60),1);                                   y+=20
    cv2.putText(panel,"+/- : HIGH    [ / ] : LOW",
                (16,y),FONT,0.48,(120,80,0),1);                                  y+=20

    # ── Oscilloscope
    osc_h = 90
    osc   = draw_oscilloscope(conf_history, high_thresh, low_thresh,
                               INFO_PANEL_WIDTH - 16, osc_h)
    panel[y:y+osc_h, 8:INFO_PANEL_WIDTH-8] = osc
    y += osc_h + 10

    cv2.line(panel,(8,y),(INFO_PANEL_WIDTH-8,y),(0,0,0),1);                      y+=22

    # Level section
    cv2.putText(panel,"LEVEL MONITOR",(8,y),FONT,0.72,(30,30,150),2);            y+=30
    cv2.putText(panel,f"Level: {level_pct}%",(16,y),FONT,0.82,(0,0,0),2);       y+=34
    cv2.putText(panel,f"Class: {pred_label}",(16,y),FONT,0.58,(80,80,80),1);    y+=24
    cv2.putText(panel,f"Conf:  {level_conf:.2f}",(16,y),FONT,0.82,(0,0,0),2);   y+=34
    cv2.putText(panel, time_remaining_str,(16,y),FONT,0.72,(0,0,180),2);         y+=28

    cv2.line(panel,(8,y),(INFO_PANEL_WIDTH-8,y),(0,0,0),1);                      y+=22

    # Level alert
    cv2.putText(panel,"LEVEL STATUS:",(8,y),FONT,0.72,(0,0,0),2);                y+=16
    cv2.rectangle(panel,(8,y),(INFO_PANEL_WIDTH-8,y+50),alert_color,-1);         y+=32
    cv2.putText(panel,alert_msg,(18,y),FONT,1.0,(255,255,255),3);                y+=36

    cv2.line(panel,(8,y),(INFO_PANEL_WIDTH-8,y),(0,0,0),1);                      y+=20

    # Anomaly alert
    cv2.putText(panel,"ANOMALY STATUS:",(8,y),FONT,0.72,(0,0,0),2);              y+=16
    cv2.rectangle(panel,(8,y),(INFO_PANEL_WIDTH-8,y+50),anomaly_color,-1);       y+=32
    cv2.putText(panel,anomaly_status,(18,y),FONT,0.75,(255,255,255),2);          y+=36

    # FPS footer
    if y < H - 20:
        cv2.putText(panel,f"FPS: {display_fps:.1f}",(8,H-12),FONT,0.55,(80,80,80),1)

    return panel


def select_roi(frame, win):
    guide = frame.copy()
    cv2.putText(guide,"DRAW BOX around drip chamber → press ENTER/SPACE",
                (10,30),FONT,0.68,(0,255,255),2)
    roi = cv2.selectROI(win, guide, fromCenter=False, showCrosshair=True)
    cv2.setWindowTitle(win, win)
    x,y,w,h = roi
    if w < 20 or h < 20:
        H_f,W_f = frame.shape[:2]
        print("[WARN] No ROI — using full frame")
        return (0, 0, W_f, H_f)
    print(f"[INFO] ROI: x={x} y={y} w={w} h={h}")
    return (x,y,w,h)


# ============================================================
#  MAIN
# ============================================================
def combined_heatmap_iv_monitor(
    video_path         = VIDEO_PATH,
    heatmap_model_path = HEATMAP_MODEL_PATH,
    level_model_path   = LEVEL_MODEL_PATH,
    output_csv         = OUTPUT_CSV,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] Device: {device}")

    hm_model = HeatmapDropDetector(grid_size=GRID_SIZE, pretrained=False)
    ckpt     = torch.load(heatmap_model_path, map_location=device)
    hm_model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    hm_model.to(device).eval()
    print("[INFO] Heatmap model loaded ✓")

    lv_model = tf.keras.models.load_model(level_model_path, compile=False)
    @tf.function(reduce_retracing=True)
    def fast_lv(x):
        return lv_model(x, training=False)
    _ = fast_lv(tf.zeros((1, MODEL_IMG_SIZE, MODEL_IMG_SIZE, 3)))
    print("[INFO] Level CNN loaded & warmed up ✓")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}"); return

    fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_f   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    comb_w    = W + INFO_PANEL_WIDTH

    WIN = "Heatmap+CNN IV Monitor"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, comb_w, H)

    ret, first = cap.read()
    if not ret: return
    roi = select_roi(first, WIN)
    roi_x, roi_y, roi_w, roi_h = roi
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out_path = video_path.replace(".mp4","_heatmap_monitor.mp4")
    out_vid  = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (comb_w, H))

    csv_fh = open(output_csv,"w",newline="")
    csv_wr = csv.writer(csv_fh)
    csv_wr.writerow(["time_s","drop_count","drip_rate_dpm","heatmap_conf",
                     "drop_state","HIGH","LOW","level_pct","level_class",
                     "level_conf","alert","anomaly_status","proc_fps"])

    counter      = HeatmapDropCounter(high=HIGH_THRESHOLD_INIT, low=LOW_THRESHOLD_INIT)
    conf_history = deque(maxlen=OSCILLOSCOPE_LEN)
    frame_idx    = 0
    display_fps  = 0.0
    t_fps_ref    = time.perf_counter()
    time_s       = 0.0

    level_pct          = -1
    pred_lbl           = "..."
    lev_conf           = 0.0
    alert_msg          = "Initialising"
    alert_color        = (100,100,100)
    time_remaining_str = "Calculating..."

    print(f"[INFO] {W}x{H} @ {fps:.0f}fps | {total_f} frames")
    print("[INFO] Controls: +/- (HIGH thresh)  [/] (LOW thresh)  r (ROI)  q (quit)\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_idx += 1
        time_s = frame_idx / fps

        if frame_idx % 30 == 0:
            display_fps = 30.0 / (time.perf_counter() - t_fps_ref + 1e-9)
            t_fps_ref   = time.perf_counter()

        # ── Heatmap inference on ROI crop
        crop    = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        rgb_c   = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        inp     = transform(rgb_c).unsqueeze(0).to(device)
        with torch.no_grad():
            heatmap_np = hm_model(inp)[0,0].cpu().numpy()

        det = counter.update(heatmap_np, frame_idx, time_s)
        conf_history.append(det['confidence'])
        rate_dpm = counter.get_rate_dpm(time_s)

        # ── Level CNN (throttled)
        if frame_idx % LEVEL_RECHECK_FRAMES == 1:
            probs     = fast_lv(tf.constant(preprocess_for_level_model(frame)))[0].numpy()
            pred_idx  = int(np.argmax(probs))
            pred_lbl  = CLASS_LABELS[pred_idx]
            lev_conf  = float(probs[pred_idx])
            level_pct = LEVEL_PERCENT_MAP[pred_lbl]
            if level_pct == 0:
                alert_msg, alert_color = "EMPTY!",      (0,0,255)
            elif level_pct <= ALERT_THRESHOLD_PCT:
                alert_msg, alert_color = "LOW (<=50%)", (0,140,255)
            else:
                alert_msg, alert_color = "Normal",      (0,150,0)
            if rate_dpm > 0 and level_pct >= 0:
                mins = TOTAL_BAG_VOLUME_ML * DROP_FACTOR_GTT_PER_ML * level_pct / 100.0 / rate_dpm
                time_remaining_str = (f"~{int(mins//60)}h {int(mins%60)}m"
                                      if mins > 60 else f"~{int(mins)} min to empty")
            else:
                time_remaining_str = "Calculating..."

        # ── Anomaly
        if rate_dpm > FREE_FLOW_THRESHOLD_DPM:
            an_status, an_color = "ANOMALY: FREE-FLOW",    (0,0,255)
        elif counter.drop_count > 0 and (time_s - counter.time_of_last_drop()) > FLOW_STOP_SECONDS:
            an_status, an_color = "ANOMALY: FLOW STOPPED", (0,100,255)
        else:
            an_status, an_color = "Normal",                 (0,150,0)

        csv_wr.writerow([f"{time_s:.2f}", counter.drop_count, f"{rate_dpm:.2f}",
                         f"{det['confidence']:.3f}", "ON" if det['has_drop'] else "OFF",
                         f"{counter.high:.2f}", f"{counter.low:.2f}",
                         level_pct, pred_lbl, f"{lev_conf:.3f}",
                         alert_msg, an_status, f"{display_fps:.1f}"])

        # ── Visualise
        vis   = apply_heatmap_to_roi(frame, heatmap_np, roi)
        if det['has_drop']:
            gy, gx = det['position']
            cx = roi_x + int(gx * roi_w / GRID_SIZE)
            cy = roi_y + int(gy * roi_h / GRID_SIZE)
            cv2.circle(vis,(cx,cy),22,(0,255,0),3)
            cv2.circle(vis,(cx,cy), 7,(0,0,255),-1)
            cv2.putText(vis,f"DROP {det['confidence']:.2f}",
                        (cx+28,cy+6),FONT,0.6,(0,255,0),2)
        cv2.putText(vis,f"FPS:{display_fps:.1f}",(W-130,30),FONT,0.75,(0,255,255),2)

        panel = draw_info_panel(
            H, counter.drop_count, rate_dpm,
            conf_history, det['has_drop'], counter.high, counter.low,
            level_pct, pred_lbl, lev_conf, time_remaining_str,
            alert_msg, alert_color, an_status, an_color, display_fps,
        )

        combined = np.hstack((panel, vis))
        out_vid.write(combined)
        cv2.imshow(WIN, combined)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'):
            print("\n[INFO] Quit."); break
        elif key == ord('r'):
            print("\n[INFO] Re-drawing ROI...")
            roi = select_roi(frame, WIN)
            roi_x,roi_y,roi_w,roi_h = roi
        elif key in (ord('+'), ord('=')):
            counter.high = min(round(counter.high + THRESHOLD_STEP, 2), 0.99)
            print(f"[TUNE] HIGH → {counter.high:.2f}")
        elif key == ord('-'):
            counter.high = max(round(counter.high - THRESHOLD_STEP, 2),
                               counter.low + THRESHOLD_STEP)
            print(f"[TUNE] HIGH → {counter.high:.2f}")
        elif key == ord(']'):
            counter.low = min(round(counter.low + THRESHOLD_STEP, 2),
                              counter.high - THRESHOLD_STEP)
            print(f"[TUNE] LOW  → {counter.low:.2f}")
        elif key == ord('['):
            counter.low = max(round(counter.low - THRESHOLD_STEP, 2), 0.01)
            print(f"[TUNE] LOW  → {counter.low:.2f}")

    cap.release(); out_vid.release(); csv_fh.close()
    cv2.destroyAllWindows()
    print(f"\n{'='*55}")
    print(f"  Frames : {frame_idx}  |  Drops: {counter.drop_count}")
    print(f"  Final HIGH={counter.high:.2f}  LOW={counter.low:.2f}")
    print(f"  Output : {out_path}")
    print(f"  CSV    : {output_csv}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    combined_heatmap_iv_monitor()
