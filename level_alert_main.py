import cv2
import numpy as np
import tensorflow as tf
from PIL import Image as pil_image
import PIL.ImageOps
import time
from datetime import datetime

"""
IV Fluid Level Detection using a CNN model with a dashboard interface.

Changes v2 (2025-07-26)
-----------------------
* **Implemented a side-by-side dashboard interface.**
* The left panel provides clear, at-a-glance status and alerts.
* The right panel shows the full image processing pipeline in a 2x2 grid:
  1. Original Input
  2. Resized Natural Image
  3. Preprocessed Negative Image
  4. Class Prediction Confidences
* Alert messages are now larger and color-coded for immediate recognition.
"""


# Optional Twilio support ----------------------------------------------------
ENABLE_SMS_ALERT = False  # <-- Set to True after filling the credentials below
TWILIO_ACCOUNT_SID = "YOUR_ACCOUNT_SID"
TWILIO_AUTH_TOKEN = "YOUR_AUTH_TOKEN"
TWILIO_FROM = "+1234567890"  # Twilio verified / purchased number
TWILIO_TO = "+1987654321"  # Destination number for alerts

if ENABLE_SMS_ALERT:
    try:
        from twilio.rest import Client
        _twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    except Exception as _e:
        print("❌ Twilio import failed – SMS alerts disabled:", _e)
        ENABLE_SMS_ALERT = False
        _twilio_client = None
else:
    _twilio_client = None
# ---------------------------------------------------------------------------

# ---------------- Configuration ----------------
MODEL_PATH = r'F:\\CCVR project\\final_working\\iv-fluids-level-detection.h5'
INPUT_PATH = r'F:\\CCVR project\\final_working\\50_3.jpg'  # or video e.g. r'F:\CCVR\drip3.mp4'

MODEL_IMG_SIZE = 32
CLASS_LABELS = ['sal_data_100', 'sal_data_80', 'sal_data_50', 'sal_data_empty']
LEVEL_PERCENT_MAP = {
    'sal_data_100': 100,
    'sal_data_80': 80,
    'sal_data_50': 50,
    'sal_data_empty': 0,
}
ALERT_THRESHOLD_PCT = 50  # Alert when ≤ 50 %

# --- Interface Layout ---
VISUAL_PANEL_WIDTH = 320
VISUAL_PANEL_HEIGHT = 240
INFO_PANEL_WIDTH = 400
FONT = cv2.FONT_HERSHEY_SIMPLEX
# ------------------------------------------------

# ====================== Utility Functions =====================

def _send_sms_alert(level_pct: int):
    if not ENABLE_SMS_ALERT or _twilio_client is None:
        return
    try:
        body = (f"⚠️ Saline level critical: {level_pct}% remaining. "
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        _twilio_client.messages.create(body=body, from_=TWILIO_FROM, to=TWILIO_TO)
        print("📤 SMS alert sent!")
    except Exception as e:
        print("❌ Failed to send SMS alert:", e)


def create_visual_panel(img, title, size=(VISUAL_PANEL_WIDTH, VISUAL_PANEL_HEIGHT)):
    """Creates a single visual panel with a title header for the 2x2 grid."""
    panel = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    header = np.full((30, size[0], 3), (40, 40, 40), dtype=np.uint8)
    cv2.putText(header, title, (10, 20), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack((header, panel))


def create_confidence_bars(probs, labels, size=(VISUAL_PANEL_WIDTH, VISUAL_PANEL_HEIGHT)):
    """Creates the confidence bar chart panel."""
    bar_panel = np.full((size[1], size[0], 3), (60, 60, 60), dtype=np.uint8)
    y0 = 40
    for i, (lbl, p) in enumerate(zip(labels, probs)):
        # Format label to be more readable, e.g., '100%' or 'Empty'
        level_str = lbl.replace('sal_data_', '') + '%'
        if level_str == 'empty%': level_str = 'Empty'
        
        text = f"{level_str}: {p:.2f}"
        cv2.putText(bar_panel, text, (10, y0 + i * 50), FONT, 0.7, (255, 255, 255), 2)
        
        # Draw the bar
        bar_w = int(p * (size[0] - 80)) # Make bar slightly shorter
        bar_color = (70, 180, 70) if p < 0.8 else (100, 255, 100)
        cv2.rectangle(bar_panel, (10, y0 + 10 + i * 50), (10 + bar_w, y0 + 35 + i * 50), bar_color, -1)

    return create_visual_panel(bar_panel, "4: Predictions", size)


def preprocess_frame(frame, target_size=(MODEL_IMG_SIZE, MODEL_IMG_SIZE)):
    """Prepares a frame for model inference and returns intermediate images for visualization."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = pil_image.fromarray(rgb)

    # 1. Resized natural image for display
    resized_natural_pil = pil_img.resize(target_size, pil_image.LANCZOS)
    resized_natural_cv = cv2.cvtColor(np.array(resized_natural_pil), cv2.COLOR_RGB2BGR)

    # 2. Inverted and resized image for model input
    inverted_pil = PIL.ImageOps.invert(pil_img)
    preprocessed_pil = inverted_pil.resize(target_size, pil_image.LANCZOS)
    preprocessed_cv = cv2.cvtColor(np.array(preprocessed_pil), cv2.COLOR_RGB2BGR)

    # 3. Final model input (normalized numpy array)
    model_input_arr = np.expand_dims(np.array(preprocessed_pil) / 255.0, axis=0)

    return {
        'model_input': model_input_arr,
        'resized_natural': resized_natural_cv,
        'preprocessed_negative': preprocessed_cv,
    }

# ============================== Main ==============================

def main():
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"✔️  Model '{MODEL_PATH}' loaded.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    is_video = INPUT_PATH.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

    if is_video:
        cap = cv2.VideoCapture(INPUT_PATH)
        if not cap.isOpened():
            print(f"❌ Error opening video: {INPUT_PATH}")
            return
        print("Processing video... Press 'q' to quit.")
    else:
        frame = cv2.imread(INPUT_PATH)
        if frame is None:
            print(f"❌ Error reading image: {INPUT_PATH}")
            return
        print("Processing image... Press 'q' to quit.")

    alert_active_sms = False # To prevent spamming SMS
    
    # Calculate total height of the 2x2 grid
    grid_total_height = 2 * (VISUAL_PANEL_HEIGHT + 30) # panel + header

    while True:
        if is_video:
            ret, frame = cap.read()
            if not ret:
                break
        
        # --- Core Logic ---
        processed = preprocess_frame(frame)
        probs = model.predict(processed['model_input'], verbose=0)[0]
        pred_idx = np.argmax(probs)
        pred_label = CLASS_LABELS[pred_idx]
        confidence = float(probs[pred_idx])
        level_pct = LEVEL_PERCENT_MAP.get(pred_label, 0)

        # --- Build Right-Side 2x2 Grid ---
        p1 = create_visual_panel(frame, "1: Original Input")
        p2 = create_visual_panel(processed['resized_natural'], "2: Resized (32x32)")
        p3 = create_visual_panel(processed['preprocessed_negative'], "3: Preprocessed (Negative)")
        p4 = create_confidence_bars(probs, CLASS_LABELS)
        
        top_row = np.hstack((p1, p2))
        bottom_row = np.hstack((p3, p4))
        visuals_grid = np.vstack((top_row, bottom_row))

        # --- Build Left-Side Info Panel ---
        info_panel = np.full((grid_total_height, INFO_PANEL_WIDTH, 3), (220, 220, 220), dtype=np.uint8)

        # 1. General Info
        cv2.putText(info_panel, "IV Fluid Monitor", (20, 50), FONT, 1.2, (0, 0, 0), 3)
        cv2.line(info_panel, (20, 70), (INFO_PANEL_WIDTH - 20, 70), (0, 0, 0), 1)
        cv2.putText(info_panel, f"Level: {level_pct}%", (30, 120), FONT, 1, (0, 0, 0), 2)
        cv2.putText(info_panel, f"Confidence: {confidence:.2f}", (30, 160), FONT, 1, (0, 0, 0), 2)

        # 2. Alert Status
        alert_msg = "Status: Normal"
        alert_color = (0, 150, 0) # Green
        if level_pct == 0:
            alert_msg = "HIGH ALERT: EMPTY!"
            alert_color = (0, 0, 255) # Red
        elif level_pct <= ALERT_THRESHOLD_PCT:
            alert_msg = "ALERT: 50% or LESS"
            alert_color = (0, 0, 255) # Red
        
        cv2.putText(info_panel, "Status:", (30, 250), FONT, 1, (0, 0, 0), 2)
        (w, h), _ = cv2.getTextSize(alert_msg, FONT, 1.2, 3)
        cv2.rectangle(info_panel, (30, 280), (INFO_PANEL_WIDTH - 30, 280 + h + 20), alert_color, -1)
        cv2.putText(info_panel, alert_msg, (40, 305), FONT, 1.0, (255, 255, 255), 3)
        
        # --- Handle SMS logic ---
        if level_pct <= ALERT_THRESHOLD_PCT:
            if not alert_active_sms:
                print(f"⚠️  Alert triggered: {alert_msg}")
                _send_sms_alert(level_pct)
                alert_active_sms = True
        else:
            alert_active_sms = False # Reset alert when level is back to normal

        # --- Combine all sections ---
        dashboard = np.hstack((info_panel, visuals_grid))

        cv2.imshow('Saline Level Detection Dashboard', dashboard)
        key = cv2.waitKey(1 if is_video else 0)
        if key == ord('q') or (not is_video and key != -1):
            break

    if is_video:
        cap.release()
    cv2.destroyAllWindows()
    print("Dashboard closed.")

if __name__ == '__main__':
    main()