import cv2
import numpy as np
import tensorflow as tf
from PIL import Image as pil_image
import PIL.ImageOps
import time

# --- Configuration ---
MODEL_PATH = 'F:\\CCVR project\\final_working\\iv-fluids-level-detection.h5' 
INPUT_PATH = 'F:\\CCVR project\\try-2.png' # or 'path/to/your/test_video.mp4'

# Model input size (must be 32x32 for the paper's model)
MODEL_IMG_SIZE = 32
CLASS_LABELS = ['sal_data_100', 'sal_data_50', 'sal_data_80', 'sal_data_empty']

# Visualization parameters
PANEL_WIDTH = 320
PANEL_HEIGHT = 240
FONT = cv2.FONT_HERSHEY_SIMPLEX
# --- End of Configuration ---

def create_panel(image, title, panel_size=(PANEL_WIDTH, PANEL_HEIGHT)):
    """Creates a standardized panel for the dashboard."""
    # Resize the image to fit the panel
    panel = cv2.resize(image, panel_size, interpolation=cv2.INTER_NEAREST)
    # Add a black bar at the top for the title
    header = np.zeros((30, panel_size[0], 3), dtype=np.uint8)
    cv2.putText(header, title, (10, 20), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    # Stack header and image
    return np.vstack((header, panel))

def create_confidence_bars(probabilities, labels, panel_size=(PANEL_WIDTH, PANEL_HEIGHT)):
    """Creates a bar chart panel for class confidences."""
    panel = np.zeros((panel_size[1], panel_size[0], 3), dtype=np.uint8)
    bar_start_y = 40
    for i, (label, prob) in enumerate(zip(labels, probabilities)):
        # Format label and probability text
        text = f"{label.replace('sal_data_', '')}: {prob:.2f}"
        cv2.putText(panel, text, (10, bar_start_y + i*50), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw the confidence bar
        bar_width = int(prob * (panel_size[0] - 20))
        cv2.rectangle(panel, (10, bar_start_y + 10 + i*50), (10 + bar_width, bar_start_y + 30 + i*50), (0, 255, 0), -1)

    return create_panel(panel, "4: Class Confidences")

def preprocess_frame(frame, target_size=(MODEL_IMG_SIZE, MODEL_IMG_SIZE)):
    """
    Performs all preprocessing and returns intermediate images for visualization.
    Returns a dictionary of images.
    """
    # Convert BGR (OpenCV) to RGB for PIL
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = pil_image.fromarray(rgb_frame)
    
    # Create natural resized version for visualization
    resized_natural_pil = pil_img.resize(target_size, pil_image.LANCZOS)
    resized_natural_cv = cv2.cvtColor(np.array(resized_natural_pil), cv2.COLOR_RGB2BGR)

    # 1. Apply negative filter
    inverted_img = PIL.ImageOps.invert(pil_img)
    
    # 2. Resize
    resized_inverted_pil = inverted_img.resize(target_size, pil_image.LANCZOS)
    resized_inverted_cv = cv2.cvtColor(np.array(resized_inverted_pil), cv2.COLOR_RGB2BGR)
    
    # 3. Rescale and expand for model
    img_array = np.array(resized_inverted_pil)
    img_array_rescaled = img_array / 255.0
    model_input = np.expand_dims(img_array_rescaled, axis=0)
    
    return {
        "model_input": model_input,
        "resized_natural": resized_natural_cv,
        "resized_inverted": resized_inverted_cv
    }

def main():
    """Main function to load model and run the visualization dashboard."""
    # Load the model
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"Model '{MODEL_PATH}' loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    is_video = INPUT_PATH.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

    if is_video:
        cap = cv2.VideoCapture(INPUT_PATH)
        if not cap.isOpened():
            print(f"Error opening video: {INPUT_PATH}")
            return
        print("Processing video... Press 'q' to quit.")
    else:
        frame = cv2.imread(INPUT_PATH)
        if frame is None:
            print(f"Error reading image: {INPUT_PATH}")
            return
        print("Processing image... Press 'q' to quit.")

    # Main loop
    prev_time = 0
    while True:
        if is_video:
            ret, frame = cap.read()
            if not ret:
                break
        
        # --- Core Logic ---
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Preprocess frame and get visualization stages
        processed_data = preprocess_frame(frame)
        
        # Make prediction
        prediction = model.predict(processed_data["model_input"], verbose=0)[0]
        confidence = np.max(prediction)
        pred_index = np.argmax(prediction)
        pred_label = CLASS_LABELS[pred_index]

        # --- Build Dashboard ---
        # Panel 1: Original Input
        panel1 = create_panel(frame, "1: Original Input")

        # Panel 2: Resized Natural Image
        panel2 = create_panel(processed_data["resized_natural"], "2: Resized Natural (32x32)")
        
        # Panel 3: Preprocessed Inverted Image (Model Input)
        panel3 = create_panel(processed_data["resized_inverted"], "3: Preprocessed Inverted (32x32)")
        
        # Panel 4: Confidence Bars
        panel4 = create_confidence_bars(prediction, CLASS_LABELS)

        # Assemble grid
        top_row = np.hstack((panel1, panel2))
        bottom_row = np.hstack((panel3, panel4))
        dashboard_grid = np.vstack((top_row, bottom_row))

        # Create status bar
        status_bar = np.zeros((40, dashboard_grid.shape[1], 3), dtype=np.uint8)
        level_text = f"Level: {pred_label.replace('sal_data_', '')}"
        conf_text = f"Confidence: {confidence:.2f}"
        fps_text = f"FPS: {fps:.1f}" if is_video else ""

        cv2.putText(status_bar, f"Prediction: {level_text} | {conf_text}", (10, 25), FONT, 0.7, (0, 255, 0), 2)
        if is_video:
            cv2.putText(status_bar, fps_text, (dashboard_grid.shape[1] - 120, 25), FONT, 0.7, (0, 255, 255), 2)
        
        # Combine grid and status bar
        final_dashboard = np.vstack((dashboard_grid, status_bar))

        cv2.imshow('Saline Level Detection Dashboard', final_dashboard)

        key = cv2.waitKey(1 if is_video else 0)
        if key == ord('q'):
            break
        
        if not is_video: # If it's an image, break after first loop
            break
            
    # Cleanup
    if is_video:
        cap.release()
    cv2.destroyAllWindows()
    print("Dashboard closed.")

if __name__ == '__main__':
    main()