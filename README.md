# IV Fluid Monitoring System

A comprehensive computer vision system for monitoring intravenous (IV) fluid administration, featuring both drop counting and fluid level detection capabilities. This system implements multiple detection approaches for robust IV fluid monitoring in clinical and research settings.

## Publication

This work has been published in **IEEE Access**:

> **An Integrated Dual-Module Computer Vision System for IV Drip Rate and Fluid Level Monitoring**
> IEEE Access, Volume 14
> DOI: [10.1109/ACCESS.2026.3687474](https://doi.org/10.1109/ACCESS.2026.3687474)
> Paper link: https://ieeexplore.ieee.org/abstract/document/11494667

If you use this work in your research, please cite the paper above.

### Abstract

Manual monitoring of intravenous (IV) therapy is labor-intensive and prone to human error, creating risks to patient safety. Prior computer-vision approaches have tackled drip rate estimation and fluid level monitoring separately, limiting their real-world clinical applicability. This work presents and validates a dual-module computer vision framework for a practical, automated IV monitoring system that addresses both tasks simultaneously.

For drip rate estimation, the paper conducts a comparative study of two deep learning paradigms on a custom dataset of 3,458 images, evaluated on a strictly held-out test set: a YOLOv8 object detector achieving mAP@0.50 of 95.34%, and a heatmap-based point-event regression network (ResNet-18 encoder with a convolutional decoder) achieving 88.44% frame-wise accuracy and 95.16% recall. A bespoke CNN handles fluid level classification, reaching 95.26% test accuracy on a public benchmark dataset.

A proof-of-concept prototype integrates both modules into a unified, automated video-based application, validated on pre-recorded clinical footage. On an end-to-end counting evaluation spanning 7 videos with drip rates from 17–94 dpm, the heatmap-based counter achieved 91.0% mean counting accuracy versus 77.5% for the YOLOv8-based counter, with the heatmap approach showing markedly greater robustness at higher infusion rates (n=2 videos at ≥47 dpm). The study establishes a validated framework for holistic IV infusion monitoring aimed at improving patient safety and reducing the burden on clinical staff.

## Project Overview

This system provides real-time monitoring of IV fluid administration through three main detection approaches:

1. **YOLO-Based Drop Detection** - Object detection and tracking for individual drops
2. **Heatmap-Based Drop Detection** - Regression-based approach using heatmap prediction
3. **CNN-Based Fluid Level Detection** - Classification of remaining fluid levels
4. **Integrated Monitoring System** - Combined drop detection and level monitoring

## YOLO-Based Drop Detection Approach

The YOLO (You Only Look Once) approach uses state-of-the-art object detection to identify and track individual IV fluid drops in real-time video streams.

### Architecture

- **Model**: YOLOv8 Nano (YOLOv8n) - optimized for speed and accuracy
- **Input**: Video frames at 640x640 resolution
- **Output**: Bounding boxes with track IDs for detected drops
- **Tracking**: Persistent object tracking across frames using YOLOv8's built-in tracker

### Methodology

The YOLO model is trained to predict bounding boxes around IV fluid drops. Each detected drop is assigned a unique track ID that persists across frames, enabling accurate drop counting. The system validates drops by:

- Minimum track duration (3 frames)
- Minimum downward travel distance (10 pixels)
- Temporal filtering to avoid false positives

### Performance

- **mAP@0.50**: 95.34% (Mean Average Precision at IoU threshold 0.5, evaluated on a held-out test set)
- **Real-time Processing**: Capable of processing video streams at 30+ FPS
- **Robustness**: Handles various lighting conditions and backgrounds

### Output

The system provides real-time visualization with bounding boxes, track IDs, and statistics including:
- Total drop count
- Drip rate (drops per minute)
- Time until next rate sample

### Training

The YOLO model was trained on 3,458 annotated images with bounding box labels in YOLO format. The dataset was split into 80% training and 20% validation sets. Training was performed using transfer learning from a pre-trained YOLOv8n model with the following hyperparameters:

- **Epochs**: 100 (with early stopping patience of 15)
- **Image Size**: 640x640 pixels
- **Batch Size**: 16
- **Learning Rate**: Adaptive with cosine annealing

## Heatmap-Based Drop Detection Approach

The heatmap approach uses a regression-based deep learning model to predict 2D heatmaps indicating drop locations, providing an alternative detection paradigm to bounding box-based methods.

### Architecture

- **Backbone**: ResNet18 pre-trained on ImageNet
- **Head**: Custom convolutional layers for heatmap prediction
- **Input**: 416x416 pixel images
- **Output**: 26x26 heatmap grid with Gaussian peaks at drop centers
- **Loss Function**: Mean Squared Error (MSE) between predicted and ground-truth heatmaps

### Methodology

The model learns to predict a 2D heatmap where each drop location is represented as a 2D Gaussian peak. The heatmap provides both detection and localization information. Drop counting is performed using rising edge detection on the maximum heatmap values:

- Detection threshold: 0.2 (configurable)
- Cooldown period: 5 frames to prevent double-counting
- Rising edge detection: Counts when heatmap intensity transitions from below to above threshold

### Performance

- **Frame-wise Accuracy**: 88.44% (evaluated on a held-out test set)
- **Recall**: 95.16%
- **Real-time Processing**: Efficient inference suitable for video streams
- **Advantages**: Provides spatial probability distribution, useful for uncertainty estimation

### Output

The system visualizes the predicted heatmap overlaid on the input image, showing the model's confidence in drop locations.

### Training

The heatmap model was trained on 3,458 annotated images with corresponding 26x26 NumPy heatmap arrays. Each heatmap contains a 2D Gaussian peak centered at the drop location. Training configuration:

- **Epochs**: 100 (with early stopping)
- **Batch Size**: 32
- **Learning Rate**: 0.001 with weight decay regularization
- **Data Split**: 80% training, 20% validation

## CNN-Based Fluid Level Monitoring

The fluid level detection system uses a convolutional neural network to classify the remaining fluid level in IV bags, providing critical monitoring capabilities for clinical applications.

### Architecture

- **Type**: Convolutional Neural Network (CNN)
- **Input**: 32x32 pixel images (preprocessed with negative filtering)
- **Output**: 4-class classification (0%, 50%, 80%, 100%)
- **Architecture**: Sequential model with Conv2D, MaxPooling2D, and Dense layers
- **Preprocessing**: Negative filtering to emphasize fluid transparency

### Methodology

The CNN model classifies the fluid level by analyzing the IV bag region. The negative filtering preprocessing step inverts the image colors, which helps emphasize the transparent fluid against various backgrounds. The model outputs probability distributions over four discrete level classes.

### Features

- Real-time level monitoring with sub-second inference
- Alert system for low fluid levels (≤50%)
- Visual dashboard with 2x2 processing grid
- Robust to various lighting conditions and backgrounds

### Output

The system provides a comprehensive dashboard showing:
- Original frame
- Preprocessed image (negative filtered)
- Level classification with confidence
- Alert status (Normal, Low, or Empty)

### Training

The level detection model was trained using a dataset from Mendeley Data with the following configuration:

- **Preprocessing**: Negative filtering and 32x32 resizing
- **Architecture**: CNN with dropout for regularization
- **Training**: Early stopping and model checkpointing
- **Classes**: 4 distinct fluid levels (0%, 50%, 80%, 100%)

## Integrated Monitoring System

The integrated system combines both drop detection (YOLO-based) and fluid level monitoring (CNN-based) into a unified real-time monitoring solution with advanced features.

### System Components

1. **YOLOv8 Drop Detection & Tracking**: Real-time drop counting and drip rate calculation
2. **CNN Fluid Level Classification**: Continuous level monitoring with alerts
3. **Time Remaining Estimation**: Calculates estimated time until bag is empty
4. **Anomaly Detection**: Rule-based detection of free-flow and flow stoppage

### Features

- **Unified Dashboard**: Side-by-side display of video feed and comprehensive statistics
- **Time Remaining Calculation**: Estimates time to empty based on current drip rate and fluid level
- **Anomaly Detection**:
  - Free-flow detection (drip rate > 200 dpm)
  - Flow stoppage detection (no drops for > 60 seconds)
- **CSV Logging**: Comprehensive logging of all metrics including:
  - Timestamp
  - Drop count
  - Drip rate (drops per minute)
  - Fluid level percentage
  - Level class
  - Alert status
  - Anomaly status

### Configuration

The integrated system allows configuration of IV setup parameters:

```python
TOTAL_BAG_VOLUME_ML = 500      # Total volume of IV bag
DROP_FACTOR_GTT_PER_ML = 15    # Drops per mL (tubing specific)
FREE_FLOW_THRESHOLD_DPM = 200  # Free-flow alert threshold
FLOW_STOP_SECONDS = 60         # Flow stoppage detection threshold
```

### Output

The integrated system provides a comprehensive monitoring interface combining:
- Live video with drop tracking visualization
- Real-time statistics panel showing:
  - Drop detection metrics (total count, current rate)
  - Level monitoring (percentage, confidence, time remaining)
  - Alert status (Normal, Low, Empty)
  - Anomaly status (Normal, Free-flow, Flow stopped)

The system processes video input and generates both a processed video output and a CSV log file for analysis.

### End-to-End Counting Evaluation

The integrated prototype was validated on pre-recorded clinical video footage spanning a 7-video set with drip rates ranging from 17 to 94 drops per minute (dpm):

- **Heatmap-based counter**: 91.0% mean counting accuracy
- **YOLOv8-based counter**: 77.5% mean counting accuracy
- **Key finding**: The heatmap-based counter showed markedly greater robustness at elevated infusion rates (proof-of-concept evaluation; n=2 videos at ≥47 dpm), making it the preferred counting approach for higher-flow scenarios.

## Project Structure

```
iv-fluids-level-monitor-and-drop-count/
├── models/
│   ├── drop_detector_yolo.pt          # YOLO model for drop detection
│   ├── drop_detector_heatmap.pth      # Heatmap model for drop detection
│   └── iv-fluids-level-detection-model.h5  # CNN model for level detection
├── notebooks/
│   ├── iv-fluids-drop-detection-yolov8.ipynb    # YOLO training notebook
│   ├── iv-fluids-drop-monitor-heatmap.ipynb     # Heatmap training notebook
│   └── iv-fluids-level-monitor.ipynb            # Level detection training notebook
├── scripts/
│   ├── drop_count_yolo.py            # YOLO-based drop counting script
│   ├── drop_count_heatmap.py         # Heatmap-based drop counting script
│   ├── level_alert_main.py           # Level detection script
│   └── combined.py                   # Integrated monitoring system
├── outputs/                           # Generated outputs (videos, CSV logs)
└── README.md                          # This file
```

## Installation & Setup

### Prerequisites

```bash
pip install opencv-python
pip install ultralytics
pip install torch torchvision
pip install tensorflow
pip install pillow
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install scikit-learn
```

## Usage

### YOLO-Based Drop Counting

```bash
python scripts/drop_count_yolo.py
```

**Configuration Options:**
- `WINDOW_DURATION`: Time window for drip rate calculation (default: 15 seconds)
- `RECHECK_INTERVAL`: Interval between rate samples (default: 30 seconds)
- `min_track_duration`: Minimum frames for valid drop detection (default: 3)
- `min_y_travel`: Minimum downward travel for valid drop (default: 10 pixels)

### Heatmap-Based Drop Counting

```bash
python scripts/drop_count_heatmap.py
```

**Configuration Options:**
- `detection_threshold`: Heatmap intensity threshold (default: 0.2)
- `cooldown_frames`: Frames between valid detections (default: 5)

### Fluid Level Detection

```bash
python scripts/level_alert_main.py
```

**Configuration Options:**
- `ALERT_THRESHOLD_PCT`: Alert threshold percentage (default: 50%)
- `MODEL_IMG_SIZE`: Input image size for model (default: 32x32)

### Integrated Monitoring System

```bash
python scripts/combined.py
```

The integrated system combines both drop detection and level monitoring with additional features for time estimation and anomaly detection.

## Performance Metrics Summary

### YOLO Drop Detection
- **mAP@0.50**: 96.1%
- **Real-time Performance**: 30+ FPS
- **Tracking Accuracy**: High precision with persistent track IDs

### Heatmap Drop Detection
- **Frame-wise Accuracy**: 88.58%
- **Spatial Localization**: Provides probability distribution over image
- **Real-time Performance**: Efficient inference suitable for video streams

### Level Detection
- **Classification Accuracy**: High accuracy on 4-class level classification
- **Real-time Inference**: Sub-second processing time
- **Robustness**: Handles various lighting conditions and backgrounds

## Configuration Parameters

### Drop Counting Parameters

```python
WINDOW_DURATION = 15    # seconds for drip rate calculation
RECHECK_INTERVAL = 30   # seconds between rate samples
min_track_duration = 3  # minimum frames for valid drop
min_y_travel = 10       # minimum downward travel (pixels)
```

### Level Detection Parameters

```python
ALERT_THRESHOLD_PCT = 50  # alert when ≤ 50%
MODEL_IMG_SIZE = 32       # input image size
CLASS_LABELS = ['sal_data_100', 'sal_data_80', 'sal_data_50', 'sal_data_empty']
```

### Integrated System Parameters

```python
TOTAL_BAG_VOLUME_ML = 500      # Total volume of IV bag
DROP_FACTOR_GTT_PER_ML = 15    # Drops per mL
FREE_FLOW_THRESHOLD_DPM = 200  # Free-flow alert threshold
FLOW_STOP_SECONDS = 60         # Flow stoppage detection threshold
```

## Training

### YOLO Drop Detection Model

The YOLO model was trained using the notebook `notebooks/iv-fluids-drop-detection-yolov8.ipynb` with:

- **Dataset**: 3,458 annotated images with bounding box labels in YOLO format
- **Model**: YOLOv8 Nano (YOLOv8n) with transfer learning
- **Training Configuration**:
  - Epochs: 100 (early stopping patience: 15)
  - Image size: 640x640 pixels
  - Batch size: 16
- **Performance**: Achieved 96.1% mAP@0.50 on validation set

### Heatmap Drop Detection Model

The heatmap model was trained using the notebook `notebooks/iv-fluids-drop-monitor-heatmap.ipynb` with:

- **Dataset**: 3,458 annotated images with corresponding 26x26 heatmap arrays
- **Architecture**: ResNet18 backbone with custom heatmap prediction head
- **Training Configuration**:
  - Epochs: 100 (with early stopping)
  - Batch size: 32
  - Learning rate: 0.001 with weight decay
  - Input size: 416x416 pixels
  - Output size: 26x26 heatmap grid
- **Performance**: Achieved 88.58% frame-wise accuracy on validation set

### Level Detection Model

The level detection model was trained using the notebook `notebooks/iv-fluids-level-monitor.ipynb` with:

- **Dataset**: IV fluids level dataset from Mendeley Data
- **Preprocessing**: Negative filtering and 32x32 resizing
- **Architecture**: CNN with dropout for regularization
- **Training**: Early stopping and model checkpointing
- **Classes**: 4 distinct fluid levels (0%, 50%, 80%, 100%)

## Dataset

### Drop Detection Dataset

The drop detection models (both YOLO and heatmap) were trained on a dataset of **3,458 annotated images** with dual annotation formats:

- **Bounding Box Labels**: YOLO-format .txt files for object detection training
- **Heatmap Targets**: 26x26 NumPy arrays with 2D Gaussian peaks for regression training

**Dataset Details:**
- **Title**: Image Dataset for Intravenous (IV) Drop Detection with Dual Annotations (Bounding Box and Heatmap)
- **Repository**: IEEE DataPort
- **DOI**: [10.21227/8g5q-zk62](https://dx.doi.org/10.21227/8g5q-zk62)
- **Link**: https://ieee-dataport.org/documents/image-dataset-intravenous-iv-drop-detection-dual-annotations-bounding-box-and-heatmap

**Dataset Citation:**
Nishant Vasantkumar Hegde, Saksham Gupta, Atul Kumar Mishra, Pratiba D, Ramakanthkumar P, Sreelakshmi K, Shankar T, "Image Dataset for Intravenous (IV) Drop Detection with Dual Annotations (Bounding Box and Heatmap)", IEEE Dataport, December 26, 2025, doi:10.21227/8g5q-zk62

### Level Detection Dataset

The level detection model was trained using datasets from Mendeley Data:
- DOI: 10.17632/9mcj3rvvxb.1
- DOI: 10.17632/n8k2zfr6xm.2

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is intended for academic and research purposes only.

## Acknowledgments

- **Publication**: An Integrated Dual-Module Computer Vision System for IV Drip Rate and Fluid Level Monitoring, IEEE Access, Vol. 14 (DOI: [10.1109/ACCESS.2026.3687474](https://doi.org/10.1109/ACCESS.2026.3687474))
- **Drop Detection Dataset**: Image Dataset for Intravenous (IV) Drop Detection with Dual Annotations (Bounding Box and Heatmap) - IEEE DataPort (DOI: 10.21227/8g5q-zk62)
- **Level Detection Dataset**: Mendeley Data (DOI: 10.17632/9mcj3rvvxb.1, 10.17632/n8k2zfr6xm.2)

## Disclaimer

This system is designed for research and educational purposes. For clinical use, additional validation and regulatory compliance may be required.
