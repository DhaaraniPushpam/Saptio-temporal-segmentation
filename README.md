# 21MIA1052_IVA_LAB_4
Spatio- Temporal segmentation
## Objective
This project aims to perform video processing tasks such as frame extraction, segmentation, and object tracking. The video used in this case study is "Donut (15-Second Ad).mp4."

## Methodology
1. **Frame Extraction**: Extract individual frames from the video.
2. **Segmentation**: Perform color-based segmentation to isolate specific elements in each frame.
3. **Object Tracking**: Track movement in the video based on detected objects.

## Algorithm / Pseudocode
### Frame Extraction
- Open the video file.
- Loop through the video frames and save each one as an image.
```python
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(output_path, frame)
cap.release()
```

### Segmentation
- Convert frames to HSV color space.
- Apply Gaussian blur and mask the color range for segmentation.
```python
hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_frame, lower_color, upper_color)
```

### Object Tracking
- Detect objects and track centroids across frames.
- Draw lines to visualize the tracking.
```python
for i, centroid in enumerate(centroids):
    cv2.line(frame, previous_centroids[i], centroid, (255, 0, 0), 2)
```

## Inference
- **Frame Extraction**: Successfully extracted 360 frames from the video.
- **Segmentation**: Processed 360 frames, detecting objects based on HSV color space.
- **Object Tracking**: Tracked objects throughout the video, and results were displayed in real-time.

## Output
- Extracted frames are saved in the `output/extracted_frames` directory.
- Processed video showing segmentation and object tracking is displayed during runtime.

## Setup
Install the required libraries:
```bash
pip install opencv-python numpy matplotlib
```

## Usage
1. Place the video in the `data/` folder.
2. Run the notebook or Python scripts:
   ```bash
   jupyter notebook src/main.ipynb
   ```
3. The extracted frames will be saved in the `output/` folder.
