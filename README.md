# 21MIA1052_IVA_LAB_4
Spatio- Temporal segmentation

## Objective
The objective of this project is to process video frames, apply color space transformations, and perform image analysis techniques such as color segmentation and edge detection using OpenCV. The workflow includes extracting frames from a video, converting them into the HSV color space, and applying Sobel edge detection for feature extraction.

## Methodology
This project follows the below steps:

1. **Frame Extraction**: Frames are extracted from a given video file and saved as individual images.
2. **Color Space Conversion**: The extracted frames are converted from the BGR to the HSV color space.
3. **Color Segmentation**: Specific color ranges in the frames are isolated using thresholding in the HSV color space.
4. **Edge Detection**: Sobel edge detection is applied to the frames to highlight edges and features within the images.
5. **Frame Processing**: Processed frames are saved, and optional visualization is done using OpenCV.

## Algorithm
1. **Frame Extraction**:
   - Open the video file.
   - Read frames in a loop.
   - Save each frame as a separate image file.
   
2. **Convert Frames to HSV Color Space**:
   - Load each extracted frame.
   - Convert the image from BGR to HSV format.
   - Save the converted frame.

3. **Color Segmentation**:
   - Define lower and upper bounds for the color of interest in HSV format.
   - Use these bounds to create a mask.
   - Apply the mask to segment the desired color from the image.

4. **Sobel Edge Detection**:
   - Convert the image to grayscale.
   - Apply the Sobel operator to compute the gradients in the x and y directions.
   - Combine the gradients to get the edge-detected image.
   - Save the edge-detected frames.

## Pseudo-code

```python
# Pseudo-code for frame extraction
def extract_frames(video_path):
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        cv2.imwrite(f"frame_{frame_count}.jpg", frame)
        frame_count += 1

# Pseudo-code for color segmentation
def color_segmentation(frame, lower_bound, upper_bound):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result

# Pseudo-code for Sobel edge detection
def sobel_edge_detection(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobel_x, sobel_y)
    return cv2.convertScaleAbs(edges)
```

## Dependencies
To run the code in this project, the following Python packages are required:

- `opencv-python`
- `numpy`
- `matplotlib`
- `scikit-image`

To install these dependencies, use the following command:

```bash
pip install -r requirements.txt
```

## Inference
- The project demonstrates the use of basic video processing and computer vision techniques.
- Color segmentation and edge detection provide insightful ways to analyze and understand the contents of each video frame.
- The use of HSV color space allows for better segmentation of colors as compared to the RGB color space.

## Output
- Extracted video frames saved as images.
- Color-segmented frames highlighting specific color ranges.
- Edge-detected frames using Sobel operators.

