# 21MIA1052_IVA_LAB_4
Spatio- Temporal segmentation

## Objective
The purpose of this project is to process a given video file to extract valuable insights. The tasks involve extracting frames from the video, performing spatio-temporal segmentation, detecting scene cuts, and visualizing the results. The project aims to detect abrupt and gradual scene transitions, highlight frames where transitions occur, and segment the foreground and background of the video frames.

## Methodology
This project follows the below steps:

### Step 1: Load Video
- Load the provided video file into the Python environment for processing.

### Step 2: Frame Extraction
- Individual frames are extracted from the video sequence for further analysis.

### Step 3: Spatio-Temporal Segmentation
- Perform segmentation of each frame using techniques like color thresholding or edge detection.
- Objects are tracked across consecutive frames to observe changes in motion and shape.
- Foreground and background are separated by identifying consistent regions over time.

### Step 4: Scene Cut Detection
- Abrupt scene changes (hard cuts) are detected by comparing pixel or histogram differences between consecutive frames.
- Gradual scene transitions (soft cuts) are detected by analyzing frame intensity changes over time.

### Step 5: Mark Scene Cuts
- Frames with detected scene cuts are highlighted.
- A summary is created to display the boundaries of the detected scene cuts.

### Step 6: Result Visualization
- Visualize frames where scene cuts are detected.
- Display segmentation results on selected frames.


## Algorithm
### **1. Video Loading and Frame Extraction**
- Open the video using OpenCV and read each frame in a loop.
- Save each frame to an output folder, ensuring proper naming conventions (e.g., `frame_0001.jpg`).
- Count the number of frames processed.

### **2. Spatio-Temporal Segmentation**
- Convert each frame to the HSV color space to enhance color segmentation.
- Apply Gaussian blurring to reduce noise and prepare the frame for color thresholding.
- Perform color thresholding using predefined upper and lower color limits.
- Use morphological operations (closing and opening) to refine the mask and improve segmentation.
- Extract the segmented objects from each frame using bitwise operations.
- Optionally apply Sobel edge detection for further object boundary refinement.

### **3. Scene Cut Detection**
- For each consecutive pair of frames, compute differences using pixel-based comparison or histogram differences.
- Detect abrupt (hard) cuts when a significant change between consecutive frames is observed.
- For gradual (soft) cuts, analyze intensity changes over time by comparing multiple consecutive frames.

### **4. Scene Cut Marking and Result Visualization**
- Mark the frames where abrupt or gradual scene cuts are detected.
- Save the color-segmented and edge-detected images.
- Display the results for both scene cut detection and segmentation using OpenCV’s GUI.


## Pseudo-code
1. Initialize video capture using OpenCV:
   a. Load the video file.
   ```python
   import cv2
   import os

   def extract_frames(video_path, output_folder='output_frames_lab4'):
       if not os.path.exists(output_folder):
           os.makedirs(output_folder)
       video_capture = cv2.VideoCapture(video_path)
       if not video_capture.isOpened():
           print("Error: Could not open video.")
           return
   ```

2. Frame Extraction Loop:
   a. For each frame in the video:
      i. Read the frame.
      ii. Save the frame as a JPEG image.
   ```python
   frame_count = 0
   while True:
       success, frame = video_capture.read()
       if not success:
           break
       frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
       cv2.imwrite(frame_filename, frame)
       frame_count += 1
       print(f"Saved {frame_filename}")
   video_capture.release()
   print(f"All {frame_count} frames have been extracted and saved.")
   return output_folder
   ```

3. Spatio-Temporal Segmentation Loop:
   a. For each extracted frame:
      i. Convert the frame to HSV color space.
      ii. Apply Gaussian blurring to the frame.
      iii. Perform color thresholding using predefined HSV limits.
      iv. Refine the threshold mask using morphological operations (closing and opening).
      v. Extract the segmented image by applying the mask to the original frame.
      vi. Apply Sobel edge detection on the grayscale version of the frame for edge enhancement.
      vii. Save both the color-segmented and Sobel edge-detected images.
   ```python
   import numpy as np
   
   lower_color = np.array([30, 100, 100])
   upper_color = np.array([90, 255, 255])

   cap = cv2.VideoCapture('video.mp4')
   while cap.isOpened():
       ret, frame = cap.read()
       if not ret:
           break

       # Convert to HSV color space
       hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

       # Apply Gaussian blur
       blurred_hsv_frame = cv2.GaussianBlur(hsv_frame, (11, 11), 0)

       # Color thresholding
       mask = cv2.inRange(blurred_hsv_frame, lower_color, upper_color)

       # Morphological operations
       kernel = np.ones((5, 5), np.uint8)
       mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
       mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

       # Extract segmented image
       color_segmented_frame = cv2.bitwise_and(frame, frame, mask=mask)

       # Sobel edge detection
       gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       sobel_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
       sobel_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
       sobel_edges = cv2.magnitude(sobel_x, sobel_y)
       sobel_edges = cv2.convertScaleAbs(sobel_edges)

       # Save results
       cv2.imwrite(f'color_segmented_{frame_count:04d}.jpg', color_segmented_frame)
       cv2.imwrite(f'sobel_edges_{frame_count:04d}.jpg', sobel_edges)

       frame_count += 1
   ```

4. Scene Cut Detection:
   a. For each consecutive pair of frames:
      i. Compare pixel-wise differences or histogram differences.
      ii. If the difference exceeds a threshold, mark a hard cut.
      iii. For gradual cuts, compute frame intensity changes over several frames.
   ```python
   prev_frame = None
   cut_threshold = 30  # example threshold for pixel difference
   while cap.isOpened():
       ret, frame = cap.read()
       if not ret:
           break
       
       if prev_frame is not None:
           diff = cv2.absdiff(frame, prev_frame)
           non_zero_count = np.count_nonzero(diff)
           if non_zero_count > cut_threshold:
               print(f"Scene cut detected at frame {frame_count}")
       prev_frame = frame
       frame_count += 1
   ```

5. Mark Scene Cuts:
   a. Highlight frames where abrupt or gradual scene cuts are detected.
   ```python
   # Highlight or store cut-detected frames
   if non_zero_count > cut_threshold:
       cv2.imwrite(f'cut_frame_{frame_count:04d}.jpg', frame)
   ```

6. Visualize Results:
   a. Use OpenCV to display segmentation and edge detection results.
   ```python
   cv2.imshow('Color Segmentation', color_segmented_frame)
   cv2.imshow('Sobel Edge Detection', sobel_edges)
   if cv2.waitKey(30) & 0xFF == ord('q'):
       break
   cap.release()
   cv2.destroyAllWindows()
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
- You can download the extracted output frames from the video from this [Google Drive link](https://drive.google.com/drive/folders/1WLwdKBHwSGD9XkYP03YTPoJfzh9l8ja7?usp=sharing).



