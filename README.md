# OpenCV Image Processing Playground

This repository contains a Python script that serves as an image processing playground using the OpenCV library. The script captures video from a webcam and allows users to experiment with various image processing techniques through an interactive interface.

## Features

- **Real-time Video Capture**: Utilizes OpenCV to capture frames from a connected webcam in real-time.
- **Interactive Trackbars**: Adjustable trackbars for tweaking parameters such as Gaussian blur, threshold, Sobel kernel sizes, and Canny edge detector thresholds.
- **Image Processing Operations**: Implements custom image processing operations, including Sobel X and Y operators, Laplacian, and more.
- **User-Triggered Actions**: Responds to keyboard inputs for actions like capturing photos, starting/stopping video recording, extracting blue color, and more.
- **Matplotlib Integration**: Displays frames processed with Laplacian, Sobel X, and Sobel Y operators using Matplotlib.

https://github.com/argrabowski/image-processing-playground/assets/64287065/8ab31bc8-6f46-4792-8830-d326ac275d97

https://github.com/argrabowski/image-processing-playground/assets/64287065/bd8663e3-4500-48e4-8733-66c869be11e9

## Getting Started

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/argrabowski/image-processing-playground.git
   cd image-processing-playground
   ```

2. **Install Dependencies:**
   ```bash
   pip install opencv-python numpy matplotlib
   ```

3. **Run the Script:**
   ```bash
   python app.py
   ```

4. **Interact with the Interface:**
   Adjust trackbars, explore different image processing operations, and use keyboard shortcuts for various functionalities.

## Keyboard Shortcuts

- **'c'**: Capture a photo with optional flash effect.
- **'v'**: Start/stop video recording.
- **'e'**: Extract blue color from the video feed.
- **'r'**: Rotate the video frame by 10 degrees.
- **'t'**: Toggle thresholding.
- **'b'**: Toggle Gaussian blur.
- **'s'**: Display a sharpened version of the video frame.
- **'d'**: Apply Canny edge detection.
- **'4'**: Display original, Laplacian, Sobel X, and Sobel Y frames using Matplotlib.

## Requirements

- Python 3.10
- OpenCV
- NumPy
- Matplotlib
