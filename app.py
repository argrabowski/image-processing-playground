import cv2 as cv
import datetime
import numpy as np
from matplotlib import pyplot as plt

# Callback function for sigmaX trackbar
def updateSigmaX(s):
    global sigmaX
    sigmaX = s

# Callback function for sigmaY trackbar
def updateSigmaY(s):
    global sigmaY
    sigmaY = s

# Callback function for threshold trackbar
def updateThresholdValue(t):
    global thresholdValue
    thresholdValue = t

# Callback function for Sobel X kernel size trackbar
def updatesobelKSizeX(k):
    global sobelKSizeX
    sobelKSizeX = k

# Callback function for Sobel Y kernel size trackbar
def updatesobelKSizeY(k):
    global sobelKSizeY
    sobelKSizeY = k

# Callback function for Canny threshold 1 trackbar
def updateCannyThreshold1(c):
    global cannyThreshold1
    cannyThreshold1 = c

# Callback function for Canny threshold 2 trackbar
def updateCannyThreshold2(c):
    global cannyThreshold2
    cannyThreshold2 = c

# Sobel operation for horizontal changes
def customSobelX(frame):
    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    return cv.filter2D(frame, -1, kernel)

# Sobel operation for vertical changes
def customSobelY(frame):
    kernel = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
    return cv.filter2D(frame, -1, kernel)

# Laplacian operation for all directions
def customLaplacian(frame):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    return cv.filter2D(frame, -1, kernel)

def main():
    # Create window
    cv.namedWindow("Camera")

    # Create video capture
    cap = cv.VideoCapture(0)
    assert cap.isOpened(), "Cannot open camera"

    # Load OpenCV logo
    logo = cv.imread('opencv.png')
    assert logo is not None, "Cannot read file"

    # Resize OpenCV logo
    logo = cv.resize(logo, None, fx=0.2, fy=0.2)
    logoHeight, logoWidth = logo.shape[:2]

    # Initialize control variables
    isRecording = False
    out = None
    rotation = 0
    threshold = False
    blur = False

    # Initialize trackbar variables
    global sigmaX
    global sigmaY
    global thresholdValue
    global sobelKSizeX
    global sobelKSizeY
    global cannyThreshold1
    global cannyThreshold2
    sigmaX = 10
    sigmaY = 20
    thresholdValue = 127
    sobelKSizeX = 5
    sobelKSizeY = 5
    cannyThreshold1 = 50
    cannyThreshold2 = 50

    # Define blue color range in HSV
    lowerBlue = np.array([110, 50, 50])
    upperBlue = np.array([130, 255, 255])

    # Create trackbars for Gaussian blur
    cv.createTrackbar("Sigma X", "Camera", sigmaX, 30, updateSigmaX)
    cv.setTrackbarMin("Sigma X", "Camera", 5)
    cv.createTrackbar("Sigma Y", "Camera", sigmaY, 30, updateSigmaY)
    cv.setTrackbarMin("Sigma Y", "Camera", 5)

    # Create trackbar for threshold value
    cv.createTrackbar("Threshold", "Camera", thresholdValue, 255, updateThresholdValue)

    # Create trackbars for Sobel kernel sizes
    cv.createTrackbar("Sobel X", "Camera", sobelKSizeX, 30, updatesobelKSizeX)
    cv.setTrackbarMin("Sobel X", "Camera", 5)
    cv.createTrackbar("Sobel Y", "Camera", sobelKSizeY, 30, updatesobelKSizeY)
    cv.setTrackbarMin("Sobel Y", "Camera", 5)

    # Create trackbars for Canny thresholds
    cv.createTrackbar("Canny Threshold 1", "Camera", cannyThreshold1, 5000, updateCannyThreshold1)
    cv.createTrackbar("Canny Threshold 2", "Camera", cannyThreshold2, 5000, updateCannyThreshold2)

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        assert ret, "Cannot read frame"
        frameHeight, frameWidth = frame.shape[:2]

        # Update frame with rotation
        matrix = cv.getRotationMatrix2D((frameWidth / 2, frameHeight / 2), rotation, 1)
        frame = cv.warpAffine(frame, matrix, (frameWidth, frameHeight))

        # Apply Gaussian blur
        if blur: frame = cv.GaussianBlur(frame, (0, 0), sigmaX, sigmaY)

        # Update frame with blended logo
        logoROI = frame[:logoHeight, :logoWidth]
        frame[:logoHeight, :logoWidth] = cv.addWeighted(logoROI, 0.5, logo, 0.5, 0)

        # Apply thresholding
        if threshold:
            # Convert to grayscale and apply threshold
            grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame = cv.threshold(grayFrame, thresholdValue, 255, cv.THRESH_BINARY)[1]

        # Get timestamp text size
        timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M")
        textSize, baseline = cv.getTextSize(timestamp, cv.FONT_HERSHEY_SIMPLEX, 1, 2)

        # Add timestamp to frame
        textX = frameWidth - textSize[0]
        textY = frameHeight - baseline
        cv.putText(frame, timestamp, (textX, textY), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Copy to top right corner
        timestampROI = frame[textY - textSize[1]:, textX:]
        frame[:-textY + textSize[1], textX:] = timestampROI

        # Update frame with red border
        frame = cv.copyMakeBorder(frame, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=(0, 0, 255))

        # Show frame and wait for keystroke
        cv.imshow("Camera", frame)
        key = cv.waitKey(1)

        # Capture photo and flash
        if key == ord('c'):
            # Flash screen
            flashFrame = frame.copy()
            if not threshold: flashFrame[:] = (255, 255, 255)
            else: flashFrame[:] = 255
            cv.imshow("Camera", flashFrame)
            cv.waitKey(100)

            # Name and save photo
            photoFilename = f"photo_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            cv.imwrite(photoFilename, frame)
            print(f"Saved {photoFilename}")

        # Start/stop recording video
        elif key == ord('v'):
            if not isRecording:
                # Name video and set recording
                isRecording = True
                videoFilename = f"video_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.avi"

                # Define codec and create video writer
                fourcc = cv.VideoWriter_fourcc(*"XVID")
                out = cv.VideoWriter(videoFilename, fourcc, 20.0, (640, 480))
                print(f"Started recording {videoFilename}")
            else:
                # Release video writer and set recording
                isRecording = False
                out.release()
                print(f"Stopped recording {videoFilename}")

        # Extract blue color
        elif key == ord('e'):
            # Convert to HSV color space and create mask
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, lowerBlue, upperBlue)

            # Apply mask to original frame and display
            extractedBlue = cv.bitwise_and(frame, frame, mask=mask)
            cv.imshow("Extracted Blue", extractedBlue)

        # Rotate frame by 10 degrees
        elif key == ord('r'):
            rotation += 10

        # Toggle threshold
        elif key == ord('t'):
            threshold = not threshold

        # Toggle Gaussian blur
        elif key == ord('b'):
            blur = not blur

        # Extract sharpened image
        elif key == ord('s'):
            # Create kernel and apply convolution
            kernel = np.array([[0, -1, 0],
                               [-1,  5, -1],
                               [0, -1, 0]])
            sharpenedFrame = cv.filter2D(frame, -1, kernel)
            cv.imshow("Sharpened", sharpenedFrame)

            # Generate Sobel gradient
            nextKey = cv.waitKey(0)
            if nextKey == ord('x'):
                # Create and apply Sobel kernel with odd size
                sobelKSizeX = cv.getTrackbarPos("Sobel X", "Camera")
                sobelKSizeX = sobelKSizeX if sobelKSizeX % 2 == 1 else sobelKSizeX + 1
                sobelX = cv.Sobel(frame, cv.CV_64F, 1, 0, ksize=sobelKSizeX)
                cv.imshow("Sobel X", sobelX)
            elif nextKey == ord('y'):
                # Create and apply Sobel kernel with odd size
                sobelKSizeY = cv.getTrackbarPos("Sobel Y", "Camera")
                sobelKSizeY = sobelKSizeY if sobelKSizeY % 2 == 1 else sobelKSizeY + 1
                sobelY = cv.Sobel(frame, cv.CV_64F, 1, 0, ksize=sobelKSizeY)
                cv.imshow("Sobel Y", sobelY)

        # Generate Canny edge detector
        elif key == ord('d'):
            # Create grayscale image and apply Canny
            grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            edges = cv.Canny(grayFrame, cannyThreshold1, cannyThreshold2)
            cv.imshow("Canny Edges", edges)

        # Generate original, laplacian, sobel X, and sobel Y
        elif key == ord('4'):
            # Apply custom operators
            sobelX = customSobelX(frame)
            sobelY = customSobelY(frame)
            laplacian = customLaplacian(frame)

            # Plot edge detection operations (OpenCV Docs)
            plt.subplot(2,2,1),plt.imshow(frame, cmap='gray')
            plt.title('Original'), plt.xticks([]), plt.yticks([])
            plt.subplot(2,2,2),plt.imshow(laplacian, cmap='gray')
            plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
            plt.subplot(2,2,3),plt.imshow(sobelX, cmap='gray')
            plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
            plt.subplot(2,2,4),plt.imshow(sobelY, cmap='gray')
            plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
            plt.show()

        # Exit app
        elif key == 27:
            break

    # Release everything
    cap.release()
    if isRecording:
        out.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
