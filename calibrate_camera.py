import cv2 as cv
from threading import Thread
import numpy as np


FILENAME = "calibration_data1.xml"

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# checkerboard Dimensions
cbrow = 6
cbcol = 9

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((cbrow * cbcol, 3), np.float32)
objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


# webcam class running a thread
class Webcam:
    def __init__(self):
        self.video_capture = cv.VideoCapture(1)
        self.current_frame = self.video_capture.read()[1]

    # create thread for capturing images
    def start(self):
        Thread(target=self._update_frame, args=()).start()

    def _update_frame(self):
        while True:
            self.current_frame = self.video_capture.read()[1]

    # get the current frame
    def get_current_frame(self):
        return self.current_frame


webcam = Webcam()
webcam.start()

for x in range(500):
    # get image from webcam
    image = webcam.get_current_frame()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9, 6), None)
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        image = cv.drawChessboardCorners(image, (9, 6), corners2, ret)
        cv.imshow('image', image)
        cv.waitKey(100)

print("calibrating...")
ret, camMatrix, distCoeff, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("camera matrix:\n", camMatrix)
print("distortion coefficients: ", distCoeff)
fs = cv.FileStorage(FILENAME, cv.FILE_STORAGE_WRITE)
fs.write("camera_matrix", camMatrix)
fs.write("distortion_coefficients", distCoeff)
fs.release()
