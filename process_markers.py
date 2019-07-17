import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import math

DICTIONARYID = 0
MARKERLENGTH = 0.05
IMG_FILENAME = "pos45.png"
XML_FILENAME = "sample_camera.xml"


# Checks if a matrix is a valid rotation matrix.
def is_rotation_matrix(R):
	Rt = np.transpose(R)
	should_be_identity = np.dot(Rt, R)
	I = np.identity(3, dtype=R.dtype)
	n = np.linalg.norm(I - should_be_identity)
	return n < 1e-6


# Calculates rotation matrix to euler angles x y z
def rotation_matrix_to_euler_angles(R):
	assert (is_rotation_matrix(R))
	sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
	singular = sy < 1e-6
	if not singular:
		x = math.atan2(R[2, 1], R[2, 2]) / math.pi * 180
		y = math.atan2(-R[2, 0], sy) / math.pi * 180
		z = math.atan2(R[1, 0], R[0, 0]) / math.pi * 180
	else:
		x = math.atan2(-R[1, 2], R[1, 1]) / math.pi * 180
		y = math.atan2(-R[2, 0], sy) / math.pi * 180
		z = 0 / math.pi * 180
	return np.array([x, y, z])


# detect markers from the input image
inputImage = cv.imread(IMG_FILENAME)
inputImage = cv.resize(inputImage,None, fx=0.2, fy=0.2, interpolation=cv.INTER_CUBIC)
dictionary = aruco.Dictionary_get(DICTIONARYID)
parameters = aruco.DetectorParameters_create()
markerCorners, markerIds, rejectedCandidates = aruco.detectMarkers(inputImage, dictionary, parameters=parameters)

# find index of center marker
index = 0
for i in range(len(markerIds)):
	if markerIds[i] == 0:
		index = i
		break

# TODO: camera calibration

# read camera calibration data
fs = cv.FileStorage(XML_FILENAME, cv.FILE_STORAGE_READ)
if not fs.isOpened():
	print("Invalid camera file")
	exit(-1)
camMatrix = fs.getNode("camera_matrix").mat()
distCoeffs = fs.getNode("distortion_coefficients").mat()

# pose estimation
if not len(markerIds) == 0:
	rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(markerCorners, MARKERLENGTH, camMatrix, distCoeffs)

# get angle from rotational matrix
# convert rotational vector rvecs to rotational matrix
rmat = np.empty([3, 3])
cv.Rodrigues(rvecs[index][0], rmat)
euler_angle = rotation_matrix_to_euler_angles(rmat)

# display annotations (IDs and pose)
imageCopy = inputImage.copy()
if not len(markerIds) == 0:
	cv.putText(imageCopy, "Cozmo Pose", (10, 20), cv.FONT_HERSHEY_PLAIN, 1, (255, 50, 255, 0))
	msg = "X(m): " + str(tvecs[index][0][0])
	cv.putText(imageCopy, msg, (10, 45), cv.FONT_HERSHEY_PLAIN, 1, (255, 50, 255, 0))
	msg = "Y(m): " + str(tvecs[index][0][1])
	cv.putText(imageCopy, msg, (10, 70), cv.FONT_HERSHEY_PLAIN, 1, (255, 50, 255, 0))
	msg = "Angle(deg): " + str(euler_angle[2])
	cv.putText(imageCopy, msg, (10, 95), cv.FONT_HERSHEY_PLAIN, 1, (255, 50, 255, 0))
	aruco.drawDetectedMarkers(imageCopy, markerCorners, markerIds)
	aruco.drawAxis(imageCopy, camMatrix, distCoeffs, rvecs[index][0], tvecs[index][0], MARKERLENGTH * 0.5)
	# get distance from other markers
	x_0 = tvecs[index][0][0]
	y_0 = tvecs[index][0][1]
	cv.putText(imageCopy, "Distances to Objects(m)", (300, 20), cv.FONT_HERSHEY_PLAIN, 1, (150, 200, 200, 0))
	for i in range(len(tvecs)):
		x = tvecs[i][0][0]
		y = tvecs[i][0][1]
		dx = max(x, x_0) - min(x, x_0)
		dy = max(y, y_0) - min(y, y_0)
		dist = math.sqrt(dx * dx + dy * dy)
		msg = "id=" + str(markerIds[i]) + ": " + str(dist)
		cv.putText(imageCopy, msg, (300, 45 + 25 * i), cv.FONT_HERSHEY_PLAIN, 1, (150, 200, 200, 0))
cv.imshow("Detect Markers", imageCopy)
cv.waitKey(0)
