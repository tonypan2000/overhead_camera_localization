import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import math

DICTIONARYID = 0
MARKERLENGTH = 0.035
IMG_FILENAME = "pos45.png"
XML_FILENAME = "calibration_data.xml"


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


cam = cv.VideoCapture(1)
cv.namedWindow("HD Pro Webcam C920")

# read camera calibration data
fs = cv.FileStorage(XML_FILENAME, cv.FILE_STORAGE_READ)
if not fs.isOpened():
    print("Invalid camera file")
    exit(-1)
camMatrix = fs.getNode("camera_matrix").mat()
distCoeffs = fs.getNode("distortion_coefficients").mat()

while True:

    ret, inputImage = cam.read()
    # detect markers from the input image
    dictionary = aruco.Dictionary_get(DICTIONARYID)
    parameters = aruco.DetectorParameters_create()
    markerCorners, markerIds, rejectedCandidates = aruco.detectMarkers(inputImage, dictionary, parameters=parameters)

    if markerIds is not None:
        # find index of center marker
        index = 0
        index1 = 0
        for i in range(len(markerIds)):
            if markerIds[i] == 0:
                index = i
            elif markerIds[i] == 1:
                index1 = i

        # pose estimation
        if len(markerIds) > 1:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(markerCorners, MARKERLENGTH, camMatrix, distCoeffs)

            # translate tvecs from relation to camera to a marker
            tvecs[index][0] -= tvecs[index1][0]

            # flip the x axis
            tvecs[index][0][0] = -tvecs[index][0][0]

            # get angle from rotational matrix
            # convert rotational vector rvecs to rotational matrix
            # convert euler in relation to a marker
            rmat = np.empty([3, 3])
            cv.Rodrigues(rvecs[index][0], rmat)
            rmat_0 = np.empty([3, 3])
            cv.Rodrigues(rvecs[index1][0], rmat_0)
            euler_angle1 = rotation_matrix_to_euler_angles(rmat_0)
            euler_angle = rotation_matrix_to_euler_angles(rmat) - euler_angle1

            # display annotations (IDs and pose)
            imageCopy = inputImage.copy()
            cv.putText(imageCopy, "Cozmo Pose", (10, 20), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0, 0))
            msg = "X(m): " + str(tvecs[index][0][0])
            cv.putText(imageCopy, msg, (10, 45), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0, 0))
            msg = "Y(m): " + str(tvecs[index][0][1])
            cv.putText(imageCopy, msg, (10, 70), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0, 0))
            # msg = "Z(m): " + str(tvecs[index][0][2])
            # cv.putText(imageCopy, msg, (10, 95), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0, 0))
            msg = "Angle(deg): " + str(euler_angle[2])
            cv.putText(imageCopy, msg, (10, 120), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0, 0))
            aruco.drawDetectedMarkers(imageCopy, markerCorners, markerIds)
            cv.imshow("Detect Markers", imageCopy)
            cv.waitKey(100)
