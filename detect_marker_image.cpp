#define _USE_MATH_DEFINES

#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>

using namespace std;
using namespace cv;

const static int DICTIONARYID = 0;
static const float MARKERLENGTH = 0.05;
static const string IMG_FILENAME = "pos45.png";
static const string XML_FILENAME = "sample_camera.xml";

static bool readCameraParameters(string filename, Mat& camMatrix, Mat& distCoeffs) {
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened()) return false;
	fs["camera_matrix"] >> camMatrix;
	fs["distortion_coefficients"] >> distCoeffs;
	return true;
}

// Checks if a matrix is a valid rotation matrix.
bool isRotationMatrix(Mat &R) {
    Mat Rt;
    transpose(R, Rt);
    Mat shouldBeIdentity = Rt * R;
    Mat I = Mat::eye(3,3, shouldBeIdentity.type());
    return norm(I, shouldBeIdentity) < 1e-6;
}
 
// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
Vec3f rotationMatrixToEulerAngles(Mat &R) {
    assert(isRotationMatrix(R));
    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
    bool singular = sy < 1e-6; // If
    float x, y, z;
    if (!singular) {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    } else {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return Vec3f(x / M_PI * 180, y / M_PI * 180, z / M_PI * 180);
}

int main() {
	// read input image
	Mat inputImage, imageCopy;
	try {
		inputImage = imread(IMG_FILENAME);
		resize(inputImage, inputImage, Size(), 0.2, 0.2, INTER_CUBIC);
	} catch (Exception& e) {
		cerr << e.msg << endl;
		return 1;
	}

	// detect markers from the input image
	vector<int> markerIds;
	vector<vector<Point2f>> markerCorners, rejectedCandidates;
	Ptr<aruco::DetectorParameters> parameters = aruco::DetectorParameters::create();
	Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(DICTIONARYID));
	aruco::detectMarkers(inputImage, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

	// find index of center marker
	int index = 0;
	for (int i = 0; i < markerIds.size(); ++i) {
		if (markerIds[i] == 0) {
			index = i;
			break;
		}
	}

	// TODO: camera calibration

	// read camera calibration data
	Mat camMatrix, distCoeffs;
	bool readOk = readCameraParameters(XML_FILENAME, camMatrix, distCoeffs);
	if (!readOk) {
		cerr << "Invalid camera file" << endl;
		return -1;
	}

	/* pose estimation */
	vector<Vec3d> rvecs, tvecs;
	if (!markerIds.empty()) {
		try {
			aruco::estimatePoseSingleMarkers(markerCorners, MARKERLENGTH, camMatrix, distCoeffs, rvecs, tvecs);
		} catch (Exception& e) {
			cerr << e.msg << endl;
			return -2;
		}
	}

	// get angle from rotational matrix
	// convert rotational vector rvecs to rotational matrix
	Mat rmat;
	Rodrigues(rvecs[index], rmat, noArray());
	Vec3f euler_angle = rotationMatrixToEulerAngles(rmat);

	// display annotations (IDs and pose)
	inputImage.copyTo(imageCopy);
	if (!markerIds.empty()) {
		putText(imageCopy, "Cozmo Pose", Point2f(10, 20), FONT_HERSHEY_PLAIN, 1, Scalar(255, 50, 255, 0));
		char str[200];
		sprintf_s(str, "X(m): %f", tvecs[index][0]);
		putText(imageCopy, str, Point2f(10, 45), FONT_HERSHEY_PLAIN, 1, Scalar(255, 50, 255, 0));
		sprintf_s(str, "Y(m): %f", tvecs[index][1]);
		putText(imageCopy, str, Point2f(10, 70), FONT_HERSHEY_PLAIN, 1, Scalar(255, 50, 255, 0));
		sprintf_s(str, "Angle(deg): %f", euler_angle[2]);
		putText(imageCopy, str, Point2f(10, 95), FONT_HERSHEY_PLAIN, 1, Scalar(255, 50, 255, 0));
		aruco::drawDetectedMarkers(imageCopy, markerCorners, markerIds);
		aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rvecs[index], tvecs[index], MARKERLENGTH * 0.5f);
		// get distance from other markers
		putText(imageCopy, "Distances to Objects(m)", Point2f(300, 20), FONT_HERSHEY_PLAIN, 1, Scalar(150, 200, 200, 0));
		for (int i = 0; i < tvecs.size(); ++i) {
			float x_0 = tvecs[index][0];
			float y_0 = tvecs[index][1];
			float x = tvecs[i][0];
			float y = tvecs[i][1];
			float dx = max(x, x_0) - min(x, x_0);
			float dy = max(y, y_0) - min(y, y_0);
			float dist = sqrt(dx * dx + dy * dy);
			int id = markerIds[i];
			sprintf_s(str, "id=%i: %f", id, dist);
			putText(imageCopy, str, Point2f(300, 45 + 25 * i), FONT_HERSHEY_PLAIN, 1, Scalar(150, 200, 200, 0));
		}
	}
	imshow("Detect Markers", imageCopy);
	waitKey(0);

	return 0;
}

