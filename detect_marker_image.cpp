#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

const static int DICTIONARYID = 0;
static const float MARKERLENGTH = 0.1;
static const string IMG_FILENAME = "pos1.png";
static const string XML_FILENAME = "sample_camera.xml";

static bool readCameraParameters(string filename, Mat& camMatrix, Mat& distCoeffs) {
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened()) return false;
	fs["camera_matrix"] >> camMatrix;
	fs["distortion_coefficients"] >> distCoeffs;
	return true;
}

int main() {
	Mat inputImage, imageCopy;
	try {
		inputImage = imread(IMG_FILENAME);
		resize(inputImage, inputImage, Size(), 0.2, 0.2, INTER_CUBIC);
	} catch (Exception& e) {
		cerr << e.msg << endl;
		return 1;
	}

	vector<int> markerIds;
	vector<vector<Point2f>> markerCorners, rejectedCandidates;
	Ptr<aruco::DetectorParameters> parameters = aruco::DetectorParameters::create();
	Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(DICTIONARYID));

	aruco::detectMarkers(inputImage, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

	// TODO: camera calibration

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

	inputImage.copyTo(imageCopy);
	if (!markerIds.empty()) {
		aruco::drawDetectedMarkers(imageCopy, markerCorners, markerIds);
	}
	for (size_t i = 0; i < markerIds.size(); i++) {
		try {
			aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rvecs[i], tvecs[i], MARKERLENGTH * 0.5f);
		}
		catch (Exception& e) {
			cerr << e.msg << endl;
			return i;
		}
	}
	imshow("Detect Markers", imageCopy);
	waitKey(0);

	return 0;
}

