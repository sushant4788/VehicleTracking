#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\imgproc\imgproc.hpp>
using namespace std;
using namespace cv;
// Declare the constants for the program 
const int MEDIAN_FILTER_MASK = 7;
const int BINARY_THRESHOLDING_VALUE = 200;
const int AREA_THRESHOLD = 150;

int main(){
	Mat frame, fore, back, medFilt, threshed, clone_threshed;
	//read the input video file frame by frame
	VideoCapture vid ("movie.wmv");
	if (!vid.isOpened()){
		cout << "Error opening media file " << endl;
		exit(EXIT_FAILURE);
	}
	//Take the initial image for getting the centers and other static information
	Mat initialImage;
	vid >> initialImage;
	Point2f center_point;
	if(initialImage.rows == 0){
		cout << "Faliure to read the initial image" << endl;
		exit (EXIT_FAILURE);
	}
	center_point.x = initialImage.cols/2;
	center_point.y = initialImage.rows/2;
	cout <<"The center point of the static image is :" <<center_point << endl;
	
	// Set the dialation and erosion mask
	Mat d_dilate = getStructuringElement( MORPH_ELLIPSE, Size (5,5), Point(0,0));
	Mat e_erode = getStructuringElement( MORPH_ELLIPSE, Size (5,5), Point(0,0));
	
	// Create a MOG2 based background subtraction object 
	BackgroundSubtractorMOG2 bg;

	// Set the initial parameters for Kalman Filter here
	Mat KalmanWin = Mat::zeros(initialImage.rows, initialImage.cols, CV_8UC3);
	KalmanFilter KF(4,2,0);

	KF.transitionMatrix = *(Mat_<float>(4,4) << 1,0,1,0,  0,1,0,1, 0,0,1,0, 0,0,0,1 );
	Mat_<float> measurement(2,1);
	measurement.setTo(Scalar(0));

	KF.statePre.at<float>(0) = center_point.x;
	KF.statePre.at<float>(1) = center_point.y;
	KF.statePre.at<float>(2) = 0;
	KF.statePre.at<float>(3) = 0;

	setIdentity(KF.measurementMatrix);
	setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(KF.errorCovPost, Scalar::all(0.1));
	// set all the named Windows here 
	namedWindow("VideoFrame", CV_WINDOW_AUTOSIZE);
	namedWindow("Processed1", CV_WINDOW_AUTOSIZE);
	namedWindow("Processed2", CV_WINDOW_AUTOSIZE);
	

	//============================================================================//
	//======================The main while loop===================================//
	//============================================================================//

	while (true){
		std :: vector <std :: vector <cv :: Point>> contours;
		std :: vector<Vec4i> hierarchy;
		vid >> frame;
		if (frame.rows == 0){
			destroyAllWindows();
		}
		bg.operator() (frame, fore);
		medianBlur(fore, medFilt, MEDIAN_FILTER_MASK);
		erode(medFilt, medFilt, e_erode);
		dilate(medFilt, medFilt, d_dilate);
		dilate(medFilt, medFilt, d_dilate);
		threshold(medFilt, threshed, BINARY_THRESHOLDING_VALUE,255,THRESH_BINARY);
		clone_threshed = threshed.clone();
		findContours(clone_threshed, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE,Point(0,0));
		vector <Rect> boundRect (contours.size());
		for (int i = 0; i < contours.size(); ++i){
			if (contourArea(contours[i]) > AREA_THRESHOLD ){
				boundRect[i] = boundingRect(Mat(contours[i]));
				rectangle(frame, boundRect[i].tl(), boundRect[i].br(), Scalar(0,0,255), 2,8,0);
				circle(KalmanWin, boundRect[i].tl(), 5, Scalar(255,0,0), -1, 8);
				Mat prediction = KF.predict();
				Point predictPoint( prediction.at<float>(0), prediction.at<float>(1) );
				measurement(0) = boundRect[i].tl().x;
				measurement(1) = boundRect[i].tl().y;
				Point measPoint(measurement(0), measurement(1));
				Mat estimated = KF.correct(measurement);
				Point statepoint(estimated.at<float>(0), estimated.at<float>(1));
				Point estimatedBR; 
				estimatedBR.x	= statepoint.x ;
				estimatedBR.y   = statepoint.y ;
				circle(KalmanWin, estimatedBR, 1, Scalar(0,255,0), -1, 8);
			}
		}
						
		imshow("Processed1", threshed);
		imshow("Processed2", KalmanWin);
		//imshow("VideoFrame", frame);
		waitKey(22);
	}

}




