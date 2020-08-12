/*#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat src,copyimg,img_th,inpaint_img_result,denosing_img_result, img_gy;
bool mousedown;
vector<vector<Point> > contours;
vector<Point> pts;
//void load_cascade(CascadeClassifier& cascade, string fname)
//{
//	String path = "C:\\opencv410\\opencv\\sources\\data\\haarcascades\\";
//	String full_name = path + fname;
//
//	CV_Assert(cascade.load(full_name));
//}
bool findPimples(Mat img)
{
	Mat bw, bgr[3];
	split(img, bgr);
	bw = bgr[1];
	int pimplescount = 0;
	vector<Rect> blurrect;
	adaptiveThreshold(bw, bw, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 5);
	dilate(bw, bw, Mat(), Point(-1, -1), 1);

	contours.clear();
	findContours(bw, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

	for (size_t i = 0; i < contours.size(); i++)
	{
		if (contourArea(contours[i]) > 25 & contourArea(contours[i]) < 120)//25,120
		{
			Rect im_rect,minRect = boundingRect(Mat(contours[i]));

			Mat imgroi(img, minRect);

			cvtColor(imgroi, imgroi, COLOR_BGR2HSV);
			Scalar color = mean(imgroi);
			cvtColor(imgroi, imgroi, COLOR_HSV2BGR);

			if (color[0] < 30 & color[1] > 90 & color[2] > 50)// 30,90,50
			{
				Point2f center;
				float radius = 0;
				minEnclosingCircle(Mat(contours[i]), center, radius);

				if (radius < 50)//50
				{
					rectangle(img, minRect, Scalar(0, 0, 0),-1);
					pimplescount++;
				}
				blurrect.push_back(minRect);
			}
		}
		
	}
	
	//putText(img, format("%d", pimplescount), Point(50, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 2);
	imshow("pimples dedector", img);//copyimg 원본이미지, img 여드름체킹이미지, img_th 이진화이미지, img_result 결과 이미지
	
	cvtColor(img, img_gy, COLOR_BGR2GRAY);
	threshold(img_gy, img_th, 0, 255, 1);

	inpaint(img, img_th, inpaint_img_result, 15, INPAINT_TELEA); //circular neighborhood의 radius, 15
	fastNlMeansDenoisingColored(inpaint_img_result, denosing_img_result, 3, 3, 7, 21);//3,3,7,21
	imshow("denosing_img_result", denosing_img_result);
	//imshow("img_th", img_th);

	//imshow("inpaint_img_result", inpaint_img_result);
	return 0;
}


int main()
{
	src = imread("pimples2.jpg",1);
	
	if (src.empty())
	{
		return -1;
	}
	copyimg=src.clone();

	imshow("원본이미지", copyimg);
	findPimples(src);
	
	waitKey(0);
	return 0;
}*/
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <iostream>
#include <algorithm>
using namespace dlib;
using namespace std;
using namespace cv;
int main()
{
	Mat image = imread("pimples.jpg", 1);
	image_window win;

	// Load face detection and pose estimation models.
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;
	deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
	
	cv_image<bgr_pixel> cimg(image);
	
	while (!win.is_closed())
	{
		// Detect faces
		
		std::vector<dlib::rectangle> faces = detector(cimg);
		
		// Find the pose of each face.
		std::vector<full_object_detection> shapes,faceroundtest;
		std::vector<dlib::point> faceround;
		std::vector<cv::Point2f> cvfaceround;
		for (unsigned long i = 0; i < faces.size(); ++i) {
			shapes.push_back(pose_model(cimg, faces[i]));
			for (int a = 1; a < shapes[i].num_parts(); a++){
				if (a >= 1 && a <= 27) {
					faceround.push_back(shapes[i].part(a));
					faceroundtest.push_back(pose_model(cimg, faceround[i]));
					
				}
			}
		}
		circle(image, Point(faceround[2].x(), faceround[2].y()), 10, Scalar(255, 0, 0), 1, 8, 0);
		// Display it all on the screen
		win.clear_overlay();
		win.set_image(cimg);
		win.add_overlay(render_face_detections(shapes));
		//win.add_overlay(render_face_detections(faceroundtest,rgb_pixel(255,0,0)));
		//Mat cvfaceimg = toMat(win);
		imshow("image", image);
		waitKey();
	}
}




