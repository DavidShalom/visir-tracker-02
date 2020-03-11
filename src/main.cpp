#include <iostream>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <cstdlib>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/video.hpp>
#include "types.h"

using namespace cv;

// Global Variables
int key;
VideoCapture camera;

// Function Headers 
void featureExtraction(void);
void sparseOpticalFlowEstimation(void);

int main(int argc, const char** argv)
{

      featureExtraction();
      sparseOpticalFlowEstimation();

      return EXIT_SUCCESS;
}

// Function Definitions 

void featureExtraction(void)
{
        if (!camera.open(0))
	{
	    std::cerr << "Can't find a camera";
	    return;
	};
	
	Mat img, img_gray;
	std::vector<Point2f> corners;
	int i, frames_counter = 0;

	while(1)
	{
		camera >> img;

		if(frames_counter % 300 == 0)
		{
		  cvtColor(img, img_gray, COLOR_BGR2GRAY); 
		  goodFeaturesToTrack(img_gray, corners, 125, 0.01, 10);
		}

		for(i = 0; i < corners.size(); i++)
		{
		  circle(img, corners[i], 3, Scalar( 255, 0, 0 ), -2);
		}   
		  
		imshow("Camera with feature Extraction", img);
		key = waitKey(5);
		if (key == 27 || key == 'q') break;
		frames_counter++;
	}
	camera.release();	
}

void sparseOpticalFlowEstimation(void)
{    
    if (!camera.open(0))
    {
        std::cerr << "Can't find a camera";
        return;
    }
    
    std::vector<Scalar> colors;
    RNG rand;
    
    for(int i = 0; i < 100; i++)
    {
        int r = rand.uniform(0, 256);
        int g = rand.uniform(0, 256);
        int b = rand.uniform(0, 256);
        colors.push_back(Scalar(r,g,b));
    }
    
    Mat old_img, old_gray;
    std::vector<Point2f> p0, p1;
    
    camera >> old_img;
    
    cvtColor(old_img, old_gray, COLOR_BGR2GRAY);
    goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);
    
    Mat mask = Mat::zeros(old_img.size(), old_img.type());
    int frames_counter = 0;
    
    while(1)
    {
      if(frames_counter % 2 == 0)
      {
	Mat img, img_gray;
	camera >> img;
	
	cvtColor(img, img_gray, COLOR_BGR2GRAY);
	
	std::vector<uchar> status;
	std::vector<float> err;
	TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
	 
	calcOpticalFlowPyrLK(old_gray, img_gray, p0, p1, status, err, Size(15,15), 2, criteria);
	std::vector<Point2f> good_new;

	for(uint i = 0; i < p0.size(); i++)
	  {
	    if(status[i] == 1)
	      {
		good_new.push_back(p1[i]);
		line(mask,p1[i], p0[i], colors[i], 2);
		circle(img, p1[i], 5, colors[i], -2);
	      }
	  }
	
	Mat final_img;
	add(img, mask, final_img);
	imshow("Camera with sparse optical flow estimation", final_img);
	key = waitKey(5);
	if (key == 27 || key == 'q') break; 
	old_gray = img_gray.clone();
	p0 = good_new;
      }
      frames_counter++;
    }
    camera.release();
}
