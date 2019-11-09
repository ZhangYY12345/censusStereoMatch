#include "methods.h"

using namespace std;
using namespace cv;

int main()
{
	Mat imgL, imgR;
	imgL = imread("D:/studying/stereo vision/research code/data/cones-png-2/cones/im2.png");
	imgR = imread("D:/studying/stereo vision/research code/data/cones-png-2/cones/im6.png");

	StereoMatchParam param;
	param.imgLeft = imgL;
	param.imgRight = imgR;
	param.isDispLeft = true;
	param.minDisparity = 0;
	param.maxDisparity = 64;
	param.winSize = 5;
	Mat disparityImg;
	disparityImg = censusStereo(param, CENSUS_2017);
	normalize(disparityImg, disparityImg, 0, 255, NORM_MINMAX);
	imwrite("3*3_census.jpg", disparityImg);
	return 0;
}