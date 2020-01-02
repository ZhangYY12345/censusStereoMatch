#pragma once
#include <opencv2/opencv.hpp>

struct StereoMatchParam
{
	cv::Mat imgLeft;
	cv::Mat imgRight;
	cv::Mat imgLeft_C;
	cv::Mat imgRight_C;

	int winSize;		//=7
	int minDisparity;	//=0
	int maxDisparity;	//=30
	bool isDispLeft;
};

enum CENSUS_ALGORITHM
{
	BASIC_CENSUS,
	CIRCLE_CENSUS,
	ROTATION_INVARIANT_CENSUS,
	UNIFORM_CENSUS,					//效果较差
	MULTISCALE_CENSUS,
	STATISTIC_MULTISCALE_CENSUS,	//效果较差
	CENSUS_2017,
};

void createMask_lines2(cv::Mat& dst);

bool checkImg(cv::Mat& src);
bool checkPairs(cv::Mat src1, cv::Mat src2);
bool checkPoint(int width, int height, int x, int y);
float getMatVal(cv::Mat img, int x, int y);

// equal histogram
void equalHisImg(cv::Mat src, cv::Mat& dst);
//filter
void filtImg(cv::Mat src, cv::Mat& dst, int winSize, double eps);

// guided filter
cv::Mat multiChl_to_oneChl_mul(cv::Mat firstImg, cv::Mat secondImg);
cv::Mat getGuidedFilter(cv::Mat guidedImg, cv::Mat inputP, int r, double eps);

//sum of absolute differences
cv::Mat computeSAD_inteOpti(StereoMatchParam param);
cv::Mat computeSAD_BFOpti(StereoMatchParam param);

// census transform algorithm : 算法在边界处明显失效
// original census algorithm
void countCensusImg(cv::Mat src, cv::Mat& dst);
void countCensusImg_circle(cv::Mat src, cv::Mat& dst, int radius, int samplePtNum = 8);
void countCensusImg_rotationInv(cv::Mat src, cv::Mat& dst);
//
int hopCount(uchar i);
void countCensusImg_uniform(cv::Mat src, cv::Mat& dst);
//
void countCensusImg_multiScale(cv::Mat src, cv::Mat& dst, int scale);
void countCensusImg_multiScale2(cv::Mat src, cv::Mat& dst, int scale);
// improved census transform algorithm
void countCensusImg_2017(cv::Mat src, cv::Mat& dst, int winSize);
////
// stereo matching with different census image computing algorithm
void countHummingDist(cv::Mat src1, cv::Mat src2, cv::Mat& dst);
cv::Mat censusStereo(StereoMatchParam param, CENSUS_ALGORITHM method);
