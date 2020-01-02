#include "methods.h"

using namespace std;
using namespace cv;

extern std::map<CENSUS_ALGORITHM, std::string> methods_str;

void makeImgForPaper()
{
	cv::Mat img_ = imread("D:/studying/stereo vision/研究生毕业论文+答辩准备/现场系统图片/支架.png");
	cv::Mat img1, img2, img3;
	
	std::vector<cv::Mat> img_split;
	split(img_, img_split);
	std::vector<cv::Mat> img_filter(3);
	for(int i = 0; i < 3; i++)
	{
		img_filter[i] = getGuidedFilter(img_, img_split[i], 7, 1e-6);
	}
	merge(img_filter, img1);

	std::vector<cv::Mat> img_equalDis(3);
	for (int i = 0; i < 3; i++)
	{
		equalizeHist(img_split[i], img_equalDis[i]);
	}
	merge(img_equalDis, img2);

	cv::Mat mask_;
	createMask_lines2(mask_);

	blur(img2, img3, Size(15, 15));
	cv::Mat img4 = cv::Mat::zeros(img_.size(), CV_8UC3);
	img3.copyTo(img4, mask_);
	bitwise_not(mask_, mask_);
	img2.copyTo(img4, mask_);
	
}

int main()
{
	Mat imgL, imgR;
	imgL = imread("D:/studying/stereo vision/research code/local-stereoMatch/data_testImgForPaper/rectifyL.jpg");//"D:/studying/stereo vision/research code/data/ALL-2views/Wood1/view1.png"
	imgR = imread("D:/studying/stereo vision/research code/local-stereoMatch/data_testImgForPaper/rectifyR.jpg");//"D:/studying/stereo vision/research code/data/ALL-2views/Wood1/view5.png"

	//"D:/studying/stereo vision/research code/local-stereoMatch/data_testImgForPaper/1L_rectify.jpg"
	//"D:/studying/stereo vision/research code/local-stereoMatch/data_testImgForPaper/1R_rectify.jpg"
	//equalHisImg(imgL, imgL);
	//equalHisImg(imgR, imgR);

	filtImg(imgL, imgL, 15, 1e-10);
	filtImg(imgR, imgR, 15, 1e-10);

	equalHisImg(imgL, imgL);
	equalHisImg(imgR, imgR);
	//{
	//	Mat hsvL, hsvR;
	//	cvtColor(imgL, hsvL, COLOR_BGR2HSV);
	//	cvtColor(imgR, hsvR, COLOR_BGR2HSV);

	//	std::vector<Mat> mat3cn;
	//	split(hsvL, mat3cn);
	//	Mat blurL, blurR;
	//	bilateralFilter(mat3cn[2], blurL, 7, 10, 3, BORDER_REFLECT);
	//	Mat detailL, detailR;
	//	detailL = mat3cn[2] - blurL;
	//	mat3cn[2] = mat3cn[2] + detailL * 2;
	//	merge(mat3cn, hsvL);
	//	cvtColor(hsvL, imgL, COLOR_HSV2BGR);

	//	mat3cn.clear();
	//	split(hsvR, mat3cn);
	//	bilateralFilter(mat3cn[2], blurR, 7, 10, 3, BORDER_REFLECT);
	//	detailR = mat3cn[2] - blurR;
	//	mat3cn[2] = mat3cn[2] + detailR * 2;
	//	merge(mat3cn, hsvR);
	//	cvtColor(hsvR, imgR, COLOR_HSV2BGR);
	//}
	resize(imgL, imgL, Size(640, 360));
	resize(imgR, imgR, Size(640, 360));

	//cv::Mat imgL_, imgR_;
	//cvtColor(imgL, imgL_, COLOR_BGR2GRAY);
	//cvtColor(imgR, imgR_, COLOR_BGR2GRAY);


	StereoMatchParam param;
	param.imgLeft = imgL;
	param.imgRight = imgR;
	param.imgLeft_C = imgL;
	param.imgRight_C = imgR;

	param.isDispLeft = true;
	param.minDisparity = 0;
	param.maxDisparity = 120;
	param.winSize = 35;
	Mat disparityImgL;
	disparityImgL = censusStereo(param, BASIC_CENSUS);//CIRCLE_CENSUS//MULTISCALE_CENSUS//BASIC_CENSUS

	param.isDispLeft = false;
	Mat disparityImgR;
	disparityImgR = censusStereo(param, BASIC_CENSUS);

	Mat disparityImgL_, disparityImgR_;
	normalize(disparityImgL, disparityImgL_, 0, 255, NORM_MINMAX);
	disparityImgL_.convertTo(disparityImgL_, CV_8UC1);
	imwrite(methods_str[BASIC_CENSUS] + "_censusL.jpg", disparityImgL_);
	normalize(disparityImgR, disparityImgR_, 0, 255, NORM_MINMAX);
	disparityImgR_.convertTo(disparityImgR_, CV_8UC1);
	imwrite(methods_str[BASIC_CENSUS] + "_censusR.jpg", disparityImgR_);

	return 0;
}