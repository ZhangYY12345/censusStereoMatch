#include "methods.h"
#include <iso646.h>
#include <future>

using namespace cv;

//BASIC_CENSUS,
//CIRCLE_CENSUS,
//ROTATION_INVARIANT_CENSUS,
//UNIFORM_CENSUS,					//效果较差
//MULTISCALE_CENSUS,
//STATISTIC_MULTISCALE_CENSUS,	//效果较差
//CENSUS_2017,

std::map<CENSUS_ALGORITHM, std::string> methods_str =
{
	{BASIC_CENSUS, "basicCensus"},
	{CIRCLE_CENSUS, "circleCensus"},
	{ROTATION_INVARIANT_CENSUS, "rotationInvCensus"},
	{UNIFORM_CENSUS, "uniformCensus"},
	{MULTISCALE_CENSUS, "multiScaleCensus"},
	{STATISTIC_MULTISCALE_CENSUS, "statisticMultiScaleCensus"},
	{CENSUS_2017, "2017Census"},
};

void createMask_lines2(cv::Mat& dst)
{
	std::vector<std::vector<cv::Point2i> > contours;
	{
		std::vector<cv::Point2i> oneContour;

		cv::Point2i p1(146, 26);
		cv::Point2i p2(159, 23);
		cv::Point2i p3(180, 24);
		cv::Point2i p4(183, 108);
		cv::Point2i p5(194, 110);
		cv::Point2i p6(195, 135);
		cv::Point2i p7(184, 138);
		cv::Point2i p8(185, 295);
		cv::Point2i p9(191, 300);
		cv::Point2i p10(191, 318);
		cv::Point2i p11(187, 320);
		cv::Point2i p12(193, 480);
		cv::Point2i p13(223, 465);
		cv::Point2i p14(217, 374);
		cv::Point2i p15(238, 379);
		cv::Point2i p16(237, 367);
		cv::Point2i p17(252, 360);
		cv::Point2i p18(269, 372);
		cv::Point2i p19(269, 426);
		cv::Point2i p20(261, 430);
		cv::Point2i p21(261, 515);
		cv::Point2i p22(183, 550);
		cv::Point2i p23(169, 546);
		cv::Point2i p24(159, 319);
		cv::Point2i p25(155, 314);
		cv::Point2i p26(131, 301);
		cv::Point2i p27(125, 303);
		cv::Point2i p28(115, 286);
		cv::Point2i p29(121, 272);
		cv::Point2i p30(133, 272);
		cv::Point2i p31(158, 282);
		cv::Point2i p32(149, 149);
		cv::Point2i p33(133, 136);
		cv::Point2i p34(108, 138);
		cv::Point2i p35(107, 114);
		cv::Point2i p36(132, 111);
		cv::Point2i p37(149, 91);
		cv::Point2i p38(146, 27);


		oneContour.push_back(p1);
		oneContour.push_back(p2);
		oneContour.push_back(p3);
		oneContour.push_back(p4);
		oneContour.push_back(p5);
		oneContour.push_back(p6);
		oneContour.push_back(p7);
		oneContour.push_back(p8);
		oneContour.push_back(p9);
		oneContour.push_back(p10);
		oneContour.push_back(p11);
		oneContour.push_back(p12);
		oneContour.push_back(p13);
		oneContour.push_back(p14);
		oneContour.push_back(p15);
		oneContour.push_back(p16);
		oneContour.push_back(p17);
		oneContour.push_back(p18);
		oneContour.push_back(p19);
		oneContour.push_back(p20);
		oneContour.push_back(p21);
		oneContour.push_back(p22);
		oneContour.push_back(p23);
		oneContour.push_back(p24);
		oneContour.push_back(p25);
		oneContour.push_back(p26);
		oneContour.push_back(p27);
		oneContour.push_back(p28);
		oneContour.push_back(p29);
		oneContour.push_back(p30);
		oneContour.push_back(p31);
		oneContour.push_back(p32);
		oneContour.push_back(p33);
		oneContour.push_back(p34);
		oneContour.push_back(p35);
		oneContour.push_back(p36);
		oneContour.push_back(p37);
		oneContour.push_back(p38);

		contours.push_back(oneContour);
	}
	//
	int width = 368;
	int height = 653;
	cv::Mat img = cv::Mat::zeros(height, width, CV_8UC1);

	drawContours(img, contours, -1, 255, FILLED);
	//imwrite("img_.jpg", img);
	bitwise_not(img, dst);
}

/**
 * \brief check if the imput image is CV_8UC1 type
 * \param src 
 */
bool checkImg(cv::Mat& src)
{
	if(!src.data)
	{
		std::cout << "empty image error" << std::endl;
		return false;
	}

	if(src.channels() != 1)
	{
		cvtColor(src, src, COLOR_BGR2GRAY);
	}
	if(src.depth() != CV_8U)
	{
		src.convertTo(src, CV_8U);
	}
	return true;
}

bool checkPairs(cv::Mat src1, cv::Mat src2)
{
	return src1.rows == src2.rows && src1.cols == src2.cols;
}

bool checkPoint(int width, int height, int x, int y)
{
	return x >= 0 && x < width && y >= 0 && y < height;
}

float getMatVal(cv::Mat img, int x, int y)
{
	if(img.depth() != CV_32F)
	{
		img.convertTo(img, CV_32F);
	}

	if(checkPoint(img.cols, img.rows, x, y))
	{
		return img.at<float>(y, x);
	}
	return 0;
}

void equalHisImg(cv::Mat src, cv::Mat& dst)
{
	std::vector<cv::Mat> img_split;
	split(src, img_split);

	std::vector<cv::Mat> img_equalDis(3);
	for (int i = 0; i < 3; i++)
	{
		equalizeHist(img_split[i], img_equalDis[i]);
	}
	merge(img_equalDis, dst);
}

void filtImg(cv::Mat src, cv::Mat& dst, int winSize, double eps)
{
	std::vector<cv::Mat> img_split;
	split(src, img_split);

	std::vector<cv::Mat> img_filter(3);
	for (int i = 0; i < 3; i++)
	{
		img_filter[i] = getGuidedFilter(src, img_split[i], winSize, eps);
		normalize(img_filter[i], img_filter[i], 0, 255, NORM_MINMAX);
		img_filter[i].convertTo(img_filter[i], CV_8U);
	}
	merge(img_filter, dst);
}

cv::Mat multiChl_to_oneChl_mul(cv::Mat firstImg, cv::Mat secondImg)
{
	if (firstImg.size != secondImg.size)
		return Mat();

	if (firstImg.depth() == CV_32F && secondImg.depth() == CV_32F
		&& ((firstImg.channels() == 3 && secondImg.channels() == 3)
			|| (firstImg.channels() == 6 && secondImg.channels() == 6)))
	{
		int width = firstImg.cols;
		int height = firstImg.rows;

		Mat res(height, width, CV_32FC1);

		if (firstImg.channels() == 3)
		{
			for (int i = 0; i < width; i++)
			{
				for (int j = 0; j < height; j++)
				{
					res.at<float>(j, i) = firstImg.at<Vec3f>(j, i).dot(secondImg.at<Vec3f>(j, i));
				}
			}
		}
		else if (firstImg.channels() == 6)
		{
			for (int i = 0; i < width; i++)
			{
				for (int j = 0; j < height; j++)
				{
					res.at<float>(j, i) = firstImg.at<Vec6f>(j, i).dot(secondImg.at<Vec6f>(j, i));
				}
			}
		}
		return res;
	}
	return firstImg.mul(secondImg);
}

cv::Mat getGuidedFilter(cv::Mat guidedImg, cv::Mat inputP, int r, double eps)
{
	if (guidedImg.size() != inputP.size())
		return Mat();

	int width = guidedImg.cols;
	int height = guidedImg.rows;

	normalize(guidedImg, guidedImg, 0, 1, NORM_MINMAX, CV_32F);
	normalize(inputP, inputP, 0, 1, NORM_MINMAX, CV_32F);

	Mat meanGuid;
	boxFilter(guidedImg, meanGuid, CV_32F, Size(r, r));
	Mat meanP;
	boxFilter(inputP, meanP, CV_32F, Size(r, r));

	std::vector<Mat> guidedImg_split;
	cv::split(guidedImg, guidedImg_split);

	std::vector<Mat> corrGuid_split;
	Mat corrGuidP;
	for (int i = 0; i < guidedImg_split.size(); i++)
	{
		Mat corrGuid_channel;
		boxFilter(guidedImg_split[i].mul(inputP), corrGuid_channel, CV_32F, Size(r, r));
		corrGuid_split.push_back(corrGuid_channel);
	}
	merge(corrGuid_split, corrGuidP);

	Mat corrGuid;
	boxFilter(guidedImg.mul(guidedImg), corrGuid, CV_32F, Size(r, r));

	Mat varGuid;
	varGuid = corrGuid - meanGuid.mul(meanGuid);

	std::vector<Mat> meanGrid_split;
	cv::split(meanGuid, meanGrid_split);

	std::vector<Mat> guidmul_split;
	Mat meanGuidmulP;
	for (int i = 0; i < meanGrid_split.size(); i++)
	{
		Mat guidmul_channel;
		guidmul_channel = meanGrid_split[i].mul(meanP);
		guidmul_split.push_back(guidmul_channel);
	}
	merge(guidmul_split, meanGuidmulP);

	Mat covGuidP;
	covGuidP = corrGuidP - meanGuidmulP;

	//create image mask for matrix adding integer
	Mat onesMat = Mat::ones(varGuid.size(), varGuid.depth());
	Mat mergeOnes;
	if (varGuid.channels() == 1)
	{
		mergeOnes = onesMat;
	}
	else if (varGuid.channels() == 3)
	{
		std::vector<Mat> oneChannel;
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);

		merge(oneChannel, mergeOnes);
	}
	else if (varGuid.channels() == 6)
	{
		std::vector<Mat> oneChannel;
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);

		merge(oneChannel, mergeOnes);
	}

	Mat a = covGuidP / (varGuid + mergeOnes * eps);
	Mat b = meanP - multiChl_to_oneChl_mul(a, meanGuid);

	boxFilter(a, a, CV_32F, Size(r, r));
	boxFilter(b, b, CV_32F, Size(r, r));

	Mat filteredImg = multiChl_to_oneChl_mul(a, guidedImg) + b;
	return filteredImg;
}

/**
 * \brief compute the disparity using SAD algorithm with fixed window size(FW) and winner takes all(WTA) strategy
 * \param param 
 * \return 
 */
cv::Mat computeSAD_inteOpti(StereoMatchParam param)
{
	if (!checkPairs(param.imgLeft, param.imgRight) 
		|| !checkImg(param.imgLeft) || !checkImg(param.imgRight) 
		|| param.winSize % 2 == 0)
	{
		std::cout << "bad function parameters, please check image and window size." << std::endl;
		return Mat();
	}

	int imgHeight = param.imgLeft.rows;
	int imgWidth = param.imgLeft.cols;
	int numDisp = param.maxDisparity - param.minDisparity + 1;

	if(param.isDispLeft)
	{
		Mat rightBorder;
		copyMakeBorder(param.imgRight, rightBorder, 0, 0, param.maxDisparity, 0, BORDER_REFLECT);

		//optimization:using integral image
		std::vector<Mat> differ_ranges;
		std::vector<Mat> differ_integral;
		for (int i = param.minDisparity; i <= param.maxDisparity; i++)
		{
			Mat differWhole(param.imgLeft.size(), CV_8U, Scalar::all(0));
			absdiff(param.imgLeft(Rect(0, 0, imgWidth, imgHeight)),
				rightBorder(Rect(param.maxDisparity - i, 0, imgWidth, imgHeight)),
				differWhole);
			differ_ranges.push_back(differWhole);

			Mat differWholeInte;
			integral(differWhole, differWholeInte, CV_32F);
			differ_integral.push_back(differWholeInte);
		}

		int halfWinSize = param.winSize / 2;

		Mat disparityMap(imgHeight, imgWidth, CV_8U, Scalar::all(0));
#pragma omp parallel for
		for (int j = 0; j < imgHeight; j++)
		{
			for (int i = 0; i < imgWidth; i++)
			{
				Mat allCost(1, numDisp, CV_32F, Scalar::all(0));
				for (int k = 0; k < numDisp; k++)
				{
					allCost.at<float>(k) = getMatVal(differ_integral[k],i + halfWinSize, j + halfWinSize)
						- getMatVal(differ_integral[k], i + halfWinSize, j - halfWinSize)
						- getMatVal(differ_integral[k], i - halfWinSize, j + halfWinSize)
						+ getMatVal(differ_integral[k], i - halfWinSize, j - halfWinSize);
				}
				Point minLoc;
				minMaxLoc(allCost, NULL, NULL, &minLoc, NULL);
				disparityMap.at<char>(j, i) = minLoc.x + param.minDisparity;
			}
		}
		return disparityMap;
	}
	else
	{
		Mat leftBorder;
		copyMakeBorder(param.imgLeft, leftBorder, 0, 0, 0, param.maxDisparity, BORDER_REFLECT);

		//optimization:using integral image
		std::vector<Mat> differ_ranges;
		std::vector<Mat> differ_integral;
		for (int i = param.minDisparity; i <= param.maxDisparity; i++)
		{
			Mat differWhole(param.imgRight.size(), CV_8U, Scalar::all(0));
			absdiff(param.imgRight(Rect(0, 0, imgWidth, imgHeight)),
				leftBorder(Rect(i, 0, imgWidth, imgHeight)),
				differWhole);
			differ_ranges.push_back(differWhole);

			Mat differWholeInte;
			integral(differWhole, differWholeInte, CV_32F);
			differ_integral.push_back(differWholeInte);
		}

		int halfWinSize = param.winSize / 2;

		Mat disparityMap(imgHeight, imgWidth, CV_8U, Scalar::all(0));
#pragma omp parallel for
		for (int j = 0; j < imgHeight; j++)
		{
			for (int i = 0; i < imgWidth; i++)
			{
				Mat allCost(1, numDisp, CV_32F, Scalar::all(0));
				for (int k = 0; k < numDisp; k++)
				{
					allCost.at<float>(k) = getMatVal(differ_integral[k], i + halfWinSize, j + halfWinSize)
						- getMatVal(differ_integral[k], i + halfWinSize, j - halfWinSize)
						- getMatVal(differ_integral[k], i - halfWinSize, j + halfWinSize)
						+ getMatVal(differ_integral[k], i - halfWinSize, j - halfWinSize);
				}
				Point minLoc;
				minMaxLoc(allCost, NULL, NULL, &minLoc, NULL);
				disparityMap.at<char>(j, i) = minLoc.x + param.minDisparity;
			}
		}
		return disparityMap;
	}
}

cv::Mat computeSAD_BFOpti(StereoMatchParam param)
{
	if (!checkPairs(param.imgLeft, param.imgRight)
		|| !checkImg(param.imgLeft) || !checkImg(param.imgRight)
		|| param.winSize % 2 == 0)
	{
		std::cout << "bad function parameters, please check image and window size." << std::endl;
		return Mat();
	}

	int imgHeight = param.imgLeft.rows;
	int imgWidth = param.imgLeft.cols;
	int numDisp = param.maxDisparity - param.minDisparity + 1;

	if (param.isDispLeft)
	{
		Mat rightBorder;
		copyMakeBorder(param.imgRight, rightBorder, 0, 0, param.maxDisparity, 0, BORDER_REFLECT);

		//optimization:using integral image
		std::vector<Mat> differ_ranges;
		std::vector<Mat> differ_BF;
		for (int i = param.minDisparity; i <= param.maxDisparity; i++)
		{
			Mat differWhole(param.imgLeft.size(), CV_8U, Scalar::all(0));
			absdiff(param.imgLeft(Rect(0, 0, imgWidth, imgHeight)),
				rightBorder(Rect(param.maxDisparity - i, 0, imgWidth, imgHeight)),
				differWhole);
			differ_ranges.push_back(differWhole);

			differWhole.convertTo(differWhole, CV_32FC1);
			Mat differWholeBF;
			sqrBoxFilter(differWhole, differWholeBF, -1, 
				Size(param.winSize, param.winSize), Point(-1, -1), false);
			differ_BF.push_back(differWholeBF);
		}

		Mat disparityMap(imgHeight, imgWidth, CV_8U, Scalar::all(0));
#pragma omp parallel for
		for (int j = 0; j < imgHeight; j++)
		{
			for (int i = 0; i < imgWidth; i++)
			{
				Mat allCost(1, numDisp, CV_32F, Scalar::all(0));
				for (int k = 0; k < numDisp; k++)
				{
					allCost.at<float>(k) = (float)differ_BF[k].at<double>(j, i);;
				}
				Point minLoc;
				minMaxLoc(allCost, NULL, NULL, &minLoc, NULL);
				disparityMap.at<char>(j, i) = minLoc.x + param.minDisparity;
			}
		}
		return disparityMap;
	}
	else
	{
		Mat leftBorder;
		copyMakeBorder(param.imgLeft, leftBorder, 0, 0, 0, param.maxDisparity, BORDER_REFLECT);

		//optimization:using integral image
		std::vector<Mat> differ_ranges;
		std::vector<Mat> differ_BF;
		for (int i = param.minDisparity; i <= param.maxDisparity; i++)
		{
			Mat differWhole(param.imgRight.size(), CV_8U, Scalar::all(0));
			absdiff(param.imgRight(Rect(0, 0, imgWidth, imgHeight)),
				leftBorder(Rect(i, 0, imgWidth, imgHeight)),
				differWhole);
			differ_ranges.push_back(differWhole);

			Mat differWholeBF;
			sqrBoxFilter(differWhole, differWholeBF, -1,
				Size(param.winSize, param.winSize), Point(-1, -1), false);
			differ_BF.push_back(differWholeBF);
		}

		int halfWinSize = param.winSize / 2;

		Mat disparityMap(imgHeight, imgWidth, CV_8U, Scalar::all(0));
#pragma omp parallel for
		for (int j = 0; j < imgHeight; j++)
		{
			for (int i = 0; i < imgWidth; i++)
			{
				Mat allCost(1, numDisp, CV_32F, Scalar::all(0));
				for (int k = 0; k < numDisp; k++)
				{
					allCost.at<float>(k) = (float)differ_BF[k].at<float>(j, i);
				}
				Point minLoc;
				minMaxLoc(allCost, NULL, NULL, &minLoc, NULL);
				disparityMap.at<char>(j, i) = minLoc.x + param.minDisparity;
			}
		}
		return disparityMap;
	}
}

/**
 * \brief original census computing algorithm, the allowed window size is 3*3
 * \param src 
 * \param dst 
 */
void countCensusImg(cv::Mat src, cv::Mat& dst)
{
	if(src.channels() != 1)
	{
		cvtColor(src, src, COLOR_BGR2GRAY);
	}

	if(src.type() != CV_8U)
	{
		src.convertTo(src, CV_8U);
	}

	dst.create(src.size(), CV_8UC1);
	dst.setTo(0);
#pragma omp parallel for
	for(int y = 1; y < src.rows - 1; y++)
	{
		for(int x = 1; x < src.cols - 1; x++)
		{
			uchar center = src.at<uchar>(y, x);
			uchar censusVal = 0;
			censusVal |= (src.at<uchar>(y - 1, x - 1) >= center) << 7;
			censusVal |= (src.at<uchar>(y - 1, x) >= center) << 6;
			censusVal |= (src.at<uchar>(y - 1, x + 1) >= center) << 5;
			censusVal |= (src.at<uchar>(y, x + 1) >= center) << 4;
			censusVal |= (src.at<uchar>(y + 1, x + 1) >= center) << 3;
			censusVal |= (src.at<uchar>(y + 1, x) >= center) << 2;
			censusVal |= (src.at<uchar>(y + 1, x - 1) >= center) << 1;
			censusVal |= (src.at<uchar>(y, x - 1) >= center) << 0;
			dst.at<uchar>(y, x) = censusVal;
		}
	}
}

/**
 * \brief census computing with circle support window
 * \param src 
 * \param dst 
 * \param radius :the radius of the neighboring circle centered at the anchor pixel
 * \param samplePtNum :the number of points on the neighboring circle taken into consideration
 */
void countCensusImg_circle(cv::Mat src, cv::Mat& dst, int radius, int samplePtNum)
{
	if (src.channels() != 1)
	{
		cvtColor(src, src, COLOR_BGR2GRAY);
	}

	if (src.type() != CV_8U)
	{
		src.convertTo(src, CV_8U);
	}

	dst.create(src.size(), CV_8UC1);
	dst.setTo(0);
#pragma omp parallel for
	for (int n = 0; n < samplePtNum; n++)
	{
		float x = static_cast<float>(radius * cos(2.0 * CV_PI * n / static_cast<float>(samplePtNum)));
		float y = static_cast<float>(-radius * sin(2.0 * CV_PI * n / static_cast<float>(samplePtNum)));

		int fx = static_cast<int>(floor(x));  //floor()向下取整
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x)); //ceil()向上取整
		int cy = static_cast<int>(ceil(y));

		float tx = x - fx; //将坐标映射到0-1之间
		float ty = y - fy;

		float w1 = (1 - tx) * (1 - ty);//计算插值权重
		float w2 = tx * (1 - ty);
		float w3 = (1 - tx) * ty;
		float w4 = tx * ty;

		for (int i = radius; i < src.rows - radius; i++)
		{
			for (int j = radius; j < src.cols - radius; j++)
			{
				float t = static_cast<float>(w1 * src.at<uchar>(i + fy, j + fx) + w2 * src.at<uchar>(i + fy, j + cx)
					+ w3 * src.at<uchar>(i + cy, j + fx) + w4 * src.at<uchar>(i + cy, j + cx));
				dst.at<uchar>(i - radius, j - radius) += ((t > src.at<uchar>(i, j))
					|| (abs(t - src.at<uchar>(i, j)) < std::numeric_limits<float>::epsilon())) << n;
			}
		}
	}
}

/**
 * \brief count rotation invariant census value for each pixel
 * \param src 
 * \param dst 
 */
void countCensusImg_rotationInv(cv::Mat src, cv::Mat& dst)
{
	if (src.channels() != 1)
	{
		cvtColor(src, src, COLOR_BGR2GRAY);
	}

	if (src.type() != CV_8U)
	{
		src.convertTo(src, CV_8U);
	}

	uchar RITable[256];
	{
		int temp, val;
		for (int i = 0; i < 256; i++)
		{
			val = i;
			for (int j = 0; j < 7; j++)
			{
				temp = i >> 1;
				if (val > temp)
					val = temp;
			}
			RITable[i] = val;
		}
	}

	dst.create(src.size(), CV_8UC1);
	dst.setTo(0);
#pragma omp parallel for
	for (int i = 1; i < src.rows - 1; i++)
	{
		for (int j = 1; j < src.cols - 1; j++)
		{
			uchar center = src.at<uchar>(i, j);
			uchar code = 0;
			code |= (src.at<uchar>(i - 1, j - 1) >= center) << 7;
			code |= (src.at<uchar>(i - 1, j) >= center) << 6;
			code |= (src.at<uchar>(i - 1, j + 1) >= center) << 5;
			code |= (src.at<uchar>(i, j + 1) >= center) << 4;
			code |= (src.at<uchar>(i + 1, j + 1) >= center) << 3;
			code |= (src.at<uchar>(i + 1, j) >= center) << 2;
			code |= (src.at<uchar>(i + 1, j - 1) >= center) << 1;
			code |= (src.at<uchar>(i, j - 1) >= center) << 0;
			dst.at<uchar>(i, j) = RITable[code];
		}
	}
}

/**
 * \brief 计算二进制跳变次数
 * \param i 
 * \return 
 */
int hopCount(uchar i)
{
	uchar a[8] = { 0 };
	int cnt = 0;
	int k = 7;
	while (k)
	{
		a[k] = i & 1;
		i = i >> 1;
		k--;
	}

	for (int j = 0; j < 8; j++)
	{
		if (a[j] != a[j+1 == 8 ? 0 : j+1])
			cnt++;
	}
	return cnt;
}

/**
 * \brief follow uniform LBP algorithm, have uniform census algorithm
 * \param src 
 * \param dst 
 */
void countCensusImg_uniform(cv::Mat src, cv::Mat& dst)
{
	if (src.channels() != 1)
	{
		cvtColor(src, src, COLOR_BGR2GRAY);
	}

	if (src.type() != CV_8U)
	{
		src.convertTo(src, CV_8U);
	}

	uchar UPTable[256];
	memset(UPTable, 0, 256 * sizeof(uchar));
	uchar temp = 1;
	for (int i = 0; i < 256; i++)
	{
		if (hopCount(i) <= 2)
		{
			UPTable[i] = temp;
			temp++;
		}
	}

	dst.create(src.rows, src.cols, CV_8UC1);
	dst.setTo(0);
#pragma omp parallel for
	for (int i = 1; i < src.rows - 1; i++)
	{
		for (int j = 1; j < src.cols - 1; j++)
		{
			uchar center = src.at<uchar>(i, j);
			uchar code = 0;
			code |= (src.at<uchar>(i - 1, j - 1) >= center) << 7;
			code |= (src.at<uchar>(i - 1, j) >= center) << 6;
			code |= (src.at<uchar>(i - 1, j + 1) >= center) << 5;
			code |= (src.at<uchar>(i, j + 1) >= center) << 4;
			code |= (src.at<uchar>(i + 1, j + 1) >= center) << 3;
			code |= (src.at<uchar>(i + 1, j) >= center) << 2;
			code |= (src.at<uchar>(i + 1, j - 1) >= center) << 1;
			code |= (src.at<uchar>(i, j - 1) >= center) << 0;
			dst.at<uchar>(i, j) = UPTable[code];
		}
	}
}

/**
 * \brief census value from multi-scale block LBP
 * \param src 
 * \param dst 
 * \param scale :window size
 */
void countCensusImg_multiScale(cv::Mat src, cv::Mat& dst, int scale)
{
	int cellSize = scale / 3;
	int offset = cellSize / 2;

	if (src.channels() != 1)
	{
		cvtColor(src, src,COLOR_BGR2GRAY);
	}

	Mat cellImg(src.rows, src.cols, src.type());
	cellImg.setTo(0);
#pragma omp parallel for
	for (int i = offset; i < src.rows - offset; i++)
	{
		for (int j = offset; j < src.cols - offset; j++)
		{
			int temp = 0;
			for (int m = -offset; m < offset + 1; m++)
			{
				for (int n = -offset; n < offset + 1; n++)
				{
					temp += src.at<uchar>(i + n, j + m);
				}
			}
			temp /= (cellSize * cellSize);
			cellImg.at<uchar>(i, j) = uchar(temp);
		}
	}
	countCensusImg(cellImg, dst);
}

/**
 * \brief census value from statistically effective multi-scale block LBP
 * \param src 
 * \param dst 
 * \param scale 
 */
void countCensusImg_multiScale2(cv::Mat src, cv::Mat& dst, int scale)
{
	countCensusImg_multiScale(src, dst, scale);

	Mat histImg;
	int histSize = 256;
	float range[] = { float(0), float(255) };
	const float* ranges = { range };
	calcHist(&dst, 1, 0, Mat(), histImg, 1, &histSize, &ranges, true, false);
	histImg.reshape(1, 1);
	std::vector<float> histVector(histImg.rows * histImg.cols);
	uchar table[256];
	memset(table, 64, 256);
	if (histImg.isContinuous())
	{
		histVector.assign((float*)histImg.datastart, (float*)histImg.dataend); //将直方图histImg变为vector向量histVector
		std::vector<float> histVectorCopy(histVector);
		sort(histVector.begin(), histVector.end(), std::greater<float>()); //对LBP特征值的数量进行排序，降序排序
		for (int i = 0; i < 63; i++)
		{
			for (int j = 0; j < histVectorCopy.size(); j++)
			{
				if (histVectorCopy[j] == histVector[i])
				{
					table[j] = i;
				}
			}
		}
	}
#pragma omp parallel for
	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			dst.at<uchar>(i, j) = table[dst.at<uchar>(i, j)];
		}
	}
}

/**
 * \brief algorithm improvement from "research on stereo vision algorithm for rescue robot"
 * \param src 
 * \param dst 
 * \param winSize 
 */
void countCensusImg_2017(cv::Mat src, cv::Mat& dst, int winSize)
{
	if (src.channels() != 1)
	{
		cvtColor(src, src, COLOR_BGR2GRAY);
	}

	if (src.type() != CV_32F)
	{
		src.convertTo(src, CV_32F);
	}

	int halfSize = winSize / 2;
	cv::Mat srcBorder;
	copyMakeBorder(src, srcBorder, halfSize, halfSize, halfSize, halfSize, BORDER_REFLECT);

	dst.create(src.size(), CV_32SC1);
	dst.setTo(0);
#pragma omp parallel for
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			int center = src.at<float>(y, x);
			cv::Mat curWin = srcBorder(Rect(x, y, winSize, winSize));
			curWin = curWin - center;
			cv::Mat winBit;
			threshold(curWin, winBit, -1, 1, THRESH_BINARY);
			winBit.convertTo(winBit, CV_32S);

			int censusVal = 0;
			int bitFlag = 0;
			// row
			{
				Mat rowB, rowE, rowBitDst;
				rowB = winBit.row(0);
				rowE = winBit.row(winSize-1);
				bitwise_xor(rowB, rowE, rowBitDst);
				for (int i = 0; i < winSize; i++)
				{
					censusVal |= rowBitDst.at<int>(0, i) << bitFlag;
					bitFlag++;
				}
			}
			// col
			{
				int i = 2;
				while(i < winSize)
				{
					censusVal |= (winBit.at<int>(i, 0) ^ winBit.at<int>(i, winSize-1)) << bitFlag;
					bitFlag++;
					i += 2;
				}
			}
			// inner left to right
			{
				int i = 1;
				while(i < winSize / 2)
				{
					censusVal |= (winBit.at<int>(i, i) ^ winBit.at<int>(winSize - 1 - i, winSize - 1 - i)) << bitFlag;
					bitFlag++;
					i++;
				}
			}
			// inner right to left
			{
				int i = 1;
				while (i < winSize / 2)
				{
					censusVal |= (winBit.at<int>(i, winSize - 1 - i) ^ winBit.at<int>(winSize - 1 - i, i)) << bitFlag;
					bitFlag++;
					i++;
				}
			}

			dst.at<int>(y, x) = censusVal;
		}
	}
}

/**
 * \brief count the humming distance between corresponding pixels in two images
 * \param src1 
 * \param src2 
 * \param dst 
 */
void countHummingDist(cv::Mat src1, cv::Mat src2, cv::Mat& dst)
{
	cv::Mat xorImg;
	bitwise_xor(src1, src2, xorImg);
	xorImg.convertTo(xorImg, CV_32S);

	dst.create(xorImg.size(), CV_32SC1);
	int countTable[16] =
	{
		0, 1, 1, 2,
		1, 2, 2, 3,
		1, 2, 2, 3,
		2, 3, 3, 4
	};

#pragma omp parallel for
	for(int i = 0; i < xorImg.rows; i++)
	{
		for(int j = 0; j < xorImg.cols; j++)
		{
			int count = 0;
			int num = xorImg.at<int>(i, j);
			while(num)
			{
				count += countTable[num & 0xF];
				num >>= 4;
			}
			dst.at<int>(i, j) = count;
		}
	}
}

/**
 * \brief traditional census transform algorithm for stereo matching
 * \param param 
 * \return 
 */
cv::Mat censusStereo(StereoMatchParam param, CENSUS_ALGORITHM method)
{
	if (!checkPairs(param.imgLeft, param.imgRight)
		|| !checkImg(param.imgLeft) || !checkImg(param.imgRight)
		|| param.winSize % 2 == 0)
	{
		std::cout << "bad function parameters, please check image and window size." << std::endl;
		return Mat();
	}

	int imgHeight = param.imgLeft.rows;
	int imgWidth = param.imgLeft.cols;
	int numDisp = param.maxDisparity - param.minDisparity + 1;

	if(param.isDispLeft)
	{
		cv::Mat rightBorder;
		copyMakeBorder(param.imgRight, rightBorder, 0, 0, param.maxDisparity, 0, BORDER_REFLECT);

		cv::Mat censusLeft, censusRight;
		switch (method)
		{
		case BASIC_CENSUS:
		{
			std::future<void> ft1 = std::async(std::launch::async, [&]
			{
				countCensusImg(param.imgLeft, censusLeft);
			});
			std::future<void> ft2 = std::async(std::launch::async, [&]
			{
				countCensusImg(rightBorder, censusRight);
			});
			ft1.wait();
			ft2.wait();
		}
			break;
		case CIRCLE_CENSUS:
		{
			std::future<void> ft1 = std::async(std::launch::async, [&]
			{
				countCensusImg_circle(param.imgLeft, censusLeft, 5);
			});
			std::future<void> ft2 = std::async(std::launch::async, [&]
			{
				countCensusImg_circle(rightBorder, censusRight, 5);
			});
			ft1.wait();
			ft2.wait();
		}
			break;
		case ROTATION_INVARIANT_CENSUS:
		{
			std::future<void> ft1 = std::async(std::launch::async, [&]
			{
				countCensusImg_rotationInv(param.imgLeft, censusLeft);
			});
			std::future<void> ft2 = std::async(std::launch::async, [&]
			{
				countCensusImg_rotationInv(rightBorder, censusRight);
			});
			ft1.wait();
			ft2.wait();
		}
			break;
		case UNIFORM_CENSUS:
		{
			std::future<void> ft1 = std::async(std::launch::async, [&]
			{
				countCensusImg_uniform(param.imgLeft, censusLeft);
			});
			std::future<void> ft2 = std::async(std::launch::async, [&]
			{
				countCensusImg_uniform(rightBorder, censusRight);
			});
			ft1.wait();
			ft2.wait();
		}
			break;
		case MULTISCALE_CENSUS:
		{
			std::future<void> ft1 = std::async(std::launch::async, [&]
			{
				countCensusImg_multiScale(param.imgLeft, censusLeft, 16);
			});
			std::future<void> ft2 = std::async(std::launch::async, [&]
			{
				countCensusImg_multiScale(rightBorder, censusRight, 16);
			});
			ft1.wait();
			ft2.wait();
		}
			break;
		case STATISTIC_MULTISCALE_CENSUS:
		{
			std::future<void> ft1 = std::async(std::launch::async, [&]
			{
				countCensusImg_multiScale2(param.imgLeft, censusLeft, param.winSize);
			});
			std::future<void> ft2 = std::async(std::launch::async, [&]
			{
				countCensusImg_multiScale2(rightBorder, censusRight, param.winSize);
			});
			ft1.wait();
			ft2.wait();
		}
			break;
		case CENSUS_2017:
		{
			std::future<void> ft1 = std::async(std::launch::async, [&]
			{
				countCensusImg_2017(param.imgLeft, censusLeft, param.winSize);
			});
			std::future<void> ft2 = std::async(std::launch::async, [&]
			{
				countCensusImg_2017(rightBorder, censusRight, param.winSize);
			});
			ft1.wait();
			ft2.wait();
		}
			break;
		}

		std::vector<Mat> dist;
		std::vector<Mat> distIntegry;
		for(int i = param.minDisparity; i <= param.maxDisparity; i++)
		{
			cv::Mat distImg;
			countHummingDist(censusLeft,
				censusRight(Rect(param.maxDisparity - i, 0, imgWidth, imgHeight)), distImg);
			dist.push_back(distImg);
			distImg.convertTo(distImg, CV_8UC1);

			cv::Mat distIntegry_;
			distIntegry_ = getGuidedFilter(param.imgLeft, distImg, param.winSize, 1e-6);
			//sqrBoxFilter(distImg, distIntegry_, -1,
			//	Size(param.winSize, param.winSize), Point(-1, -1), true);
			distIntegry.push_back(distIntegry_);
		}

		Mat disparityMap(imgHeight, imgWidth, CV_64F, Scalar::all(0));
#pragma omp parallel for
		for(int y = 0; y < imgHeight; y++)
		{
			for(int x = 0; x < imgWidth; x++)
			{
				Mat allCost(1, numDisp, CV_32F, Scalar::all(0));
				for (int k = 0; k < numDisp; k++)
				{
					allCost.at<float>(k) = (float)distIntegry[k].at<float>(y, x);
				}
				Point minLoc;
				minMaxLoc(allCost, NULL, NULL, &minLoc, NULL);
				disparityMap.at<double>(y, x) = minLoc.x + param.minDisparity;
			}
		}
		return disparityMap;
	}
	else
	{
		cv::Mat leftBorder;
		copyMakeBorder(param.imgLeft, leftBorder, 0, 0, 0, param.maxDisparity, BORDER_REFLECT);

		cv::Mat censusLeft, censusRight;
		switch (method)
		{
		case BASIC_CENSUS:
		{
			std::future<void> ft1 = std::async(std::launch::async, [&]
			{
				countCensusImg(leftBorder, censusLeft);
			});
			std::future<void> ft2 = std::async(std::launch::async, [&]
			{
				countCensusImg(param.imgRight, censusRight);
			});
			ft1.wait();
			ft2.wait();
		}
			break;
		case CIRCLE_CENSUS:
		{
			std::future<void> ft1 = std::async(std::launch::async, [&]
			{
				countCensusImg_circle(leftBorder, censusLeft, param.winSize);
			});
			std::future<void> ft2 = std::async(std::launch::async, [&]
			{
				countCensusImg_circle(param.imgRight, censusRight, param.winSize);
			});
			ft1.wait();
			ft2.wait();
		}
			break;
		case ROTATION_INVARIANT_CENSUS:
		{
			std::future<void> ft1 = std::async(std::launch::async, [&]
			{
				countCensusImg_rotationInv(leftBorder, censusLeft);
			});
			std::future<void> ft2 = std::async(std::launch::async, [&]
			{
				countCensusImg_rotationInv(param.imgRight, censusRight);
			});
			ft1.wait();
			ft2.wait();
		}
			break;
		case UNIFORM_CENSUS:
		{
			std::future<void> ft1 = std::async(std::launch::async, [&]
			{
				countCensusImg_uniform(leftBorder, censusLeft);
			});
			std::future<void> ft2 = std::async(std::launch::async, [&]
			{
				countCensusImg_uniform(param.imgRight, censusRight);
			});
			ft1.wait();
			ft2.wait();
		}
			break;
		case MULTISCALE_CENSUS:
		{
			std::future<void> ft1 = std::async(std::launch::async, [&]
			{
				countCensusImg_multiScale(leftBorder, censusLeft, param.winSize);
			});
			std::future<void> ft2 = std::async(std::launch::async, [&]
			{
				countCensusImg_multiScale(param.imgRight, censusRight, param.winSize);
			});
			ft1.wait();
			ft2.wait();
		}
			break;
		case STATISTIC_MULTISCALE_CENSUS:
		{
			std::future<void> ft1 = std::async(std::launch::async, [&]
			{
				countCensusImg_multiScale2(leftBorder, censusLeft, param.winSize);
			});
			std::future<void> ft2 = std::async(std::launch::async, [&]
			{
				countCensusImg_multiScale2(param.imgRight, censusRight, param.winSize);
			});
			ft1.wait();
			ft2.wait();
		}
			break;
		case CENSUS_2017:
		{
			std::future<void> ft1 = std::async(std::launch::async, [&]
			{
				countCensusImg_2017(leftBorder, censusLeft, param.winSize);
			});
			std::future<void> ft2 = std::async(std::launch::async, [&]
			{
				countCensusImg_2017(param.imgRight, censusRight, param.winSize);
			});
			ft1.wait();
			ft2.wait();
		}
			break;
		}

		std::vector<Mat> dist;
		std::vector<Mat> distIntegry;
		for (int i = param.minDisparity; i <= param.maxDisparity; i++)
		{
			cv::Mat distImg;
			countHummingDist(censusRight,
				censusLeft(Rect(i, 0, imgWidth, imgHeight)), distImg);
			dist.push_back(distImg);

			cv::Mat distIntegry_;
			distIntegry_ = getGuidedFilter(param.imgRight, distImg, param.winSize, 1e-6);

			//sqrBoxFilter(distImg, distIntegry_, -1,
			//	Size(param.winSize, param.winSize), Point(-1, -1), false);
			distIntegry.push_back(distIntegry_);
		}

		Mat disparityMap(imgHeight, imgWidth, CV_64F, Scalar::all(0));
#pragma omp parallel for
		for (int y = 0; y < imgHeight; y++)
		{
			for (int x = 0; x < imgWidth; x++)
			{
				Mat allCost(1, numDisp, CV_32F, Scalar::all(0));
				for (int k = 0; k < numDisp; k++)
				{
					allCost.at<float>(k) = (float)distIntegry[k].at<float>(y, x);
				}
				Point minLoc;
				minMaxLoc(allCost, NULL, NULL, &minLoc, NULL);
				disparityMap.at<double>(y, x) = minLoc.x + param.minDisparity;
			}
		}
		return disparityMap;
	}
}
