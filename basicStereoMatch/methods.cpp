#include "methods.h"
#include <iso646.h>

using namespace cv;

/**
 * /brief check if the imput image is CV_8UC1 type
 * /param src 
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

/**
 * /brief compute the disparity using SAD algorithm with fixed window size(FW) and winner takes all(WTA) strategy
 * /param param 
 * /return 
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
 * /brief original census computing algorithm, the allowed window size is 3*3
 * /param src 
 * /param dst 
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
 * /brief count the humming distance between corresponding pixels in two images
 * /param src1 
 * /param src2 
 * /param dst 
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
 * /brief traditional census transform algorithm for stereo matching
 * /param param 
 * /return 
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
			countCensusImg(param.imgLeft, censusLeft);
			countCensusImg(rightBorder, censusRight);
			break;
		case CIRCLE_CENSUS:
			countCensusImg_circle(param.imgLeft, censusLeft, param.winSize);
			countCensusImg_circle(rightBorder, censusRight, param.winSize);
			break;
		case ROTATION_INVARIANT_CENSUS:
			countCensusImg_rotationInv(param.imgLeft, censusLeft);
			countCensusImg_rotationInv(rightBorder, censusRight);
			break;
		case UNIFORM_CENSUS:
			countCensusImg_uniform(param.imgLeft, censusLeft);
			countCensusImg_uniform(rightBorder, censusRight);
			break;
		case MULTISCALE_CENSUS:
			countCensusImg_multiScale(param.imgLeft, censusLeft, param.winSize);
			countCensusImg_multiScale(rightBorder, censusRight, param.winSize);
			break;
		case STATISTIC_MULTISCALE_CENSUS:
			countCensusImg_multiScale2(param.imgLeft, censusLeft, param.winSize);
			countCensusImg_multiScale2(rightBorder, censusRight, param.winSize);
			break;
		case CENSUS_2017:
			countCensusImg_2017(param.imgLeft, censusLeft, param.winSize);
			countCensusImg_2017(rightBorder, censusRight, param.winSize);
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
			sqrBoxFilter(distImg, distIntegry_, -1,
				Size(param.winSize, param.winSize), Point(-1, -1), true);
			distIntegry.push_back(distIntegry_);
		}

		Mat disparityMap(imgHeight, imgWidth, CV_8U, Scalar::all(0));
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
				disparityMap.at<char>(y, x) = minLoc.x + param.minDisparity;
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
			countCensusImg(leftBorder, censusLeft);
			countCensusImg(param.imgRight, censusRight);
			break;
		case CIRCLE_CENSUS:
			countCensusImg_circle(leftBorder, censusLeft, param.winSize);
			countCensusImg_circle(param.imgRight, censusRight, param.winSize);
			break;
		case ROTATION_INVARIANT_CENSUS:
			countCensusImg_rotationInv(leftBorder, censusLeft);
			countCensusImg_rotationInv(param.imgRight, censusRight);
			break;
		case UNIFORM_CENSUS:
			countCensusImg_uniform(leftBorder, censusLeft);
			countCensusImg_uniform(param.imgRight, censusRight);
			break;
		case MULTISCALE_CENSUS:
			countCensusImg_multiScale(leftBorder, censusLeft, param.winSize);
			countCensusImg_multiScale(param.imgRight, censusRight, param.winSize);
			break;
		case STATISTIC_MULTISCALE_CENSUS:
			countCensusImg_multiScale2(leftBorder, censusLeft, param.winSize);
			countCensusImg_multiScale2(param.imgRight, censusRight, param.winSize);
			break;
		case CENSUS_2017:
			countCensusImg_2017(leftBorder, censusLeft, param.winSize);
			countCensusImg_2017(param.imgRight, censusRight, param.winSize);
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
			sqrBoxFilter(distImg, distIntegry_, -1,
				Size(param.winSize, param.winSize), Point(-1, -1), false);
			distIntegry.push_back(distIntegry_);
		}

		Mat disparityMap(imgHeight, imgWidth, CV_8U, Scalar::all(0));
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
				disparityMap.at<char>(y, x) = minLoc.x + param.minDisparity;
			}
		}
		return disparityMap;
	}
}
