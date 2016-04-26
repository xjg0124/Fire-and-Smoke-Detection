//-----------------------------------【】----------------------------  
//     代码作用：分块帧差运动检测and火焰颜色模型    
//     OpenCV源代码版本：3.0.0    
//     by：xuejiguang
//     Time:2016/04/25	
//-------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
using namespace std;
using namespace cv;

//计算二值图像素大于0的个数
int bSums(Mat src)
{
	int counter = 0;
	//迭代器访问像素点
	Mat_<uchar>::iterator it = src.begin<uchar>();
	Mat_<uchar>::iterator itend = src.end<uchar>();
	for (; it != itend; ++it)
	{
		if ((*it)>0) counter += 1;//二值化后，像素点是0或者255
	}
	return counter;
}
//图像块结构体
struct FraRIO
{
	Mat frameRIO;
	int point_x;
	int point_y;
	bool RIO_flag;

};
//每帧图像分块
vector<FraRIO>  DivFra(Mat &image, int width, int height)
{
	char name = 1;
	int m, n;
	m = image.rows / height;
	n = image.cols / width;
	vector<FraRIO> FraRIO_Out;
	FraRIO temFraRIO;
	for (int j = 0; j<m; j++)
	{
		for (int i = 0; i<n; i++)
		{
			Mat temImage(height, width, CV_8UC3, cv::Scalar(0, 0, 0));//
			Mat imageROI = image(Rect(i*width, j*height, temImage.cols, temImage.rows));//rect(x, y, width, height)选定感兴趣区域
			addWeighted(temImage, 1.0, imageROI, 1.0, 0., temImage);//复制扫描出的边界内数据

			temFraRIO.frameRIO = temImage.clone();
			temFraRIO.point_x = i*width;
			temFraRIO.point_y = j*height;
			FraRIO_Out.push_back(temFraRIO);
		}
	}
	return FraRIO_Out;
}

void ImgMean(float& c1, float& c2, float& c3, Mat pImg)
{
	int nPixel = pImg.rows*pImg.cols;	// number of pixels in image
	c1 = 0; c2 = 0; c3 = 0;

	//累加各通道的值
	MatConstIterator_<Vec3b> it = pImg.begin<Vec3b>();
	MatConstIterator_<Vec3b> itend = pImg.end<Vec3b>();

	while (it != itend)
	{
		c1 += (*it)[0];
		c2 += (*it)[1];
		c3 += (*it)[2];
		it++;

	}
	//累加各通道的值

	c1 = c1 / nPixel;
	c2 = c2 / nPixel;
	c3 = c3 / nPixel;
}

Mat ColorDet(Mat srcImg){
	Mat m_pcurFrameYCrCb;
	Mat pImgResult;

	m_pcurFrameYCrCb.create(srcImg.size(), srcImg.type());
	pImgResult.create(srcImg.size(), srcImg.type());
	//cvtColor(srcImg, m_pcurFrameYCrCb, CV_BGR2YCrCb);
	m_pcurFrameYCrCb = srcImg.clone();

	float yy_mean = 0, cr_mean = 0, cb_mean = 0;
	ImgMean(cb_mean, cr_mean, yy_mean, m_pcurFrameYCrCb);
	uchar r = 0, g = 0, b = 0;
	uchar yy = 0, cr = 0, cb = 0;


	for (int i = 0; i<srcImg.rows; i++){
		for (int j = 0; j<srcImg.cols; j++){

			b = srcImg.at<Vec3b>(i, j)[0];
			g = srcImg.at<Vec3b>(i, j)[1];
			r = srcImg.at<Vec3b>(i, j)[2];

			cb = m_pcurFrameYCrCb.at<Vec3b>(i, j)[0];
			cr = m_pcurFrameYCrCb.at<Vec3b>(i, j)[1];
			yy = m_pcurFrameYCrCb.at<Vec3b>(i, j)[2];

			if (r>120 && yy>cb&&cr>cb&&yy>yy_mean && (abs(cb - cr)>40))
				//if (r>12 && r>g && g>b && yy>cb&&cr>cb && cr>cr_mean  && cb<cb_mean && yy>yy_mean && (abs(cb - cr)>40))
			{
				pImgResult.at<Vec3b>(i, j)[0] = 255;
				pImgResult.at<Vec3b>(i, j)[1] = 255;
				pImgResult.at<Vec3b>(i, j)[2] = 255;

			}
			else
			{
				pImgResult.at<Vec3b>(i, j)[0] = 0;
				pImgResult.at<Vec3b>(i, j)[1] = 0;
				pImgResult.at<Vec3b>(i, j)[2] = 0;

			}

		}
	}
	//彩色图转灰度图
	cvtColor(pImgResult, pImgResult, COLOR_BGR2GRAY);

	return pImgResult;
}

int main()
{
	//读取视频
	VideoCapture capture("C:/Users/Administrator/Desktop/FireCollect/pos/fire indoor/fire_indoor_16.avi");
	if (!capture.isOpened())
		return -1;

	//
	double rate = capture.get(CV_CAP_PROP_FPS);
	int delay = 1000 / rate;

	//定义当前帧和临时帧
	Mat frame, tem_frame;

	//定义前一帧和当前帧分割后的结构体数组
	vector<FraRIO> framePro_RIO;
	vector<FraRIO> frame_RIO;

	bool flag = false;

	namedWindow("image", CV_WINDOW_AUTOSIZE);

	while (capture.read(frame)){
		//对当前帧进行分割，大小24*24
		frame_RIO = DivFra(frame, 24, 24);

		if (false == flag)
		{
			//当前帧数组赋值前一帧数组
			framePro_RIO = frame_RIO;
			flag = true;
		}
		else
		{
			//
			vector<FraRIO>::iterator it_pro = framePro_RIO.begin();
			vector<FraRIO>::iterator it = frame_RIO.begin();
			//对结构体数组进行遍历
			while (it != frame_RIO.end() && it_pro != framePro_RIO.end())
			{
				//当前帧与前一帧做差,存入临时帧
				absdiff(it->frameRIO, it_pro->frameRIO, tem_frame);
				//彩色图转灰度图
				cvtColor(tem_frame, tem_frame, COLOR_BGR2GRAY);
				//二值化
				threshold(tem_frame, tem_frame, 80, 255, CV_THRESH_BINARY);

				//cout << bSums(ColorDet(it->frameRIO)) << endl;
				tem_frame = tem_frame & ColorDet(it->frameRIO);
				//cout << bSums(tem_frame) << endl;

				//找出像素大于0的区块
				if (bSums(tem_frame)>0){
					//画出矩形框
					rectangle(frame, cvPoint(it->point_x, it->point_y), cvPoint(it->point_x + it->frameRIO.cols, it->point_y + it->frameRIO.rows), Scalar(255, 0, 0), 1, 1, 0);//能够实时显示运动物体
				}
				it++;
				it_pro++;
			}
			//
			framePro_RIO = frame_RIO;

			imshow("image", frame);

			waitKey(delay);
		}
	}
	return 0;
}