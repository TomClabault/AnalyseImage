#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <iostream>

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

void kirshFilter(const cv::Mat& inputImage, cv::Mat& outputImage);
void sobelFilter(const cv::Mat& inputImage, cv::Mat& outputDerivX, cv::Mat& outputDerivY);
void prewittFilter(const cv::Mat& inputImage, cv::Mat& outputImage);

void convolution(const cv::Mat& inputImage, cv::Mat& outputImage, float** kernel, int kernel_size);

void gradientDirection(const cv::Mat& derivX, const cv::Mat& derivY, cv::Mat& gradientDir);

void print_kernel(float** kernel, unsigned int kernel_size);

void readMask(unsigned int& maskSize, float*** kernel, std::string maskFilePath);
void readImage(cv::Mat& image, const std::string& inputImagePath);

void sumImages(cv::Mat& outputSumImage, const cv::Mat& inputImage1, const cv::Mat& inputImage2);

void tresholding(const cv::Mat& inputImage, cv::Mat& outputImage, unsigned int treshold);
