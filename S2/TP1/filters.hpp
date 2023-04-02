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
void prewittFilter(const cv::Mat& inputImage, cv::Mat& outputDerivX, cv::Mat& outputDerivY);

void convolution(const cv::Mat& inputImage, cv::Mat& outputImage, float** kernel, int kernel_size);

void gaussianBlur(const cv::Mat& inputImage, cv::Mat& outputBlurred, unsigned int kernel_size, float sigma);
void gradientDirection(const cv::Mat& derivX, const cv::Mat& derivY, cv::Mat& gradientDir);
void gradientMagnitude(const cv::Mat& derivX, const cv::Mat& derivY, cv::Mat& gradientMagnitude);

void multiply_rgb_by_grayscale(const cv::Mat& input_rgb, const cv::Mat& input_grayscale, cv::Mat& output_image);

void normalize_grayscale_image(const cv::Mat& input_image, cv::Mat& output_image);

void print_kernel(float** kernel, unsigned int kernel_size);

void readMask(unsigned int& maskSize, float*** kernel, std::string maskFilePath);
void readImage(cv::Mat& image, const std::string& inputImagePath);

void sumImages(cv::Mat& outputSumImage, const cv::Mat& inputImage1, const cv::Mat& inputImage2);

void low_treshold(cv::Mat& inputImage, cv::Mat& outputImage, unsigned char threshold);
void tresholding(const cv::Mat& inputImage, cv::Mat& outputImage, unsigned int treshold);
