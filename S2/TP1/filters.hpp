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

void binarize(const cv::Mat& edge_image, cv::Mat& out_binarized);

void cannyEdgeDetection(const cv::Mat& inputImagePreprocessed, unsigned char low_threshold, unsigned char high_threshold, cv::Mat& outputCanny);
void compute_histogram(const cv::Mat& input_image, unsigned int* histogram);
void convolution(const cv::Mat& inputImage, cv::Mat& outputImage, float** kernel, int kernel_size);

void gaussianBlur(const cv::Mat& inputImage, cv::Mat& outputBlurred, unsigned int kernel_size, float sigma);
void gradientDirection(const cv::Mat& derivX, const cv::Mat& derivY, cv::Mat& gradientDir);
void gradientMagnitudeNormalized(const cv::Mat& derivX, const cv::Mat& derivY, cv::Mat& gradientMagnitude);

void houghTransform(const cv::Mat& binarized_edge_image, int nb_theta, int nb_rho, cv::Mat& hough_space, cv::Mat& output_lines);

void multiply_rgb_by_grayscale(const cv::Mat& input_rgb, const cv::Mat& input_grayscale, cv::Mat& output_image);

void normalize_grayscale_image(const cv::Mat& input_image, cv::Mat& output_image);
void normalize_grayscale_s16_to_u8(const cv::Mat& input_s16, cv::Mat& output_u8_normalized);
void normalize_grayscale_u16_to_u8(const cv::Mat& u16_image, cv::Mat& u8_image_normalized);

void print_kernel(float** kernel, unsigned int kernel_size);

void readMask(unsigned int& maskSize, float*** kernel, std::string maskFilePath);
void readImage(cv::Mat& image, const std::string& inputImagePath);

void sumImages(cv::Mat& outputSumImage, const cv::Mat& inputImage1, const cv::Mat& inputImage2);

void low_treshold(cv::Mat& inputImage, cv::Mat& outputImage, unsigned char threshold);
void thresholding(const cv::Mat& inputImage, cv::Mat& outputImage, unsigned int treshold);

/**
 * @brief Applies a thresholding to each pixel of the \param input_image
 * based on the mean value of the neighborhood of the pixel
 * @param input_image The input image
 * @param output_image The thresholded output image
 * @param C A constant to subtract from the computed mean value (this relaxes
 * the threshold)
 */
void local_mean_thresholding(const cv::Mat& input_image, cv::Mat& output_image, unsigned int neighborhood_size = 5, unsigned char C = 0);
void global_otsu_thresholding(const cv::Mat& input_image, cv::Mat& output_thresholded_image);
