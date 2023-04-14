#include "filters.hpp"
#include "settings.hpp"

#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>



int main(int argc, char** argv)
{
    cv::Mat inputImagePreprocessed, outputKirsch, outputCanny, outputBinarized, outputDerivX, outputDerivY, gradient, gradientThresholded;

    if (argc < 2)
    {
        std::cout << "Usage: ./main <settings_file_path>" << std::endl;;

        return -1;
    }

    const char* settings_file_path = argv[1];

    Settings settings;
    read_settings(settings_file_path, settings);

    cv::imshow("Input image", settings.input_image);
    if (settings.preprocess_blur)
        gaussianBlur(settings.input_image, inputImagePreprocessed, settings.gaussian_kernel_size, settings.gaussian_kernel_sigma);
    else
        inputImagePreprocessed = settings.input_image;

    if (settings.kernel_type == "Kirsch" || settings.kernel_type == "kirsch")
    {
        cv::Mat outputKirschNorm, outputKirschThresholded;

        kirshFilter(inputImagePreprocessed, outputKirsch);
        normalize_grayscale_s16_to_u8(outputKirsch, outputKirschNorm);

        threshold_u8_by_settings(settings, outputKirschNorm, outputKirschThresholded);
        binarize(outputKirschThresholded, outputBinarized);

        cv::imshow("Normal Kirsch", outputKirschNorm);
        cv::imshow("Binarized", outputBinarized);
    }
    else if (settings.kernel_type == "Canny" || settings.kernel_type == "canny")
    {
        cannyEdgeDetection(inputImagePreprocessed, settings.canny_low_threshold, settings.canny_high_threshold, outputCanny);

        binarize(outputCanny, outputBinarized);
    }
    else if (settings.kernel_type =="Sobel" || settings.kernel_type =="sobel")
        sobelFilter(inputImagePreprocessed, outputDerivX, outputDerivY);
    else if (settings.kernel_type =="Prewitt" || settings.kernel_type =="prewitt")
        prewittFilter(inputImagePreprocessed, outputDerivX, outputDerivY);
    else
    {
        std::cout << "Kernel type not recognized\n";

        return -1;
    }

    if ((settings.kernel_type == "Sobel" || settings.kernel_type == "sobel") || (settings.kernel_type == "Prewitt" || settings.kernel_type == "prewitt"))
    {
        cv::Mat outputDerivXU8Norm, outputDerivYU8Norm, outputDerivXThresh, outputDerivYThresh;
        cv::Mat gradient_magnitude_normalized, gradient_magnitude_thresholded, gradient_direction, gradient_composite;

        gradientMagnitudeNormalized(outputDerivX, outputDerivY, gradient_magnitude_normalized);
        gradientDirection(outputDerivX, outputDerivY, gradient_direction);
        multiply_rgb_by_grayscale(gradient_direction, gradient_magnitude_normalized, gradient_composite);
        threshold_u8_by_settings(settings, gradient_magnitude_normalized, gradient_magnitude_thresholded);
        binarize(gradient_magnitude_thresholded, outputBinarized);

        normalize_grayscale_s16_to_u8(outputDerivX, outputDerivXU8Norm);
        normalize_grayscale_s16_to_u8(outputDerivY, outputDerivYU8Norm);

        cv::imshow("X", outputDerivXU8Norm);
        cv::imshow("Y", outputDerivYU8Norm);
        cv::imshow("gradientMagnitude/edge image", gradient_magnitude_normalized);
        cv::imshow("threshold", gradient_magnitude_thresholded);
        cv::imshow("gradientMagnitudeNormalized", gradient_magnitude_normalized);
        cv::imshow("gradientDirection", gradient_direction);
        cv::imshow("gradientDirection*Magnitude", gradient_composite);
    }

    cv::Mat hough_space, outputLines;
    cv::imshow("Input Hough", outputBinarized);

    cv::Mat hough_space_norm;
    hough_transform(outputBinarized, settings.hough_transform_nb_theta, settings.hough_transform_nb_rho, settings.hough_transform_threshold, hough_space, outputLines);
    normalize_grayscale_u16_to_u8(hough_space, hough_space_norm);

    cv::imshow("Hough Space", hough_space_norm);
    cv::imshow("Output lines", outputLines);

    cv::waitKey(0);
}
