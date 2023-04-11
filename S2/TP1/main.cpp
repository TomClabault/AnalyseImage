#include "filters.hpp"

#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char** argv)
{
    std::string kernelType;
    std::string inputImagePath;

    unsigned int threshold;

    bool preprocessBlur;
    unsigned int gaussianBlurKernelSize;
    float gaussianBlurSigma;

    cv::Mat input_image, inputImagePreprocessed, outputKirsch, outputCanny, outputBinarized, outputDerivX, outputDerivY;

    if (argc < 7)
    {
        std::cout << "Usage: ./main <imageFile> <kernelType> <treshold> <preprocessBlur> <gaussianKernelSize> <gaussianKernelSigma>\n\n" << "<kernelType> can be either of: {sobel, prewitt, kirsh} " << std::endl;

        return -1;
    }

    readImage(input_image, argv[1]);
    kernelType = std::string(argv[2]);
    threshold = atoi(argv[3]);
    preprocessBlur = std::string(argv[4]) != "false" && std::string(argv[4]) != "False";

    cv::imshow("Input image", input_image);
    if (preprocessBlur)
    {
        gaussianBlurKernelSize = atoi(argv[5]);
        gaussianBlurSigma = atof(argv[6]);

        gaussianBlur(input_image, inputImagePreprocessed, gaussianBlurKernelSize, gaussianBlurSigma);
    }
    else
        inputImagePreprocessed = input_image;

    if (kernelType == "Kirsch" || kernelType == "kirsch")
        kirshFilter(inputImagePreprocessed, outputKirsch);
    else if (kernelType == "Canny" || kernelType == "canny")
    {
        if (argc < 9)//Missing low and high threshold on the command line
        {
            std::cout << "Missing low and high threshold for the double threshold step of the canny edge detection:\n";
            std::cout << "Usage: ./main <imageFile> canny UNUSED <preprocessBlur> <gaussianKernelSize> <gaussianKernelSigma> <lowThresholdValue> <highThresholdValue>\n\n" << std::endl;

            return -1;
        }

        unsigned char canny_low_threshold, canny_high_threshold;

        canny_low_threshold = atoi(argv[7]);
        canny_high_threshold = atoi(argv[8]);

        cannyEdgeDetection(inputImagePreprocessed, canny_low_threshold, canny_high_threshold, outputCanny);
    }
    else if (kernelType == "Sobel" || kernelType == "sobel")
        sobelFilter(inputImagePreprocessed, outputDerivX, outputDerivY);
    else if (kernelType == "Prewitt" || kernelType == "prewitt")
        prewittFilter(inputImagePreprocessed, outputDerivX, outputDerivY);
    else if (kernelType == "free2")//X and Y filters are given in files
        ;
    else if (kernelType == "free4")//4 "diagonal directions" filters are given in files
        ;
    else
    {
        std::cout << "Kernel type not recognized\n";

        return -1;
    }

    cv::Mat gradient, gradientThresholded;


    if (kernelType == "Kirsch" || kernelType == "kirsch")
    {
        cv::Mat outputKirschNorm;
        normalize_grayscale_image(outputKirsch, outputKirschNorm);
        binarize(outputKirschNorm, outputBinarized);

        cv::imshow("Output", outputBinarized);
    }
    else if (kernelType == "Canny" || kernelType == "canny")
    {
        binarize(outputCanny, outputBinarized);

        cv::imshow("Output", outputBinarized);
    }
    else if ((kernelType == "Sobel" || kernelType == "sobel") || (kernelType == "Prewitt" || kernelType == "prewitt"))
    {
        cv::Mat outputDerivXU8Norm, outputDerivYU8Norm, outputDerivXThresh, outputDerivYThresh;
        cv::Mat gradient_magnitude, gradient_magnitude_thresholded, gradient_magnitude_thresholded_t, gradient_magnitude_thresholded_o, gradient_direction, gradient_composite;

        gradientMagnitudeNormalized(outputDerivX, outputDerivY, gradient_magnitude);
        gradientDirection(outputDerivX, outputDerivY, gradient_direction);
        multiply_rgb_by_grayscale(gradient_direction, gradient_magnitude, gradient_composite);
        local_mean_thresholding(inputImagePreprocessed, gradient_magnitude_thresholded, 5, 2);
        global_otsu_thresholding(inputImagePreprocessed, gradient_magnitude_thresholded_o);
        thresholding(inputImagePreprocessed, gradient_magnitude_thresholded_t, 127);
        binarize(gradient_magnitude_thresholded, outputBinarized);

        normalize_grayscale_s16_to_u8(outputDerivX, outputDerivXU8Norm);
        normalize_grayscale_s16_to_u8(outputDerivY, outputDerivYU8Norm);

//        cv::imshow("X", outputDerivXU8Norm);
//        cv::imshow("Y", outputDerivYU8Norm);
//        cv::imshow("gradientMagnitude/edge image", gradient_magnitude);
        cv::imshow("meanThreshold", gradient_magnitude_thresholded);
        cv::imshow("otsuThreshold", gradient_magnitude_thresholded_o);
        cv::imshow("threshold", gradient_magnitude_thresholded_t);
        cv::imshow("gradientMagnitudeNormalized", gradient_magnitude);
//        cv::imshow("edge image binarized", outputBinarized);
//        cv::imshow("gradientDirection", gradient_direction);
//        cv::imshow("gradientDirection*Magnitude", gradient_composite);
    }

    cv::Mat hough_space, outputLines;
    cv::imshow("Input Hough", outputBinarized);

    cv::Mat hough_space_norm;
    houghTransform(outputBinarized, 180*4, 180*4, hough_space, outputLines);
    normalize_grayscale_u16_to_u8(hough_space, hough_space_norm);

    cv::imshow("Hough Space", hough_space_norm);
    cv::imshow("Output lines", outputLines);

    cv::waitKey(0);
}
