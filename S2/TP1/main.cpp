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

    cv::Mat inputImage, inputImagePreprocessed, outputKirsch, outputCanny, outputBinarized, outputDerivX, outputDerivY;

    if (argc < 7)
    {
        std::cout << "Usage: ./main <imageFile> <kernelType> <treshold> <preprocessBlur> <gaussianKernelSize> <gaussianKernelSigma>\n\n" << "<kernelType> can be either of: {sobel, prewitt, kirsh} " << std::endl;

        return -1;
    }

    readImage(inputImage, argv[1]);
    kernelType = std::string(argv[2]);
    threshold = atoi(argv[3]);
    preprocessBlur = std::string(argv[4]) != "false" && std::string(argv[4]) != "False";

    if (preprocessBlur)
    {
        gaussianBlurKernelSize = atoi(argv[5]);
        gaussianBlurSigma = atof(argv[6]);

        gaussianBlur(inputImage, inputImagePreprocessed, gaussianBlurKernelSize, gaussianBlurSigma);
    }
    else
        inputImagePreprocessed = inputImage;

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
        cv::Mat outputDerivXNorm, outputDerivYNorm, outputDerivXThresh, outputDerivYThresh;
        cv::Mat gradient_magnitude, gradient_direction, gradient_composite;

        //normalize_grayscale_image(outputDerivX, outputDerivXNorm);
        //normalize_grayscale_image(outputDerivY, outputDerivYNorm);
        int x = 119, y = 102;
        std::cout << "outputDerivX: " <<  outputDerivX.at<short int>(y, x) << " outputDerivY: " << outputDerivY.at<short int>(y, x) << std::endl;

        gradientMagnitudeNormalized(outputDerivX, outputDerivY, gradient_magnitude);
        gradientDirection(outputDerivX, outputDerivY, gradient_direction);
        multiply_rgb_by_grayscale(gradient_direction, gradient_magnitude, gradient_composite);
        binarize(gradient_magnitude, outputBinarized);

        cv::Mat test_x = cv::abs(outputDerivX);// cv::Mat(outputDerivX.rows, outputDerivX.cols, CV_8U);
        //for (int i = 0; i < outputDerivX.rows; i++)
            //for (int j = 0; j < outputDerivX.cols; j++)
                //test_x.at<unsigned char>(i, j) = std::abs(outputDerivX.at<short int>(i, j));

        cv::imshow("X", test_x);
        //cv::imshow("X", outputDerivX);
        cv::imshow("Y", outputDerivY);
        cv::imshow("gradientMagnitude/edge image", gradient_magnitude);
        cv::imshow("edge image binarized", outputBinarized);
        cv::imshow("gradientDirection", gradient_direction);
        cv::imwrite("gradient_direction.png", gradient_direction);//TODO remove
        cv::imshow("gradientDirection*Magnitude", gradient_composite);
    }

    int size_x = 512 / 4;
    int size_y = 512 / 4;

    cv::Mat hough_space, outputLines;
    //outputBinarized = cv::Mat(size_x, size_y, CV_8U);
    //outputBinarized.setTo(cv::Scalar(0, 0, 0));

/*
    cv::line(outputBinarized, cv::Point(64 / 4, 64 / 4), cv::Point(96 / 4, 64 / 4), cv::Scalar(255, 255, 255));
    cv::line(outputBinarized, cv::Point(64 / 4, 64 / 4), cv::Point(64 / 4, 96 / 4), cv::Scalar(255, 255, 255));
    cv::line(outputBinarized, cv::Point(96 / 4, 64 / 4), cv::Point(96 / 4, 96 / 4), cv::Scalar(255, 255, 255));
    cv::line(outputBinarized, cv::Point(64 / 4, 96 / 4), cv::Point(96 / 4, 96 / 4), cv::Scalar(255, 255, 255));

    cv::line(outputBinarized, cv::Point(32 / 4, 32 / 4), cv::Point(48 / 4, 32 / 4), cv::Scalar(255, 255, 255));
    cv::line(outputBinarized, cv::Point(32 / 4, 32 / 4), cv::Point(32 / 4, 48 / 4), cv::Scalar(255, 255, 255));
    cv::line(outputBinarized, cv::Point(48 / 4, 32 / 4), cv::Point(48 / 4, 48 / 4), cv::Scalar(255, 255, 255));
    cv::line(outputBinarized, cv::Point(32 / 4, 48 / 4), cv::Point(48 / 4, 48 / 4), cv::Scalar(255, 255, 255));
    */

    cv::line(outputBinarized, cv::Point(size_x / 4, size_y / 2), cv::Point(size_x / 4 * 3, size_y / 2), cv::Scalar(255, 255, 255));
    cv::imshow("Input Hough", outputBinarized);

    cv::Mat hough_space_norm;
    houghTransform(outputBinarized, 180*4, 180*4, hough_space, outputLines);
    normalize_grayscale_u16_to_u8(hough_space, hough_space_norm);

    cv::imshow("Hough Space", hough_space_norm);
    cv::imshow("Output lines", outputLines);

    cv::waitKey(0);
}
