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

    cv::Mat inputImage, inputImagePreprocessed, outputImage, outputDerivX, outputDerivY;

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

        std::cout << gaussianBlurKernelSize << ", " << gaussianBlurSigma << std::endl;
        gaussianBlur(inputImage, inputImagePreprocessed, gaussianBlurKernelSize, gaussianBlurSigma);

        cv::imshow("blur", inputImagePreprocessed);
        cv::waitKey(0);
    }
    else
        inputImagePreprocessed = inputImage;

    if (kernelType == "Kirsh" || kernelType == "kirsh")
        kirshFilter(inputImagePreprocessed, outputImage);
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
        std::cout << "Kernel type unrecognized\n";

        return -1;
    }

    cv::Mat gradient, gradientSum, gradientThresholded;

    gradientDirection(outputDerivX, outputDerivY, gradient, 5);
    //gradientMagnitude(outputDerivX, outputDerivY, gradient, 5);
    sumImages(gradientSum, outputDerivX, outputDerivY);
    tresholding(gradientSum, gradientThresholded, threshold);

    cv::imshow("derivX", outputDerivX);
    cv::imshow("derivY", outputDerivY);
    cv::imshow("gradientXY", gradientSum);
    cv::imshow("gradientDirection", gradient);
    cv::imshow("Tresholded", gradientThresholded);
    cv::waitKey(0);
}
