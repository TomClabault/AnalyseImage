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

    cv::Mat inputImage, inputImagePreprocessed, outputKirsch, outputDerivX, outputDerivY;

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

    if (kernelType == "Kirsch" || kernelType == "kirsch")
        kirshFilter(inputImagePreprocessed, outputKirsch);
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

    cv::Mat gradient, gradientThresholded;


    if (kernelType == "Kirsch" || kernelType == "kirsch")
    {
        cv::Mat outputKirschNorm;
        normalize_grayscale_image(outputKirsch, outputKirschNorm);

        cv::imshow("Output", outputKirschNorm);
    }
    else if ((kernelType == "Sobel" || kernelType == "sobel") || (kernelType == "Prewitt" || kernelType == "prewitt"))
    {
        cv::Mat outputDerivXNorm, outputDerivYNorm, outputDerivXThresh, outputDerivYThresh;
        cv::Mat gradient_magnitude, gradient_direction, gradient_composite;

        normalize_grayscale_image(outputDerivX, outputDerivXNorm);
        normalize_grayscale_image(outputDerivY, outputDerivYNorm);

        //low_treshold(outputDerivXNorm, outputDerivXThresh, 32);
        //low_treshold(outputDerivYNorm, outputDerivYThresh, 32);

        gradientMagnitude(outputDerivX, outputDerivY, gradient_magnitude);
        gradientDirection(outputDerivXNorm, outputDerivYNorm, gradient_direction);
        multiply_rgb_by_grayscale(gradient_direction, gradient_magnitude, gradient_composite);

        cv::imshow("X", outputDerivXNorm);
        cv::imshow("Y", outputDerivYNorm);
        //cv::imshow("XThresh", outputDerivXThresh);
        //cv::imshow("YThresh", outputDerivYThresh);
        cv::imshow("gradientMagnitude", gradient_magnitude);
        cv::imshow("gradientDirection", gradient_direction);
        cv::imshow("gradientComposite", gradient_composite);
    }
    cv::waitKey(0);
}
