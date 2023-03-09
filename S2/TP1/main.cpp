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

    unsigned int treshold;

    cv::Mat inputImage, outputImage, outputDerivX, outputDerivY;

    if (argc < 4)
    {
        std::cout << "Usage: ./main <imageFile> <kernelType> <treshold>" << std::endl;

        return -1;
    }

    readImage(inputImage, argv[1]); 
    kernelType = std::string(argv[2]);
    treshold = atoi(argv[3]);

    if (kernelType == "Kirsh" || kernelType == "kirsh")
        kirshFilter(inputImage, outputImage);
    else if (kernelType == "Sobel" || kernelType == "sobel")
        sobelFilter(inputImage, outputDerivX, outputDerivY);
    else if (kernelType == "Prewitt" || kernelType == "prewitt")
        prewittFilter(inputImage, outputImage);
    else if (kernelType == "free2")//X and Y filters are given in files
        ;
    else if (kernelType == "free4")//4 "diagonal directions" filters are given in files
        ;
    else
    {
        std::cout << "Kernel type unrecognized\n";

        return -1;
    }

    cv::Mat gradientDir;
    gradientDirection(outputDerivX, outputDerivY, gradientDir);

    cv::imshow("derivX", outputDerivX);
    cv::imshow("derivY", outputDerivY);
    cv::imshow("gradientDirection", gradientDir);
    cv::waitKey(0);
}
