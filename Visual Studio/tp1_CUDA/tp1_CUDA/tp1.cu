#include <chrono>
#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "benchmark.hpp"
#include "histogram.hpp"

typedef cv::Mat_<unsigned char> OpenCVGrayscaleMat;

int main()
{
    /*Benchmark::CPU_benchmark();
    Benchmark::GPU_benchmark();*/

    std::ifstream inputEtirementCPU("outputEtirementCPU.data");
    std::ifstream inputEtirementGPU("outputEtirementGPU.data");

    std::ifstream inputEgalisationCPU("outputEgalisationCPU.data");
    std::ifstream inputEgalisationGPU("outputEgalisationGPU.data");

    std::ofstream outputEtirementRatio("outputEtirementRatio.data");
    std::ofstream outputEgalisationRatio("outputEgalisationRatio.data");

    while (!inputEtirementCPU.eof())
    {
        double MPX, timeCPUEtirement, timeGPUEtirement, timeCPUEgalisation, timeGPUEgalisation;

        inputEtirementCPU >> MPX;
        inputEtirementCPU >> timeCPUEtirement;
        inputEtirementGPU >> MPX;
        inputEtirementGPU >> timeGPUEtirement;

        inputEgalisationCPU >> MPX;
        inputEgalisationCPU >> timeCPUEgalisation;
        inputEgalisationGPU >> MPX;
        inputEgalisationGPU >> timeGPUEgalisation;

        outputEtirementRatio << MPX << " " << timeCPUEtirement / timeGPUEtirement << std::endl;
        outputEgalisationRatio << MPX << " " << timeCPUEgalisation / timeGPUEgalisation << std::endl;
    }

    outputEtirementRatio.close();
    outputEgalisationRatio.close();

    std::exit(0);

    /*std::string image_path = cv::samples::findFile("./UnequalizedHawkes.jpg");
    OpenCVGrayscaleMat img = imread(image_path, cv::IMREAD_GRAYSCALE);

    if (img.empty())
    {
        std::cout << "Impossible de lire l'image" << std::endl;

        return 0;
    }


    unsigned char min, max;
    HistogramCPU histogram(&img);

    histogram.imgMinMax(min, max);
    histogram.computeHistogramCumule();

    OpenCVGrayscaleMat outImgEgalisation(img.rows, img.cols);
    histogram.egalisationHisto(outImgEgalisation);

    OpenCVGrayscaleMat outImgEtirement(img.rows, img.cols);
    histogram.etirementHistogramme(outImgEtirement, 133, 255, min, max);


    OpenCVScalarMat histOriginal = HistogramCPU::drawHistogram(img);
    cv::imshow("Original histogram", histOriginal);
    cv::imshow("Original image", img);

    OpenCVScalarMat histEgalisation = HistogramCPU::drawHistogram(outImgEgalisation);
    cv::imshow("Egalisation histogram", histEgalisation);
    cv::imshow("Egalisation image", outImgEgalisation);

    OpenCVScalarMat histEtirement = HistogramCPU::drawHistogram(outImgEtirement);
    cv::imshow("Etirement histogram", histEtirement);
    cv::imshow("Etirement image", outImgEtirement);

    cv::waitKey(0);*/
}