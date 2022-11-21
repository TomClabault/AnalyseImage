#include <chrono>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "benchmark.hpp"
#include "histogram.hpp"

typedef cv::Mat_<unsigned char> OpenCVGrayscaleMat;

int main()
{
    Benchmark::CPU_benchmark();
    Benchmark::GPU_benchmark();

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