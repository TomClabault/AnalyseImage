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
    Benchmark::benchmark();

    std::string image_path = cv::samples::findFile("./UnequalizedHawkes.jpg");    
    OpenCVGrayscaleMat img = imread(image_path, cv::IMREAD_GRAYSCALE);

    if(img.empty())
    {
        std::cout << "Impossible de lire l'image" << std::endl;

        return 0;
    }


    unsigned char min, max;
    Histogram histogram(&img);

    histogram.imgMinMax(min, max);
    histogram.computeHistogramCumule();

    OpenCVGrayscaleMat outImgEgalisation(img.rows, img.cols);
    histogram.egalisationHisto(outImgEgalisation);

    OpenCVGrayscaleMat outImgEtirement(img.rows, img.cols);
    histogram.etirementHistogramme(outImgEtirement, 133, 255, min, max);


    OpenCVScalarMat histOriginal = Histogram::drawHistogram(img);
    cv::imshow("Original histogram", histOriginal);
    cv::imshow("Original image", img);

    OpenCVScalarMat histEgalisation = Histogram::drawHistogram(outImgEgalisation);
    cv::imshow("Egalisation histogram", histEgalisation);
    cv::imshow("Egalisation image", outImgEgalisation);

    OpenCVScalarMat histEtirement = Histogram::drawHistogram(outImgEtirement);
    cv::imshow("Etirement histogram", histEtirement);
    cv::imshow("Etirement image", outImgEtirement);

    cv::waitKey(0);
}