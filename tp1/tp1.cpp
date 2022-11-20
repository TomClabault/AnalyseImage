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

    std::string image_path = cv::samples::findFile("./lena_gray.bmp");    
    OpenCVGrayscaleMat img = imread(image_path, cv::IMREAD_GRAYSCALE);

    if(img.empty())
    {
        std::cout << "Impossible de lire l'image" << std::endl;

        return 0;
    }

    OpenCVGrayscaleMat outImg(img.rows, img.cols);


    unsigned char min, max;
    Histogram histogram(&img);

    histogram.imgMinMax(min, max);
    histogram.computeHistogramCumule();
    histogram.etirementHistogramme(outImg, 133, 255, min, max);

    cv::namedWindow("Output");
    cv::imshow("Output", outImg);
    cv::waitKey(0);
}