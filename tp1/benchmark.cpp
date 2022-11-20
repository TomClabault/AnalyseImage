#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <chrono>
#include <iostream>

#include "benchmark.hpp"
#include "histogram.hpp"

typedef cv::Mat_<unsigned char> OpenCVGrayscaleMat;

class Timer
{
public:
    Timer(std::string message, int iterations)
    {
        m_message = message;
        m_iterations = iterations;
        m_start = std::chrono::high_resolution_clock::now();
    }

    ~Timer()
    {
        m_stop = std::chrono::high_resolution_clock::now();

        long long int duration = std::chrono::duration_cast<std::chrono::microseconds>(m_stop - m_start).count();

        std::cout << m_message << ": " << duration << "us total" << std::endl;
        std::cout << (duration / (double)m_iterations) << "us average" << std::endl << std::endl;
    }

private:
    std::string m_message;

    unsigned int m_iterations;

    std::chrono::time_point<std::chrono::steady_clock> m_start, m_stop;
};

void Benchmark::benchmark()
{
    std::string image_path = cv::samples::findFile("./lena_gray.bmp");
    OpenCVGrayscaleMat img = imread(image_path, cv::IMREAD_GRAYSCALE);

    OpenCVGrayscaleMat egaliseHistoImg(img.rows, img.cols);
    OpenCVGrayscaleMat etireHistoImg(img.rows, img.cols);

    Histogram histogram(&img);
    histogram.computeHistogramCumule();

    int iterations = 1000;

    {
        Timer timer("Egalisation histogramme", iterations);

        for (int i = 0; i < iterations; i++)
            histogram.egalisationHisto(egaliseHistoImg);
    }

    unsigned char min, max;
    histogram.imgMinMax(min, max);
    {
        Timer timer("Etirement histogramme", iterations);

        for (int i = 0; i < iterations; i++)
            histogram.etirementHistogramme(etireHistoImg, 0, 255, min, max);
    }
}