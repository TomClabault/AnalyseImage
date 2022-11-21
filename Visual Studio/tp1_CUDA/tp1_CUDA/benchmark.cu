#include <cuda_runtime.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <chrono>
#include <iostream>

#include "benchmark.hpp"
#include "histogram.hpp"
#include "timers.hpp"

#include "cudaUtils.hpp"

typedef cv::Mat_<unsigned char> OpenCVGrayscaleMat;

void Benchmark::CPU_benchmark()
{
    std::string image_path = cv::samples::findFile("./99MpxImg.jpg");
    OpenCVGrayscaleMat img = imread(image_path, cv::IMREAD_GRAYSCALE);

    OpenCVGrayscaleMat egaliseHistoImg(img.rows, img.cols);
    OpenCVGrayscaleMat etireHistoImg(img.rows, img.cols);

    HistogramCPU histogram(&img);
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

void Benchmark::GPU_benchmark()
{
    std::string image_path = cv::samples::findFile("./99MpxImg.jpg");
    OpenCVGrayscaleMat img = imread(image_path, cv::IMREAD_GRAYSCALE);

    uchar* CPU_imgData = img.data;
    uchar* CPU_imgOutData = (uchar*) malloc(sizeof(uchar) * img.rows * img.cols);
    
    uchar* GPU_imgData;
    uchar* GPU_imgOutData;
    HANDLE_CUDA_ERROR (cudaMalloc(&GPU_imgData, sizeof(uchar) * img.rows * img.cols));
    HANDLE_CUDA_ERROR (cudaMalloc(&GPU_imgOutData, sizeof(uchar) * img.rows * img.cols));
    HANDLE_CUDA_ERROR (cudaMemcpy(GPU_imgData, CPU_imgData, sizeof(uchar) * img.rows * img.cols, cudaMemcpyHostToDevice));

    unsigned char imgMin, imgMax;
    HistogramCPU histogram(&img);
    histogram.imgMinMax(imgMin, imgMax);

    {
        CudaTimer timer("Etirement histogramme", 10000);

        cudaEvent_t m_start, m_stop;
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_stop);

        cudaEventRecord(m_start);
        for(int i = 0; i < 10000; i++)
            etirementHistogramme << <NB_BLOCKS, NB_THREADS >> > (GPU_imgData, img.rows, img.cols, GPU_imgOutData, (unsigned int)133, (unsigned int)255, imgMin, imgMax);
        cudaEventRecord(m_stop);
        cudaEventSynchronize(m_stop);
    }

    cudaMemcpy(CPU_imgOutData, GPU_imgOutData, sizeof(uchar) * img.rows * img.cols, cudaMemcpyDeviceToHost);

    OpenCVGrayscaleMat outImg(img.rows, img.cols, CPU_imgOutData);
    cv::imshow("Original image", img);
    cv::imshow("Image etiree", outImg);

    cv::waitKey(0);
}
