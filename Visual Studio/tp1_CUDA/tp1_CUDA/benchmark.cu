#include <cuda_runtime.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <chrono>
#include <iostream>
#include <fstream>

#include "benchmark.hpp"
#include "histogram.hpp"
#include "timers.hpp"

#include "cudaUtils.hpp"

typedef cv::Mat_<unsigned char> OpenCVGrayscaleMat;

#define START_FACTOR 0.01
#define END_FACTOR 0.8
#define NB_STEPS 50

#define ITERATIONS_CPU 1000
#define ITERATIONS_GPU 1000

void Benchmark::CPU_benchmark()
{
    std::string image_path = cv::samples::findFile("./112MpxImg.jpg");
    OpenCVGrayscaleMat img = imread(image_path, cv::IMREAD_GRAYSCALE);

    OpenCVGrayscaleMat egaliseHistoImg(img.rows, img.cols);
    OpenCVGrayscaleMat etireHistoImg(img.rows, img.cols);

    HistogramCPU histogram(&img);
    histogram.computeHistogramCumule();
    unsigned char min, max;
    histogram.imgMinMax(min, max);

    std::ofstream outputEtirement("outputEtirementCPU.data");
    std::ofstream outputEgalisation("outputEgalisationCPU.data");

    double step = (END_FACTOR - START_FACTOR) / NB_STEPS;
    double currentFactor = START_FACTOR;
    while (currentFactor < END_FACTOR)
    {
        int rows = currentFactor * img.rows;
        int cols = currentFactor * img.cols;
        double MPX = (rows * cols) / 1000000.0;

        double bestMilliseconds = -1;
        for (int i = 0; i < ITERATIONS_CPU; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            histogram.egalisationHisto(egaliseHistoImg, rows, cols);
            auto stop = std::chrono::high_resolution_clock::now();

            long long int duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
            if (duration < bestMilliseconds || bestMilliseconds == -1)
                bestMilliseconds = duration / 1000.0;
        }
        std::cout << "[" << MPX << "Mpx] Egalisation best time: " << bestMilliseconds << "ms" << std::endl;
        outputEgalisation << MPX << " " << bestMilliseconds << std::endl;

        bestMilliseconds = -1;
        for (int i = 0; i < ITERATIONS_CPU; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            histogram.etirementHistogramme(etireHistoImg, rows, cols, 0, 255, min, max);
            auto stop = std::chrono::high_resolution_clock::now();

            long long int duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
            if (duration < bestMilliseconds || bestMilliseconds == -1)
                bestMilliseconds = duration / 1000.0;
        }
        std::cout << "[" << MPX << "Mpx] Etirement best time: " << bestMilliseconds << "ms" << std::endl;
        outputEtirement << MPX << " " << bestMilliseconds << std::endl;

        currentFactor += step;
    }
}

void Benchmark::GPU_benchmark()
{
    std::string image_path = cv::samples::findFile("./112MpxImg.jpg");
    OpenCVGrayscaleMat img = imread(image_path, cv::IMREAD_GRAYSCALE);

    uchar* CPU_imgData = img.data;
    uchar* CPU_imgOutData = (uchar*) malloc(sizeof(uchar) * img.rows * img.cols);
    
    uchar* GPU_imgData;
    uchar* GPU_imgOutData;
    uint* GPU_histoCumule;
    HANDLE_CUDA_ERROR(cudaMalloc(&GPU_imgData, sizeof(uchar) * img.rows * img.cols));
    HANDLE_CUDA_ERROR(cudaMalloc(&GPU_imgOutData, sizeof(uchar) * img.rows * img.cols));
    HANDLE_CUDA_ERROR(cudaMalloc(&GPU_histoCumule, sizeof(uint) * 256));
    HANDLE_CUDA_ERROR(cudaMemcpy(GPU_imgData, CPU_imgData, sizeof(uchar) * img.rows * img.cols, cudaMemcpyHostToDevice));

    unsigned char imgMin, imgMax;
    HistogramCPU histogram(&img);
    histogram.imgMinMax(imgMin, imgMax);
    histogram.computeHistogramCumule();
    HANDLE_CUDA_ERROR(cudaMemcpy(GPU_histoCumule, histogram.getHistogramCumule(), sizeof(uint) * 256, cudaMemcpyHostToDevice));

    std::ofstream outputEtirement("outputEtirementGPU.data");
    std::ofstream outputEgalisation("outputEgalisationGPU.data");

    double step = (END_FACTOR - START_FACTOR) / NB_STEPS;
    double currentFactor = START_FACTOR;
    while(currentFactor < END_FACTOR)
    {
        int rows = currentFactor * img.rows;
        int cols = currentFactor * img.cols;
        double MPX = (rows * cols) / 1000000.0;

        float bestMilliseconds = -1;
        cudaEvent_t start, stop;
        for (int i = 0; i < ITERATIONS_GPU; i++)
        {
            float currentMilliseconds;

            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);

            etirementHistogramme << <NB_BLOCKS, NB_THREADS >> > (GPU_imgData, rows, cols, GPU_imgOutData, (unsigned int)133, (unsigned int)255, imgMin, imgMax);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&currentMilliseconds, start, stop);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            if (bestMilliseconds == -1 || bestMilliseconds > currentMilliseconds)
                bestMilliseconds = currentMilliseconds;
        }
        std::cout << "[" << MPX << "Mpx] Etirement best time: " << bestMilliseconds << "ms" << std::endl;
        outputEtirement << MPX << " " << bestMilliseconds << std::endl;

        bestMilliseconds = -1;
        for (int i = 0; i < ITERATIONS_GPU; i++)
        {
            float currentMilliseconds;

            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);

            egalisationHisto << <NB_BLOCKS, NB_THREADS >> > (GPU_imgData, rows, cols, GPU_imgOutData, GPU_histoCumule);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&currentMilliseconds, start, stop);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            if (bestMilliseconds == -1 || bestMilliseconds > currentMilliseconds)
                bestMilliseconds = currentMilliseconds;
        }
        std::cout << "[" << MPX << "Mpx] Egalisation best time: " << bestMilliseconds << "ms" << std::endl;
        outputEgalisation << MPX << " " << bestMilliseconds << std::endl;

        currentFactor += step;
    }

    outputEgalisation.close();
    outputEtirement.close();

    /*cudaMemcpy(CPU_imgOutData, GPU_imgOutData, sizeof(uchar) * img.rows * img.cols, cudaMemcpyDeviceToHost);

    OpenCVGrayscaleMat outImg(img.rows, img.cols, CPU_imgOutData);
    cv::imshow("Original image", img);
    cv::imshow("Image etiree", outImg);

    cv::waitKey(0);*/
}
