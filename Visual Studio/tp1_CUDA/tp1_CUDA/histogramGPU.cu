#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void etirementHistogramme(unsigned char* imgData_devicePtr, unsigned int rows, unsigned int cols, unsigned char* outImg_devicePtr,
                                     unsigned int a, unsigned int b, unsigned char minValue, unsigned char maxValue)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ba = b - a;
    float maxMin = maxValue - minValue;

    while (threadId < rows * cols)
    {
        outImg_devicePtr[threadId] = ba * (imgData_devicePtr[threadId] - minValue) / maxMin + a;

        threadId += gridDim.x * blockDim.x;
    }
}

__global__ void egalisationHisto(unsigned char* imgData_devicePtr, unsigned int rows, unsigned int cols, unsigned char* outImg_devicePtr, unsigned int* histogramCumule)
{
    double a = 255;

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    while (threadId < rows * cols)
    {
        outImg_devicePtr[threadId] = a * ((histogramCumule[imgData_devicePtr[threadId]]) / (double)(rows * cols));

        threadId += gridDim.x * blockDim.x;
    }
}