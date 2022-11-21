#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void etirementHistogramme(unsigned char* imgData_devicePtr, unsigned int rows, unsigned int cols, unsigned char* outImg_devicePtr,
                                     unsigned int a, unsigned int b, unsigned char minValue, unsigned char maxValue)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    while (threadId < rows * cols)
    {
        outImg_devicePtr[threadId] = (b - a) * (imgData_devicePtr[threadId] - minValue) / (maxValue - minValue) + a;

        threadId += gridDim.x * blockDim.x;
    }
}

__global__ void egalisationHisto(unsigned char* imgData_devicePtr, unsigned char* outImg_devicePtr)
{

}