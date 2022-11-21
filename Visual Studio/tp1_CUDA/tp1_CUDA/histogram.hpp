#include <cuda_runtime.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>

typedef cv::Mat_<unsigned char> OpenCVGrayscaleMat;
typedef cv::Mat_<cv::Scalar> OpenCVScalarMat;

class HistogramCPU
{
public:
    HistogramCPU(OpenCVGrayscaleMat* img);

    void computeHistogram();
    void computeHistogramCumule();
    
    void etirementHistogramme(OpenCVGrayscaleMat& outImg, unsigned int a, unsigned int b, unsigned char minValue, unsigned char maxValue);
    void egalisationHisto(OpenCVGrayscaleMat& outImg);

    /**
     * Calcule la valeur d'intensité lumineuse minimale et maximale de l'image
     * en regardant tous ses pixels
     * 
     * @param [out] min La valeur minimale d'intensité de l'image
     * @param [out] max La valeur maximale d'intensité de l'image
     */
    void imgMinMax(unsigned char& min, unsigned char& max);

    unsigned int* getHistogram() { return m_histogramValues; };

    static OpenCVScalarMat drawHistogram(OpenCVGrayscaleMat& img);

private:
	OpenCVGrayscaleMat* m_img;

    unsigned int m_histogramValues[256];
    unsigned int m_histogramCumuleValues[256];
};

__global__ void etirementHistogramme(unsigned char* imgData_devicePtr, unsigned int rows, unsigned int cols, unsigned char* outImg_devicePtr,
                                     unsigned int a, unsigned int b, unsigned char minValue, unsigned char maxValue);
__global__ void egalisationHisto(unsigned char* imgData_devicePtr, unsigned char* outImg_devicePtr);
