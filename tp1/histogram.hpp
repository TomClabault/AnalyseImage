#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#define SIMD_OPTIMIZATION 1

typedef cv::Mat_<unsigned char> OpenCVGrayscaleMat;

class Histogram
{
public:
	Histogram(OpenCVGrayscaleMat* img);

    void computeHistogram();
    void computeHistogramCumule();
    
    void etirementHistogramme(OpenCVGrayscaleMat& outImg, unsigned int a, unsigned int b, unsigned char minValue, unsigned char maxValue);
    void egalisationHisto(OpenCVGrayscaleMat& outImg);

#if SIMD_OPTIMIZATION
    void etirementHistogramme_SIMD(OpenCVGrayscaleMat& outImg, unsigned int a, unsigned int b, unsigned char minValue, unsigned char maxValue);
    void egalisationHisto_SIMD(OpenCVGrayscaleMat& outImg);
#endif

    /**
     * Calcule la valeur d'intensité lumineuse minimale et maximale de l'image
     * en regardant tous ses pixels
     * 
     * @param [out] min La valeur minimale d'intensité de l'image
     * @param [out] max La valeur maximale d'intensité de l'image
     */
    void imgMinMax(unsigned char& min, unsigned char& max);

private:
	OpenCVGrayscaleMat* m_img;

    int m_histogramValues[256];
    int m_histogramCumuleValues[256];
};
