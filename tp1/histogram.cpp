#include "histogram.hpp"

Histogram::Histogram(OpenCVGrayscaleMat* img) : m_img(img) {}

void Histogram::computeHistogram()
{
    memset(m_histogramValues, 0, 256 * sizeof(unsigned int));

    for (int i = 0; i < m_img->rows; i++)
        for (int j = 0; j < m_img->cols; j++)
            m_histogramValues[(*m_img)(i, j)]++;
}

void Histogram::computeHistogramCumule()
{
    computeHistogram();

    m_histogramCumuleValues[0] = m_histogramValues[0];
    for (int i = 1; i < 256; i++)
        m_histogramCumuleValues[i] = m_histogramCumuleValues[i - 1] + m_histogramValues[i];
}

void Histogram::etirementHistogramme(OpenCVGrayscaleMat& outImg, unsigned int a, unsigned int b, unsigned char minValue, unsigned char maxValue)
{
    for (int i = 0; i < m_img->rows; i++)
        for (int j = 0; j < m_img->cols; j++)
            outImg(i, j) = (b - a) * (((*m_img)(i, j) - minValue) / (double)(maxValue - minValue));
}

void Histogram::egalisationHisto(OpenCVGrayscaleMat& outImg)
{
    double a = (std::pow(2, 8) - 1);

    for (int i = 0; i < m_img->rows; i++)
        for (int j = 0; j < m_img->cols; j++)
            outImg(i, j) = a * ((m_histogramCumuleValues[(*m_img)(i, j)]) / (double)(m_img->rows * m_img->cols));
}

void Histogram::imgMinMax(unsigned char& min, unsigned char& max)
{
    min = max = (*m_img)(0, 0);
    for (int i = 0; i < m_img->rows; i++)
    {
        for (int j = 0; j < m_img->cols; j++)
        {
            unsigned char pixel_i_j = (*m_img)(i, j);
            if (pixel_i_j < min)
                min = pixel_i_j;

            if (pixel_i_j > max)
                max = pixel_i_j;
        }
    }
}