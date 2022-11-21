#include "histogram.hpp"

HistogramCPU::HistogramCPU(OpenCVGrayscaleMat* img) : m_img(img) {}

void HistogramCPU::computeHistogram()
{
    memset(m_histogramValues, 0, 256 * sizeof(unsigned int));

    for (int i = 0; i < m_img->rows; i++)
        for (int j = 0; j < m_img->cols; j++)
            m_histogramValues[(*m_img)(i, j)]++;
}

void HistogramCPU::computeHistogramCumule()
{
    computeHistogram();

    m_histogramCumuleValues[0] = m_histogramValues[0];
    for (int i = 1; i < 256; i++)
        m_histogramCumuleValues[i] = m_histogramCumuleValues[i - 1] + m_histogramValues[i];
}

void HistogramCPU::etirementHistogramme(OpenCVGrayscaleMat& outImg, unsigned int a, unsigned int b, unsigned char minValue, unsigned char maxValue)
{
    for (int i = 0; i < m_img->rows; i++)
        for (int j = 0; j < m_img->cols; j++)
            outImg(i, j) = (b - a) * (((*m_img)(i, j) - minValue) / (double)(maxValue - minValue)) + a;
}

void HistogramCPU::egalisationHisto(OpenCVGrayscaleMat& outImg)
{
    double a = (std::pow(2, 8) - 1);

    for (int i = 0; i < m_img->rows; i++)
        for (int j = 0; j < m_img->cols; j++)
            outImg(i, j) = a * ((m_histogramCumuleValues[(*m_img)(i, j)]) / (double)(m_img->rows * m_img->cols));
}

void HistogramCPU::imgMinMax(unsigned char& min, unsigned char& max)
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

OpenCVScalarMat HistogramCPU::drawHistogram(OpenCVGrayscaleMat& img) {
    // Compute histogram
    HistogramCPU histogram(&img);
    histogram.computeHistogram();

    // Draw histogram
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / 256);
    int valueMax = *std::max_element(histogram.getHistogram(), histogram.getHistogram() + 255);

    cv::Mat_<cv::Scalar> outHist(hist_h, hist_w, cv::Scalar(0, 0, 0));

    for (int i = 1; i < 256; i++) {
        cv::line(
            outHist,
            cv::Point(bin_w * (i - 1), hist_h - (cvRound((int)histogram.getHistogram()[i - 1]) * hist_h) / valueMax),
            cv::Point(bin_w * (i), hist_h - (cvRound((int)histogram.getHistogram()[i]) * hist_h) / valueMax),
            cv::Scalar(255, 255, 255), 2, 8, 0
        );
    }

    return outHist;
}
