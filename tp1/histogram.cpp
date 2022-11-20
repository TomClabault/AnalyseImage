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

#if SIMD_OPTIMIZATION
__m128i at_home_mm256_cvtepi32_epu8(__m256i a)
{
    //[0, 0, 0, X, 0, 0, 0, X, 0, 0, 0, X, 0, 0, 0, X, 0, 0, 0, X, 0, 0, 0, X, 0, 0, 0, X, 0, 0, 0 X]

}

void convertToBytes(const uint32_t* source, uint8_t* dest, size_t count)
{
    // 4 bytes of the shuffle mask to fetch bytes 0, 4, 8 and 12 from a 16-bytes source vector
    constexpr int shuffleScalar = 0x0C080400;
    // Mask to shuffle first 8 values of the batch, making first 8 bytes of the result
    const __m256i shuffMaskLow = _mm256_setr_epi32(shuffleScalar, -1, -1, -1, -1, shuffleScalar, -1, -1);
    // Mask to shuffle last 8 values of the batch, making last 8 bytes of the result
    const __m256i shuffMaskHigh = _mm256_setr_epi32(-1, -1, shuffleScalar, -1, -1, -1, -1, shuffleScalar);
    // Indices for the final _mm256_permutevar8x32_epi32
    const __m256i finalPermute = _mm256_setr_epi32(0, 5, 2, 7, 0, 5, 2, 7);

    const uint32_t* const sourceEnd = source + count;
    // Vectorized portion, each iteration handles 16 values.
    // Round down the count making it a multiple of 16.
    const size_t countRounded = count & ~((size_t)15);
    const uint32_t* const sourceEndAligned = source + countRounded;
    while (source < sourceEndAligned)
    {
        // Load 16 inputs into 2 vector registers
        const __m256i s1 = _mm256_load_si256((const __m256i*)source);
        const __m256i s2 = _mm256_load_si256((const __m256i*)(source + 8));
        source += 16;
        // Shuffle bytes into correct positions; this zeroes out the rest of the bytes.
        const __m256i low = _mm256_shuffle_epi8(s1, shuffMaskLow);
        const __m256i high = _mm256_shuffle_epi8(s2, shuffMaskHigh);
        // Unused bytes were zeroed out, using bitwise OR to merge, very fast.
        const __m256i res32 = _mm256_or_si256(low, high);
        // Final shuffle of the 32-bit values into correct positions
        const __m256i res16 = _mm256_permutevar8x32_epi32(res32, finalPermute);
        // Store lower 16 bytes of the result
        _mm_storeu_si128((__m128i*)dest, _mm256_castsi256_si128(res16));
        dest += 16;
    }

    // Deal with the remainder
    while (source < sourceEnd)
    {
        *dest = (uint8_t)(*source);
        source++;
        dest++;
    }
}

void Histogram::egalisationHisto_SIMD(OpenCVGrayscaleMat& outImg)
{
    __m256 a = _mm256_set1_ps(std::pow(2, 8) - 1);
    __m256 invMN = _mm256_set1_ps(1.0f / (float)(m_img->rows * m_img->cols));

    for (int i = 0; i < m_img->rows; i++)
        for (int j = 0; j < m_img->cols / 16; j += 16)
        {
            __m128i inImg = _mm_loadu_si128((const __m128i*)m_img->data);

            __m256i inImg256_A = _mm256_cvtepu8_epi32(inImg);
            __m256i inImg256_B = _mm256_cvtepu8_epi32(_mm_srli_si128(inImg, 8));

            __m256i m_histoValues_A = _mm256_i32gather_epi32(m_histogramCumuleValues, inImg256_A, 4);
            __m256i m_histoValues_B = _mm256_i32gather_epi32(m_histogramCumuleValues, inImg256_B, 4);

            __m256 m_histoValues_Af = _mm256_cvtepi32_ps(m_histoValues_A);
            __m256 m_histoValues_Bf = _mm256_cvtepi32_ps(m_histoValues_B);

            __m256 numerator_A = _mm256_mul_ps(a, m_histoValues_Af);
            __m256 numerator_B = _mm256_mul_ps(a, m_histoValues_Af);

            __m256 out_Af = _mm256_mul_ps(numerator_A, invMN);
            __m256 out_Bf = _mm256_mul_ps(numerator_B, invMN);

            __m256i out_A = _mm256_cvtps_epi32(out_Af);
            __m256i out_B = _mm256_cvtps_epi32(out_Bf);

            __m128i out_A128 = at_home_mm256_cvtepi32_epu8(out_A);
            __m128i out_B128 = at_home_mm256_cvtepi32_epu8(out_B);

            __m128i out_AB = _mm_or_si128(_mm_slli_si128(out_A128, 8), out_B128);

            __debugbreak();

            /*outImg(i, j + 0) = a * ((m_histogramCumuleValues[(*m_img)(i, j + 0)]) / (double)(m_img->rows * m_img->cols));
            outImg(i, j + 1) = a * ((m_histogramCumuleValues[(*m_img)(i, j + 1)]) / (double)(m_img->rows * m_img->cols));
            outImg(i, j + 2) = a * ((m_histogramCumuleValues[(*m_img)(i, j + 2)]) / (double)(m_img->rows * m_img->cols));
            outImg(i, j + 3) = a * ((m_histogramCumuleValues[(*m_img)(i, j + 3)]) / (double)(m_img->rows * m_img->cols));*/

        }
}

void Histogram::etirementHistogramme_SIMD(OpenCVGrayscaleMat& outImg, unsigned int a, unsigned int b, unsigned char minValue, unsigned char maxValue)
{

}
#endif

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