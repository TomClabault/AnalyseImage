#include "filters.hpp"

void readMask(unsigned int& maskSize, float*** kernel, std::string maskFilePath)
{
    std::ifstream maskFile(maskFilePath);
    if (maskFile.fail())
    {
        std::cout << "Impossible d'ouvrir " << maskFilePath << "\n";
        std::exit(0);
    }

    maskFile >> maskSize;
    *kernel = (float**)malloc(sizeof(float*) * maskSize);
    for (unsigned int i = 0; i < maskSize; i++)
        (*kernel)[i] = (float*)malloc(sizeof(float) * maskSize);

    for (unsigned int i = 0; i < maskSize; i++)
    {
        for (unsigned int j = 0; j < maskSize; j++)
        {
            char number[32];
            maskFile.getline(number, 32, ',');

            (*kernel)[i][j] = atof(number);
        }
    }
}

void readImage(cv::Mat& image, const std::string& inputImagePath)
{
    cv::Mat imageRGB = cv::imread(inputImagePath, cv::IMREAD_COLOR);
    if (imageRGB.empty()) {
        std::cout << "Impossible d'ouvrir l'image...\n";
        std::exit(0);
    }

    cv::cvtColor(imageRGB, image, cv::COLOR_RGB2GRAY);
}

void convolution(const cv::Mat& inputImage, cv::Mat& outputImage, float** kernel, int kernel_size)
{
    int half_kernel_size = kernel_size / 2;
    float kernel_sum = 0;

    for (int i = 0; i < kernel_size; i++)
        for (int j = 0; j < kernel_size; j++)
            kernel_sum += std::abs(kernel[i][j]);

    for (int y_img = 0; y_img < inputImage.rows; y_img++) {
        for (int x_img = 0; x_img < inputImage.cols; x_img++) {
            unsigned char current_pixel_value = inputImage.at<unsigned char>(y_img, x_img);

            float new_pixel_value = 0;
            for (int y_kernel = 0; y_kernel < kernel_size; y_kernel++) {
                for (int x_kernel = 0; x_kernel < kernel_size; x_kernel++) {
                    int y_kernel_shift = y_kernel - half_kernel_size;
                    int x_kernel_shift = x_kernel - half_kernel_size;

                    int y_pos = y_img + y_kernel_shift;
                    int x_pos = x_img + x_kernel_shift;

                    unsigned char pixel_value;

                    //Si on est en train de dépasser des bords de l'image
                    if (y_pos < 0 || y_pos >= inputImage.rows || x_pos < 0 || x_pos >= inputImage.cols)
                        //On va considérer que la valeur du pixel est la même que celle du pixel courant
                        pixel_value = current_pixel_value;
                    else
                        pixel_value = inputImage.at<unsigned char>(y_pos, x_pos);

                    new_pixel_value += pixel_value * kernel[y_kernel][x_kernel];
                }
            }

            float new_value = (new_pixel_value / kernel_sum);
            outputImage.at<unsigned char>(y_img, x_img) = (unsigned char)(new_value < 0 ? 0 : new_value);
        }
    }
}

template <size_t N>
void convolution(const cv::Mat& inputImage, cv::Mat& outputImage, const float kernel[N][N])
{
    int half_kernel_size = N / 2;

    float kernel_sum = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            kernel_sum += std::abs(kernel[i][j]);

    for (int y_img = 0; y_img < inputImage.rows; y_img++) {
        for (int x_img = 0; x_img < inputImage.cols; x_img++) {
            unsigned char current_pixel_value = inputImage.at<unsigned char>(y_img, x_img);

            float new_pixel_value = 0;
            for (int y_kernel = 0; y_kernel < N; y_kernel++) {
                for (int x_kernel = 0; x_kernel < N; x_kernel++) {
                    int y_kernel_shift = y_kernel - half_kernel_size;
                    int x_kernel_shift = x_kernel - half_kernel_size;

                    int y_pos = y_img + y_kernel_shift;
                    int x_pos = x_img + x_kernel_shift;

                    unsigned char pixel_value;

                    //Si on est en train de dépasser des bords de l'image
                    if (y_pos < 0 || y_pos >= inputImage.rows || x_pos < 0 || x_pos >= inputImage.cols)
                        //On va considérer que la valeur du pixel est la même que celle du pixel courant
                        pixel_value = current_pixel_value;
                    else
                        pixel_value = inputImage.at<unsigned char>(y_pos, x_pos);

                    new_pixel_value += pixel_value * kernel[y_kernel][x_kernel];
                }
            }

            float new_value = (new_pixel_value / kernel_sum);
            outputImage.at<unsigned char>(y_img, x_img) = (unsigned char)(new_value < 0 ? 0 : new_value);
        }
    }
}

void gradientDirection(const cv::Mat& derivX, const cv::Mat& derivY, cv::Mat& gradientDir)
{
    gradientDir = cv::Mat(derivX.rows, derivX.cols, CV_8UC3);

    float min = 1000;
    float max = 0;

    for (int i = 0; i < derivX.rows; i++)
        for (int j = 0; j < derivY.cols; j++)
        {
            if (derivY.at<unsigned char>(i, j) == 0)
                gradientDir.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
            else {
                if ( 4 * 180 * std::atan((float)derivX.at<unsigned char>(i, j) / (float)derivY.at<unsigned char>(i, j)) / M_PI > max)
                    max = 4 * 180 * std::atan((float)derivX.at<unsigned char>(i, j) / (float)derivY.at<unsigned char>(i, j)) / M_PI;
                if (4 * 180 * std::atan((float)derivX.at<unsigned char>(i, j) / (float)derivY.at<unsigned char>(i, j)) / M_PI < min)
                    min = 4 * 180 * std::atan((float)derivX.at<unsigned char>(i, j) / (float)derivY.at<unsigned char>(i, j)) / M_PI;

                gradientDir.at<cv::Vec3b>(i, j) = cv::Vec3b(2 * std::atan((float)derivX.at<unsigned char>(i, j) / (float)derivY.at<unsigned char>(i, j)) / M_PI * 255, 255, 255);
                //std::cout << 180 * std::atan((float)derivX.at<unsigned char>(i, j) / (float)derivY.at<unsigned char>(i, j)) / M_PI << std::endl;
            }
        }

    cv::cvtColor(gradientDir, gradientDir, cv::COLOR_HSV2BGR);

    std::cout << "min, max = " << min << ", " << max << "\n";
}

void tresholding(const cv::Mat& inputImage, cv::Mat& outputImage, unsigned int treshold)
{
    for (int i = 0; i < inputImage.rows; i++)
    {
        for (int j = 0; j < inputImage.cols; j++)
        {
            unsigned char pixelValue = inputImage.at<unsigned char>(i, j);
            outputImage.at<unsigned char>(i, j) = pixelValue >= treshold ? 255 : 0;
        }
    }
}

void sumImages(cv::Mat& outputSumImage, const cv::Mat& inputImage1, const cv::Mat& inputImage2)
{
    if ((inputImage1.rows != inputImage2.rows) || (inputImage1.cols != inputImage2.cols))
        return;

    for (int i = 0; i < inputImage1.rows; i++)
    {
        for (int j = 0; j < inputImage1.cols; j++)
        {
            int pixelValue = inputImage1.at<unsigned char>(i, j) + inputImage2.at<unsigned char>(i, j);

            outputSumImage.at<unsigned char>(i, j) = (unsigned char)(pixelValue > 255 ? 255 : pixelValue);
        }
    }
}

void print_kernel(float** kernel, unsigned int kernel_size)
{
    for (unsigned int i = 0; i < kernel_size; i++)
    {
        for (unsigned int j = 0; j < kernel_size; j++)
        {
            std::cout << kernel[i][j] << ", ";
        }

        std::cout << std::endl;
    }
}

void kirshFilter(const cv::Mat& inputImage, cv::Mat& outputImage)
{

}

void sobelFilter(const cv::Mat& inputImage, cv::Mat& outputDerivX, cv::Mat& outputDerivY)
{
    static const float sobelKernelX[3][3] =
    {
        {1, 0, -1},
        {2, 0, -2},
        {1, 0, -1}
    };

    static const float sobelKernelY[3][3] =
    {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1}
    };

    outputDerivX = cv::Mat(inputImage.rows, inputImage.cols, inputImage.type());
    outputDerivY = cv::Mat(inputImage.rows, inputImage.cols, inputImage.type());

    convolution<3>(inputImage, outputDerivX, sobelKernelX);
    convolution<3>(inputImage, outputDerivY, sobelKernelY);
}

void prewittFilter(const cv::Mat& inputImage, cv::Mat& outputImage)
{

}
