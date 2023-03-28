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

void compute_gaussian_kernel(float** kernel, unsigned int kernel_size, float sigma) {
    unsigned int half_size = kernel_size / 2;
    float kernel_sum = 0;

    for (unsigned int y = 0; y < kernel_size; y++) {
        for (unsigned int x = 0; x < kernel_size; x++) {
            int shift_x = x - half_size;
            int shift_y = y - half_size;

            kernel[y][x] = 1.0 / (2 * M_PI * sigma * sigma) * exp(-((shift_x * shift_x + shift_y * shift_y) / (2 * sigma * sigma)));

            kernel_sum += kernel[y][x];
        }
    }

    //Pour être sûr que la somme des valeurs du noyau = 1
    for (unsigned int y = 0; y < kernel_size; y++) {
        for (unsigned int x = 0; x < kernel_size; x++) {
            kernel[y][x] /= kernel_sum;
        }
    }
}

void gaussianBlur(const cv::Mat& inputImage, cv::Mat& outputImage, unsigned int kernel_size, float sigma) {

    if (kernel_size == 0 || (kernel_size % 2 == 0)) {
        throw std::invalid_argument("The kernel size needs to be stricly positive and odd.");

        return;
    }

    float** kernel = (float**)malloc(sizeof(float*) * kernel_size);
    if (kernel == NULL)
        return;

    for (unsigned int i = 0; i < kernel_size; i++) {
        kernel[i] = (float*)malloc(sizeof(float) * kernel_size);

        if (kernel[i] == NULL)
            return;
    }

    compute_gaussian_kernel(kernel, kernel_size, sigma);

    unsigned int rows = inputImage.rows;
    unsigned int cols = inputImage.cols;

    int half_kernel_size = kernel_size / 2;

    outputImage = cv::Mat(inputImage.rows, inputImage.cols, inputImage.type());

    for (unsigned int y_img = 0; y_img < rows; y_img++) {
        for (unsigned int x_img = 0; x_img < cols; x_img++) {
            unsigned char current_pixel_value = inputImage.at<unsigned char>(y_img, x_img);

            double new_pixel_value = 0;
            for (int y_kernel = 0; y_kernel < kernel_size; y_kernel++) {
                for (int x_kernel = 0; x_kernel < kernel_size; x_kernel++) {
                    int y_kernel_shift = y_kernel - half_kernel_size;
                    int x_kernel_shift = x_kernel - half_kernel_size;

                    unsigned int y_pos = y_img + y_kernel_shift;
                    unsigned int x_pos = x_img + x_kernel_shift;

                    unsigned char pixel_value;

                    //Si on est en train de dépasser des bords de l'image
                    if (y_pos < 0 || y_pos >= rows || x_pos < 0 || x_pos >= cols) {
                        //On va considérer que la valeur du pixel est la même que celle du pixel courant
                        pixel_value = current_pixel_value;
                    }
                    else {
                        pixel_value = inputImage.at<unsigned char>(y_pos, x_pos);
                    }

                    new_pixel_value += pixel_value * kernel[y_kernel][x_kernel];
                }
            }

            outputImage.at<unsigned char>(y_img, x_img) = (unsigned char)new_pixel_value;
        }
    }
}

void gradientDirection(const cv::Mat& derivX, const cv::Mat& derivY, cv::Mat& gradientDir, float treshold)
{
    gradientDir = cv::Mat(derivX.rows, derivX.cols, CV_8UC3);

    for (int i = 0; i < derivX.rows; i++)
        for (int j = 0; j < derivY.cols; j++)
        {
            if (derivY.at<unsigned char>(i, j) <= treshold)
                gradientDir.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
            else
                gradientDir.at<cv::Vec3b>(i, j) = cv::Vec3b(2 * std::atan((float)derivX.at<unsigned char>(i, j) / (float)derivY.at<unsigned char>(i, j)) / M_PI * 255, 255, 255);
        }

    cv::cvtColor(gradientDir, gradientDir, cv::COLOR_HSV2BGR);
}

void gradientMagnitude(const cv::Mat& derivX, const cv::Mat& derivY, cv::Mat& gradientMagnitude, float treshold)
{
    /*gradientMagnitude = cv::Mat(derivX.rows, derivX.cols, CV_8UC3);

    for (int i = 0; i < derivX.rows; i++)
        for (int j = 0; j < derivY.cols; j++)
        {
            float value = std::sqrt((float)derivX.at<unsigned char>(i, j) * (float)derivX.at<unsigned char>(i, j) + (float)derivY.at<unsigned char>(i, j) * (float)derivY.at<unsigned char>(i, j));

            gradientDirection.at<cv::Vec3b>(i, j) = cv::Vec3b( * 255, 255, 255);
        }*/
}

void tresholding(const cv::Mat& inputImage, cv::Mat& outputImage, unsigned int treshold)
{
    outputImage = cv::Mat(inputImage.rows, inputImage.cols, inputImage.type());

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
    if ((inputImage1.rows != inputImage2.rows) || (inputImage1.cols != inputImage2.cols)) {
        std::cout << "Input images to sumImages must be the same dimensions\n";

        return;
    }

    outputSumImage = cv::Mat(inputImage1.rows, inputImage1.cols, inputImage1.type());

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
    /*static const float sobelKernelX[3][3] =
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

    convolution<3>(inputImage, outputDerivX, kirschX);
    convolution<3>(inputImage, outputDerivY, sobelKernelY);
    convolution<3>(inputImage, outputDerivY, sobelKernelY);
    convolution<3>(inputImage, outputDerivY, sobelKernelY);*/
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

void prewittFilter(const cv::Mat& inputImage, cv::Mat& outputDerivX, cv::Mat& outputDerivY)
{
    static const float prewittKernelX[3][3] =
    {
        {1, 0, -1},
        {1, 0, -1},
        {1, 0, -1}
    };

    static const float prewittKernelY[3][3] =
    {
        {1, 1, 1},
        {0, 0, 0},
        {-1, -1, -1}
    };

    outputDerivX = cv::Mat(inputImage.rows, inputImage.cols, inputImage.type());
    outputDerivY = cv::Mat(inputImage.rows, inputImage.cols, inputImage.type());

    convolution<3>(inputImage, outputDerivX, prewittKernelX);
    convolution<3>(inputImage, outputDerivY, prewittKernelY);
}
