#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void readMask(unsigned int& maskSize, float*** kernel, std::string maskFilePath)
{
    std::ifstream maskFile(maskFilePath);

    maskFile >> maskSize;
    *kernel = (float**) malloc(sizeof(float*) * maskSize);
    for (unsigned int i = 0; i < maskSize; i++)
        (*kernel)[i] = (float*) malloc(sizeof(float) * maskSize);

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

    cv::imshow("input", imageRGB);
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

            outputImage.at<unsigned char>(y_img, x_img) = (unsigned char)(new_pixel_value / kernel_sum);
        }
    }
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

int main(int argc, char** argv)
{
    std::string maskFilePath;
    std::string inputImagePath;

    unsigned int kernelXSize, kernelYSize;
    float** kernelX, **kernelY;
    unsigned int treshold;

    cv::Mat inputImage;
    cv::Mat outputImageX, outputImageY;

    if (argc < 5)
    {
        std::cout << "Usage: ./main <imageFile> <kernelFileX> <kernelFileY> <treshold>" << std::endl;

        return -1;
    }

    readImage(inputImage, argv[1]);
    readMask(kernelXSize, &kernelX, argv[2]);
    readMask(kernelYSize, &kernelY, argv[3]);
    treshold = atoi(argv[4]);

    outputImageX = cv::Mat(inputImage.rows, inputImage.cols, inputImage.type());
    outputImageY = cv::Mat(inputImage.rows, inputImage.cols, inputImage.type());
    convolution(inputImage, outputImageX, kernelX, kernelXSize);
    convolution(inputImage, outputImageY, kernelY, kernelYSize);
    tresholding(outputImageX, outputImageX, treshold);
    tresholding(outputImageY, outputImageY, treshold);

    cv::Mat sumXY = cv::Mat(inputImage.rows, inputImage.cols, inputImage.type());
    sumImages(sumXY, outputImageX, outputImageY);

    cv::imshow("input", outputImageY);
    cv::waitKey(0);
}
