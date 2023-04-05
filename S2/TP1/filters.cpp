#include "filters.hpp"

#include <iostream>

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

                    //Si on est en train de d�passer des bords de l'image
                    if (y_pos < 0 || y_pos >= inputImage.rows || x_pos < 0 || x_pos >= inputImage.cols)
                        //On va considerer que la valeur du pixel est la meme que celle du pixel courant
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

/**
 * N is the size of the kernel for the convolution
 * T is the type of the pixels of the output image
 */
template <size_t N, typename T>
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

            if (x_img == 157 && y_img == 58)
                std::cout << "pixel_value: " << (int)inputImage.at<unsigned char>(y_img, x_img) << std::endl;

            float new_pixel_value = 0;
            for (int y_kernel = 0; y_kernel < N; y_kernel++) {
                for (int x_kernel = 0; x_kernel < N; x_kernel++) {
                    int y_kernel_shift = y_kernel - half_kernel_size;
                    int x_kernel_shift = x_kernel - half_kernel_size;

                    int y_pos = y_img + y_kernel_shift;
                    int x_pos = x_img + x_kernel_shift;

                    T pixel_value;

                    //Si on est en train de depasser des bords de l'image
                    if (y_pos < 0 || y_pos >= inputImage.rows || x_pos < 0 || x_pos >= inputImage.cols)
                        //On va considerer que la valeur du pixel est la meme que celle du pixel courant
                        pixel_value = current_pixel_value;
                    else
                        pixel_value = inputImage.at<unsigned char>(y_pos, x_pos);

                    new_pixel_value += pixel_value * kernel[y_kernel][x_kernel];
                }
            }

            float new_value = (new_pixel_value / kernel_sum);
            outputImage.at<T>(y_img, x_img) = new_value;
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

    //Pour etre sur que la somme des valeurs du noyau = 1
    for (unsigned int y = 0; y < kernel_size; y++) {
        for (unsigned int x = 0; x < kernel_size; x++) {
            kernel[y][x] /= kernel_sum;
        }
    }
}

void gaussianBlur(const cv::Mat& inputImage, cv::Mat& outputImage, unsigned int kernel_size, float sigma) {

    if (kernel_size == 0 || (kernel_size % 2 == 0)) {
        std::cout << "The kernel size needs to be stricly positive and odd.";

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

                    //Si on est en train de d�passer des bords de l'image
                    if (y_pos < 0 || y_pos >= rows || x_pos < 0 || x_pos >= cols) {
                        //On va consid�rer que la valeur du pixel est la m�me que celle du pixel courant
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

void gradientDirection(const cv::Mat& derivX, const cv::Mat& derivY, cv::Mat& gradientDir)
{
    gradientDir = cv::Mat(derivX.rows, derivX.cols, CV_8UC3);
    float min = INFINITY, max = -INFINITY;
    float minatan = INFINITY, maxatan = -INFINITY;

    for (int i = 0; i < derivX.rows; i++)
        for (int j = 0; j < derivY.cols; j++)
        {
            if (derivY.at<short int>(i, j) == 0 && derivX.at<short int>(i, j) == 0)
            {
                gradientDir.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);

                continue;
            }

            if (j == 159 && i == 67)
            {
                std::cout << "y: " << (float)derivY.at<short int>(i, j) << ", x: " << (float)derivX.at<short int>(i, j) << std::endl;
            }

            float hsl_angle = (std::atan2((float)derivY.at<short int>(i, j), (float)derivX.at<short int>(i, j)) + M_PI) / 2 / M_PI * 360;
            cv::Vec3b hsl_value = cv::Vec3b(hsl_angle / 2, 127, 255);

            cv::Mat hsl_mat = cv::Mat(1, 1, CV_8UC3); hsl_mat.at<cv::Vec3b>(0, 0) = hsl_value;
            cv::Mat bgr_mat = cv::Mat(1, 1, CV_8UC3);
            cv::cvtColor(hsl_mat, bgr_mat, cv::COLOR_HLS2BGR);

            gradientDir.at<cv::Vec3b>(i, j) = cv::Vec3b(hsl_angle / 2, 127, 255);

            float angleatan = std::atan2((float)derivY.at<short int>(i, j), (float)derivX.at<short int>(i, j));
            if (minatan > angleatan)
                minatan = angleatan;
            if (maxatan < angleatan)
                maxatan = angleatan;

            if (min > hsl_angle)
                min = hsl_angle;
            if( max < hsl_angle)
                max = hsl_angle;
        }

    std::cout << "min hsl angle: " << min << ", max hsl angle: " << max << std::endl;
    std::cout << "min atan: " << minatan << ", max atan: " << maxatan << std::endl;

    cv::cvtColor(gradientDir, gradientDir, cv::COLOR_HLS2BGR);
}

void gradientMagnitudeNormalized(const cv::Mat& derivX, const cv::Mat& derivY, cv::Mat& gradientMagnitude)
{
    gradientMagnitude = cv::Mat(derivX.rows, derivX.cols, CV_8U);

    unsigned short int* temp_nonnormalized = new unsigned short int[gradientMagnitude.rows * gradientMagnitude.cols];

    float max_value = -INFINITY;
    for (int i = 0; i < derivX.rows; i++)
    {
        for (int j = 0; j < derivX.cols; j++)
        {
            float x = (float)derivX.at<short int>(i, j);
            float y = (float)derivY.at<short int>(i, j);

            float magnitude = std::sqrt(x * x + y * y);
            if (max_value < magnitude)
                max_value = magnitude;

            temp_nonnormalized[i * derivX.cols + j] = magnitude;
        }
    }

    for (int i = 0; i < derivX.rows; i++)
    {
        for (int j = 0; j < derivX.cols; j++)
        {
            float value = temp_nonnormalized[i * derivX.cols + j] / max_value;

            gradientMagnitude.at<unsigned char>(i, j) = value * 255;
        }
    }

    delete[] temp_nonnormalized;
}

void angleMatrix(const cv::Mat& derivX, const cv::Mat& derivY, cv::Mat& angle_matrix)
{
    angle_matrix = cv::Mat(derivX.rows, derivX.cols, CV_8U);

    for (int i = 0; i < derivX.rows; i++)
    {
        for (int j = 0; j < derivX.cols; j++)
        {
            unsigned char y, x;
            x = derivX.at<unsigned char>(i, j);
            y = derivY.at<unsigned char>(i, j);

            //Angle between 0 and 180 degrees
            angle_matrix.at<unsigned char>(i, j) = (std::atan2(y, x) + M_PI) / 2 * 180;
        }
    }
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

void low_treshold(cv::Mat& input_image, cv::Mat& output_image, unsigned char threshold)
{
    output_image = cv::Mat(input_image.rows, input_image.cols, input_image.type());

    for (int i = 0; i < input_image.rows; i++)
    {
        for (int j = 0; j < input_image.cols; j++)
        {
            unsigned char value = input_image.at<unsigned char>(i, j);
            output_image.at<unsigned char>(i, j) = (value >= threshold ? value : 0);
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

void multiply_rgb_by_grayscale(const cv::Mat& input_rgb, const cv::Mat& input_grayscale, cv::Mat& output_image)
{
    output_image = cv::Mat(input_rgb.rows, input_rgb.cols, input_rgb.type());

    for (int i = 0; i < input_rgb.rows; i++)
    {
        for (int j = 0; j < input_rgb.cols; j++)
        {
            //std::cout << input_grayscale.at<unsigned char>(i, j) / 255.0f << std::endl;
            cv::Vec3b new_val = input_rgb.at<cv::Vec3b>(i, j) * (input_grayscale.at<unsigned char>(i, j) / 255.0f);


            output_image.at<cv::Vec3b>(i, j) = new_val;
        }
    }
}

void binarize(const cv::Mat& edge_image, cv::Mat& out_binarized)
{
    out_binarized = cv::Mat(edge_image.rows, edge_image.cols, edge_image.type());

    for (int i = 0; i < edge_image.rows; i++)
        for (int j = 0; j < edge_image.cols; j++)
            out_binarized.at<unsigned char>(i, j) = edge_image.at<unsigned char>(i, j) > 0 ? 255 : 0;
}

void normalize_grayscale_image(const cv::Mat& input_image, cv::Mat& output_image)
{
    double min_val, max_val;

    cv::minMaxLoc(input_image, &min_val, &max_val);

    output_image = cv::Mat(input_image.rows, input_image.cols, input_image.type());
    output_image = input_image / max_val * 255;
}

void normalize_grayscale_u16_to_u8(const cv::Mat& u16_image, cv::Mat& u8_image_normalized)
{
    double min_val, max_val;

    cv::minMaxLoc(u16_image, &min_val, &max_val);

    u8_image_normalized = cv::Mat(u16_image.rows, u16_image.cols, CV_8U);

    for (int i = 0; i < u16_image.rows; i++)
    {
        for (int j = 0; j < u16_image.cols; j++)
        {
            u8_image_normalized.at<unsigned char>(i, j) = u16_image.at<unsigned short int>(i, j) / max_val * 255;
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

void non_maximum_suppresion(const cv::Mat& gradientIntensityNorm, const cv::Mat& angle_matrix, cv::Mat& non_maxi_suppressed)
{
    non_maxi_suppressed = cv::Mat(gradientIntensityNorm.rows, gradientIntensityNorm.cols, gradientIntensityNorm.type());
    non_maxi_suppressed.setTo(cv::Scalar(0, 0, 0));

    for (int i = 1; i < gradientIntensityNorm.rows - 1; i++)
    {
        for (int j = 1; j < gradientIntensityNorm.cols - 1; j++)
        {
            unsigned char current_pixel = gradientIntensityNorm.at<unsigned char>(i, j);
            unsigned char angle = angle_matrix.at<unsigned char>(i, j);

            unsigned char left_pixel = 0, right_pixel = 0;
            if ((angle > 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180))//Edge direction = horizontal
            {
                left_pixel = gradientIntensityNorm.at<unsigned char>(i, j - 1);
                right_pixel = gradientIntensityNorm.at<unsigned char>(i, j + 1);
            }
            else if (angle >= 22.5 && angle < 67.5)//Edge direction = NW / SE
            {
                left_pixel = gradientIntensityNorm.at<unsigned char>(i + 1, j - 1);
                right_pixel = gradientIntensityNorm.at<unsigned char>(i - 1, j + 1);
            }
            else if (angle >= 67.5 && angle < 112.5)//Edge direction = vertical
            {
                left_pixel = gradientIntensityNorm.at<unsigned char>(i + 1, j);
                right_pixel = gradientIntensityNorm.at<unsigned char>(i - 1, j);
            }
            else if (angle >= 112.5 && angle < 157.5)//Edge direction = NE / SW
            {
                left_pixel = gradientIntensityNorm.at<unsigned char>(i - 1, j - 1);
                right_pixel = gradientIntensityNorm.at<unsigned char>(i + 1, j + 1);
            }

            if (left_pixel > current_pixel || right_pixel > current_pixel)
                non_maxi_suppressed.at<unsigned char>(i, j) = 0;
            else
                non_maxi_suppressed.at<unsigned char>(i, j) = current_pixel;
        }
    }
}

void double_threshold(const cv::Mat& non_maxi_suppressed, float low_threshold, float high_threshold, cv::Mat& double_thresholded)
{
    double_thresholded = cv::Mat(non_maxi_suppressed.rows, non_maxi_suppressed.cols, CV_8U);
    for (int i = 0; i < non_maxi_suppressed.rows; i++)
    {
        for (int j = 0; j < non_maxi_suppressed.cols; j++)
        {
            unsigned char current_pixel = non_maxi_suppressed.at<unsigned char>(i, j);

            double_thresholded.at<unsigned char>(i, j) = current_pixel < low_threshold ? 0 : (current_pixel > high_threshold ? 255 : 127);
        }
    }
}

void hysteresis(const cv::Mat& double_tresholded, cv::Mat& output_hysteresis)
{
    output_hysteresis = cv::Mat(double_tresholded.rows, double_tresholded.cols, CV_8U);
    output_hysteresis.setTo(cv::Scalar(0, 0, 0));

    for (int i = 1; i < double_tresholded.rows - 1; i++)
    {
        for (int j = 1; j < double_tresholded.cols - 1; j++)
        {
            unsigned char current_pixel = double_tresholded.at<unsigned char>(i, j);

            //Weak edge that needs to be tresholded
            if (double_tresholded.at<unsigned char>(i, j) == 127)
            {
                //Looking for a strong pixel in the vicinity of the current pixel
                bool found = false;
                for (int ii = i - 1; ii <= i + 1; ii++)
                {
                    for (int jj = j - 1; jj <= j + 1; jj++)
                    {
                        //Is it a strong pixel ?
                        if (double_tresholded.at<unsigned char>(ii, jj) == 255)
                        {
                            output_hysteresis.at<unsigned char>(i, j) = 255;

                            found = true;

                            break;
                        }
                    }

                    if (found)
                        break;
                }

                //We didn't find a strong pixel around the current pixel
                if (!found)
                    output_hysteresis.at<unsigned char>(i, j) = 0;
            }
            else
                output_hysteresis.at<unsigned char>(i, j) = current_pixel;
        }
    }
}

void cannyEdgeDetection(const cv::Mat& inputImagePreprocessed, unsigned char low_threshold, unsigned char high_threshold, cv::Mat& outputCanny)
{
    cv::Mat derivX, derivY;
    sobelFilter(inputImagePreprocessed, derivX, derivY);


    cv::Mat gradientIntensity, gradientIntensityNorm;
    gradientMagnitudeNormalized(derivX, derivY, gradientIntensity);
    normalize_grayscale_image(gradientIntensity, gradientIntensityNorm);


    cv::Mat angle_matrix, non_maxi_suppressed;
    angleMatrix(derivX, derivY, angle_matrix);
    non_maximum_suppresion(gradientIntensityNorm, angle_matrix, non_maxi_suppressed);


    cv::Mat double_thresholded;
    double_threshold(non_maxi_suppressed, low_threshold, high_threshold, double_thresholded);
    hysteresis(double_thresholded, outputCanny);
}

void houghTransform(const cv::Mat& binarized_edge_image, int nb_theta, int nb_rho, cv::Mat& hough_space, cv::Mat& output_lines)
{
    float img_hypot = std::hypot(binarized_edge_image.rows, binarized_edge_image.cols);
    float theta_step = 180.0f / nb_theta;

    hough_space = cv::Mat(nb_theta, nb_rho, CV_16U);
    hough_space.setTo(cv::Scalar(0, 0, 0));

    for (int i = 0; i < binarized_edge_image.rows; i++)
    {
        for (int j = 0; j < binarized_edge_image.cols; j++)
        {
            //Is this is an edge point
            if (binarized_edge_image.at<unsigned char>(i, j) == 255)
            {
                for (unsigned int theta_index = 0; theta_index < nb_theta; theta_index++)
                {
                    float theta = theta_step * theta_index;
                    float rho = j * std::cos(theta / 180 * M_PI) + i * std::sin(theta / 180 * M_PI);

                    int rho_index = std::abs(rho) / img_hypot * nb_rho;

                    hough_space.at<unsigned short int>(theta_index, rho_index)++;
                }
            }
        }
    }

    double minValue, maxValue;
    cv::minMaxLoc(hough_space, &minValue, &maxValue);

    float threshold = maxValue * 0.75;
    std::vector<float> rhos, thetas;//Rhos and thetas of the lines detected in the image

    for (int i = 0; i < hough_space.rows; i++)
    {
        for (int j = 0; j < hough_space.cols; j++)
        {
            if (hough_space.at<unsigned short int>(i, j) >= threshold)
            {
                rhos.push_back((float)j / nb_rho * img_hypot * 2 - img_hypot);
                thetas.push_back((float)i / nb_theta * 180);
            }
        }
    }

    output_lines = cv::Mat(binarized_edge_image.rows, binarized_edge_image.cols, binarized_edge_image.type());
    output_lines.setTo(cv::Scalar(0, 0, 0));
    for (int i = 0; i < rhos.size(); i++)
    {
        float rho = rhos.at(i);
        float theta = thetas.at(i);

        std::cout << "rho, theta: " << rho << ", " << theta << std::endl;

        cv::Point a(0, rho / std::sin(theta));
        cv::Point b(binarized_edge_image.cols - 1, (rho - (binarized_edge_image.cols - 1) * std::cos(theta))/ std::sin(theta));

        if (i == rhos.size() - 1)
            cv::line(output_lines, a, b, cv::Scalar(127, 0, 0), 1);
        else
            cv::line(output_lines, a, b, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

void kirshFilter(const cv::Mat& inputImage, cv::Mat& outputImage)
{
    static const float kirschN[3][3] =
    {
        {5, 5, 5},
        {-3, 0, -3},
        {-3, -3, -3}
    };

    static const float kirschNE[3][3] =
    {
        {5, 5, -3},
        {5, 0, -3},
        {-3, -3, -3}
    };

    static const float kirschE[3][3] =
    {
        {5, -3, -3},
        {5, 0,  -3},
        {5, -3, -3}
    };

    static const float kirschSE[3][3] =
    {
        {-3, -3, -3},
        {5, 0, -3},
        {5, 5, -3}
    };

    cv::Mat outputDerivN = cv::Mat(inputImage.rows, inputImage.cols, CV_16S);
    cv::Mat outputDerivNE = cv::Mat(inputImage.rows, inputImage.cols, CV_16S);
    cv::Mat outputDerivE= cv::Mat(inputImage.rows, inputImage.cols, CV_16S);
    cv::Mat outputDerivSE = cv::Mat(inputImage.rows, inputImage.cols, CV_16S);
    outputImage = cv::Mat(inputImage.rows, inputImage.cols, CV_8U);

    convolution<3, short int>(inputImage, outputDerivN, kirschN);
    convolution<3, short int>(inputImage, outputDerivNE, kirschNE);
    convolution<3, short int>(inputImage, outputDerivE, kirschE);
    convolution<3, short int>(inputImage, outputDerivSE, kirschSE);

    for (int i = 0; i < outputDerivN.rows; i++)
    {
        for (int j = 0; j < outputDerivN.cols; j++)
        {
            short int N = outputDerivN.at<short int>(i, j);
            short int NE = outputDerivNE.at<short int>(i, j);
            short int E = outputDerivE.at<short int>(i, j);
            short int SE = outputDerivSE.at<short int>(i, j);

            outputImage.at<short int>(i, j) = std::max(N, std::max(NE, std::max(E, SE)));
        }
    }
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

    outputDerivX = cv::Mat(inputImage.rows, inputImage.cols, CV_16S);
    outputDerivY = cv::Mat(inputImage.rows, inputImage.cols, CV_16S);

    convolution<3, short int>(inputImage, outputDerivX, sobelKernelX);
    convolution<3, short int>(inputImage, outputDerivY, sobelKernelY);
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

    outputDerivX = cv::Mat(inputImage.rows, inputImage.cols, CV_16S);
    outputDerivY = cv::Mat(inputImage.rows, inputImage.cols, CV_16S);

    convolution<3, short int>(inputImage, outputDerivX, prewittKernelX);
    convolution<3, short int>(inputImage, outputDerivY, prewittKernelY);
}
