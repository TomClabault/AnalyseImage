#include "settings.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <locale>
#include <fstream>

// trim from start (in place)
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
                return !std::isspace(ch);
            }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
                return !std::isspace(ch);
            }).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    rtrim(s);
    ltrim(s);
}

void read_image(const std::string& inputImagePath, cv::Mat& image)
{
    cv::Mat imageRGB = cv::imread(inputImagePath, cv::IMREAD_COLOR);
    if (imageRGB.empty()) {
        std::cout << "Impossible d'ouvrir l'image...\n";
        std::exit(0);
    }

    cv::cvtColor(imageRGB, image, cv::COLOR_RGB2GRAY);
}

void read_settings(const char* filepath, Settings& settings)
{
    std::ifstream input_file(filepath);
    if (!input_file)
    {
        std::cout << "Unable to open the settings file: " << filepath << std::endl;

        std::exit(0);
    }

    settings.threshold_method = "None";
    std::string line;
    while (std::getline(input_file, line))
    {
        trim(line);

        if (line[0] == '#' || line.length() == 0)
            continue;

        std::string instruction = line.substr(0, line.find('='));
        std::string object = line.substr(line.find('=') + 1);

        if (instruction == "imageFile")
            read_image(object, settings.input_image);
        else if (instruction == "kernelType")
            settings.kernel_type = object;
        else if (instruction == "thresholdMethod")
        {
            int threshold_manual = std::atoi(object.c_str());
            if (threshold_manual != 0)
                settings.threshold_manual = threshold_manual;
            else
                settings.threshold_method = object;
        }
        else if (instruction == "thresholdAdaptativeLocalMeanNeighborhoodSize")
            settings.thresh_adapt_mean_local_neighborhood_size = std::atoi(object.c_str());
        else if (instruction == "thresholdAdaptativeLocalMeanConstant")
            settings.thresh_adapt_mean_local_constant = std::atof(object.c_str());
        else if (instruction == "preprocessBlur")
            settings.preprocess_blur = object != "False" && object != "false";
        else if (instruction == "gaussianKernelSize")
            settings.gaussian_kernel_size = std::atoi(object.c_str());
        else if (instruction == "gaussianKernelSigma")
            settings.gaussian_kernel_sigma = std::atoi(object.c_str());
        else if (instruction == "lowThresholdCanny")
            settings.canny_low_threshold = std::atoi(object.c_str());
        else if (instruction == "highThresholdCanny")
            settings.canny_high_threshold = std::atoi(object.c_str());
        else if (instruction == "houghTransformNbRho")
            settings.hough_transform_nb_rho = std::atoi(object.c_str());
        else if (instruction == "houghTransformNbTheta")
            settings.hough_transform_nb_theta = std::atoi(object.c_str());
    }
}
