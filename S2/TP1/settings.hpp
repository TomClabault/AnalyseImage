#ifndef SETTINGS_HPP
#define SETTINGS_HPP

#include <opencv2/core.hpp>

struct Settings
{
    cv::Mat input_image;

    std::string kernel_type;

    std::string threshold_method;
    int threshold_manual = -1;

    int thresh_adapt_mean_local_neighborhood_size = -1;
    float thresh_adapt_mean_local_constant = -INFINITY;

    int canny_low_threshold, canny_high_threshold;

    bool preprocess_blur;
    int gaussian_kernel_size;
    float gaussian_kernel_sigma;

    int hough_transform_nb_rho, hough_transform_nb_theta;
};

void read_image(const std::string& inputImagePath, cv::Mat& image);

void read_settings(const char* filepath, Settings& settings);

#endif
