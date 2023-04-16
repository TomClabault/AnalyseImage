#ifndef SETTINGS_HPP
#define SETTINGS_HPP

#include <opencv2/core.hpp>

struct Settings
{
    cv::Mat input_image;

    std::string kernel_type;

    std::string threshold_method;
    int threshold_manual = -1;
    int hysteresis_low_threshold;
    int hysteresis_high_threshold;

    int thresh_adapt_mean_local_neighborhood_size = -1;
    float thresh_adapt_mean_local_constant = -INFINITY;

    int canny_low_threshold, canny_high_threshold;

    bool preprocess_blur;
    int gaussian_kernel_size;
    float gaussian_kernel_sigma;

    int hough_transform_nb_rho, hough_transform_nb_theta;
    float hough_transform_threshold;//Threshold between 0 and 1 that will
    //be used to determine whether a given line (rho/theta pair) will be kept
    //for the final result or not. A line is accepted if it has received at least
    //"threshold * max_vote" votes. max_vote being the maximum number of vote
    //that any line received in the image
};

void read_image(const std::string& inputImagePath, cv::Mat& image);

void read_settings(const char* filepath, Settings& settings);

#endif
