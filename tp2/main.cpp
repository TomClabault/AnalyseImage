#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"

#include "regionGrowing.hpp"

int main() {
    std::string inputImagePath;

    OpenCVGrayscaleMat inputImage = cv::imread(inputImagePath);

    if(inputImage.empty()) {
        std::cout << "Impossible d'ouvrir l'image\n";

        return 0;
    }
    
    RegionGrowing regionGrowing(&inputImage);

    std::cout << "hello";
}