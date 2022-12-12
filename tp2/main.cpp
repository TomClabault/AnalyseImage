#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"

#include "regionGrowing.hpp"

typedef cv::Mat_<unsigned char> OpenCVGrayscaleMat;

int main() {
    std::string inputImagePath;

    OpenOpenCVGrayscaleMat inputImage = imread(inputImagePath);

    if(inputImage.empty()) {
        std::cout << "Impossible d'ouvrir l'image\n";

        return 0;
    }
    
    RegionGrowing regionGrowing(inputImage);

    std::cout << "hello";
}