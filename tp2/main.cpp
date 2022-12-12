#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"

#include "regionGrowing.hpp"

#define TEST_IMAGE_HEIGHT 8
#define TEST_IMAGE_WIDTH 8

int main() {
    std::string inputImagePath = "lena_gray.bmp";

    OpenCVGrayscaleMat inputImage = cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);

    if(inputImage.empty()) {
        std::cout << "Impossible d'ouvrir l'image\n";

        return 0;
    }

    OpenCVGrayscaleMat testImage(TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT);
    for(int i = 0; i < TEST_IMAGE_HEIGHT; i++) {
        for(int j = 0; j < TEST_IMAGE_WIDTH; j++) {
            testImage(i, j) = i * TEST_IMAGE_WIDTH + j + 1;

            std::cout << (unsigned int)testImage(i, j) << ", ";
        }

        std::cout << std::endl;
    }

    RegionGrowing regionGrowing(&testImage);

    std::vector<std::pair<unsigned int, unsigned int>> positions;
    positions.push_back(std::pair<unsigned int, unsigned int>(0, 0));
    positions.push_back(std::pair<unsigned int, unsigned int>(2, 2));
    positions.push_back(std::pair<unsigned int, unsigned int>(7, 7));

    regionGrowing.placeSeedsManual(positions);
    regionGrowing.segmentationDifference(5);
}