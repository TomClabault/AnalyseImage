#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"

#include "regionGrowing.hpp"

#define TEST_IMAGE_HEIGHT 8
#define TEST_IMAGE_WIDTH 8

int main() {
    std::string inputImagePath = "simpleImageGrayscale.png";

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

    RegionGrowing regionGrowing(&inputImage);

    std::vector<std::pair<unsigned int, unsigned int>> positionsSimpleImageGrayscale;
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(26, 53));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(83, 43));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(100, 93));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(43, 134));

    regionGrowing.placeSeedsManual(positionsSimpleImageGrayscale);
    regionGrowing.segmentationDifference(5);
    regionGrowing.showSegmentation();
}