#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"

#include "benchmark.hpp"
#include "regionGrowing.hpp"

#define TEST_IMAGE_WIDTH 10
#define TEST_IMAGE_HEIGHT 9

int main() {
    std::string inputImagePath = "exampleImage.png";

    OpenCVGrayscaleMat inputImage = cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);
    //cv::Mat inputImage = cv::imread(inputImagePath, cv::IMREAD_COLOR);

    if(inputImage.empty()) {
        std::cout << "Impossible d'ouvrir l'image\n";

        return 0;
    }

    OpenCVGrayscaleMat testImage(TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH);
    for(int i = 0; i < TEST_IMAGE_HEIGHT; i++) {
        for(int j = 0; j < TEST_IMAGE_WIDTH; j++) {
            testImage(i, j) = 0;
        }
    }

    RegionGrowingDifference regionGrowing(&inputImage);

    std::vector<std::pair<unsigned int, unsigned int>> positionsSimpleImageGrayscale;
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(88, 52));

    //{ Benchmark benchmark("Blur time"); regionGrowing.blur(7, 3.5); }
    regionGrowing.placeSeedsManual(positionsSimpleImageGrayscale);
    //regionGrowing.placeSeedsRandom(24, true);
    { Benchmark benchmark("Segmentation time"); regionGrowing.segmentation(30); }
    regionGrowing.showSegmentation("Segmentation before fusion", true);
    { Benchmark benchmark("Region fusion time"); regionGrowing.regionFusion(30); }
    regionGrowing.showSegmentation("Segmentation after fusion", true);
    //regionGrowing.removeNoise(100);
    regionGrowing.showSegmentation("Segmentation after noise removal", true);
    regionGrowing.showRegionBorders("Bordure des regions", true);

    cv::waitKey(0);
}
