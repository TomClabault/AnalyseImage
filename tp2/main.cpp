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
    std::string inputImagePath = "lena_gray.bmp";

    OpenCVGrayscaleMat inputImage = cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);

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

    /*
    RegionGrowingAverage regionGrowing(&inputImage);

    std::vector<std::pair<unsigned int, unsigned int>> positionsSimpleImageGrayscale;
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(383, 143));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(118, 426));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(322, 272));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(474, 145));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(152, 172));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(106, 368));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(272, 455));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(209, 343));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(329, 19));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(450, 314));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(181, 77));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(96, 71));

    { Benchmark bench("Blur time"); regionGrowing.blur(5, 1); }
    regionGrowing.placeSeedsManual(positionsSimpleImageGrayscale);
    //regionGrowing.placeSeedsRandom(20);
    { Benchmark bench("Segmentation time"); regionGrowing.segmentation(15); }
    regionGrowing.showSegmentation("Segmentation before fusion", true);
    regionGrowing.regionFusion(15);
    regionGrowing.showSegmentation("Segmentation after fusion", true);
    regionGrowing.removeNoise(100);
    regionGrowing.showSegmentation("Segmentation after noise removal", true);
    regionGrowing.showRegionBorders("Bordure des regions", true);
    cv::waitKey(0);
    */

    RegionGrowingDifference regionGrowing(&inputImage);

    std::vector<std::pair<unsigned int, unsigned int>> positionsSimpleImageGrayscale;
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(383, 143));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(118, 426));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(322, 272));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(474, 145));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(152, 172));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(106, 368));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(272, 455));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(209, 343));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(329, 19));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(450, 314));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(181, 77));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(96, 71));

    //regionGrowing.blur(7, 1);
    regionGrowing.placeSeedsManual(positionsSimpleImageGrayscale);
    //regionGrowing.placeSeedsRandom(12);
    regionGrowing.segmentation(5);
    regionGrowing.showSegmentation("Segmentation before fusion", true);
    regionGrowing.regionFusion(10);
    regionGrowing.showSegmentation("Segmentation after fusion", true);
    regionGrowing.removeNoise(100);
    regionGrowing.showSegmentation("Segmentation after noise removal", true);
    regionGrowing.showRegionBorders("Bordure des regions", true);
    cv::waitKey(0);
}