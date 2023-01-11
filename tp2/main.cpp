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
    std::string inputImagePath = "lena_color.png";

    //OpenCVGrayscaleMat inputImage cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);
    cv::Mat inputImage = cv::imread(inputImagePath, cv::IMREAD_COLOR);

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
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(72, 428));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(467, 242));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(83, 348));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(379, 242));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(205, 489));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(56, 61));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(369, 438));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(145, 94));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(452, 119));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(269, 405));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(497, 474));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(185, 157));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(482, 368));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(470, 89));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(282, 466));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(23, 135));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(156, 218));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(379, 78));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(274, 236));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(371, 157));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(402, 387));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(236, 8));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(93, 257));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(204, 336));

    { Benchmark benchmark("Blur time"); regionGrowing.blur(7, 3.5); }
    regionGrowing.placeSeedsManual(positionsSimpleImageGrayscale);
    //regionGrowing.placeSeedsRandom(24, true);
    { Benchmark benchmark("Segmentation time"); regionGrowing.segmentation(30, RegionGrowing::rgb_distance_LInfinity); }
    regionGrowing.showSegmentation("Segmentation before fusion", true);
    { Benchmark benchmark("Region fusion time"); regionGrowing.regionFusion(30); }
    regionGrowing.showSegmentation("Segmentation after fusion", true);
    regionGrowing.removeNoise(100);
    regionGrowing.showSegmentation("Segmentation after noise removal", true);
    regionGrowing.showRegionBorders("Bordure des regions", true);

    cv::waitKey(0);
}
