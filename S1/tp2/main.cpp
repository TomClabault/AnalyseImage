#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"

#include "benchmark.hpp"
#include "regionGrowing.hpp"

int main() {
    std::string inputImagePath = "images/000009.jpg";

    //OpenCVGrayscaleMat inputImage = cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);
    cv::Mat inputImage = cv::imread(inputImagePath, cv::IMREAD_COLOR);

    cv::imshow("Image d'origine", inputImage);

    if(inputImage.empty()) {
        std::cout << "Impossible d'ouvrir l'image\n";

        return 0;
    }

    RegionGrowingAverage regionGrowing(&inputImage);

    std::vector<std::pair<unsigned int, unsigned int>> positionsSimpleImageGrayscale;
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(283, 431));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(120, 366));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(328, 184));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(48, 254));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(487, 46));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(493, 317));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(202, 230));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(210, 372));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(227, 162));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(501, 446));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(449, 263));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(391, 31));

    { Benchmark benchmark("Blur time"); regionGrowing.blur(5, 1); }
    //regionGrowing.placeSeedsManual(positionsSimpleImageGrayscale);
    regionGrowing.placeSeedsRandom(64);
    { Benchmark benchmark("Segmentation time"); regionGrowing.segmentation(50, RegionGrowing::rgb_distance_L1); }
    regionGrowing.showSegmentation("Segmentation before fusion", false);
    { Benchmark benchmark("Region fusion time"); regionGrowing.regionFusion(60); }
    regionGrowing.showSegmentation("Segmentation after fusion", false);
    regionGrowing.removeNoise(100);
    regionGrowing.showSegmentation("Segmentation after noise removal", false);
    regionGrowing.showRegionBorders("Bordure des regions", true);

    cv::waitKey(0);
}
