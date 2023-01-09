#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"

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

        std::cout << std::endl;
    }

    RegionGrowing regionGrowing(&inputImage);

    std::vector<std::pair<unsigned int, unsigned int>> positionsSimpleImageGrayscale;
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(386, 344));
    /*positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(115, 69));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(125, 138));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(112, 218));
    positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(35, 168));*/

    std::vector<std::pair<unsigned int, unsigned int>> positionsTestImage;
    positionsTestImage.push_back(std::pair<unsigned int, unsigned int>(0, 0));
    positionsTestImage.push_back(std::pair<unsigned int, unsigned int>(1, 0));
    positionsTestImage.push_back(std::pair<unsigned int, unsigned int>(7, 7));

    //regionGrowing.blur(9, 1);
    //regionGrowing.placeSeedsManual(positionsSimpleImageGrayscale);
    regionGrowing.placeSeedsRandom(12);
    regionGrowing.segmentationDifference(5);
    regionGrowing.showSegmentation("Segmentation before fusion", true);
    regionGrowing.regionFusion(10);
    regionGrowing.showSegmentation("Segmentation after fusion", true);
    //regionGrowing.removeNoise(100);
    regionGrowing.showSegmentation("Segmentation after noise removal", true);
    regionGrowing.showRegionBorders("Bordure des regions", true);
    cv::waitKey(0);
}