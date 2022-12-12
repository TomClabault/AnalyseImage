#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"

typedef cv::Mat_<unsigned char> OpenCVGrayscaleMat;

class RegionGrowing {
public:
    RegionGrowing(OpenCVGrayscaleMat* image);

    /**
     * Positionne les germes qui seront utilisés pour le region growing.
     * Cette méthode positionne nb_seeds_per_row**2 germes
     * 
     * @param nb_seeds_per_row
     */
    void placeSeeds(unsigned int nb_seeds_per_row);

    /**
     * Lance la segmentation de l'image (grossissement des germes)
     */
    void segmentation();
    void regionFusion();

private:
    bool m_seedsPlaced;

    OpenCVGrayscaleMat* m_image;

    unsigned int** region_matrix;
};
