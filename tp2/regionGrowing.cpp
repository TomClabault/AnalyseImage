#include "regionGrowing.hpp"
#include <vector>

RegionGrowing::RegionGrowing(OpenCVGrayscaleMat* image) {
    m_image = image;

    //Création de la matrice de la même taille que l'image pour accueillir les données
    //du région growing
    region_matrix = (unsigned int**)malloc(sizeof(unsigned int) * image->rows * image->cols);
}

void RegionGrowing::placeSeeds(unsigned int nb_seeds_per_row) {
    m_seedsPlaced = true;


}

void RegionGrowing::segmentation() {
    //On vérifie bien que les seeds sont placées avant de commencer
    if(!m_seedsPlaced) {
        return;
    }

    
}

void RegionGrowing::regionFusion() {

}

void randomRGBColor(int rgb[])
{
    for(int i = 0; i < 3; i++) {
        rgb[i] = rand()%256;
    }
}

void RegionGrowing::showSegmentation() {
    std::vector<std::vector<int>> distinct_colors = {
        { 255, 179, 0 }, { 128, 62, 117 },
        { 255, 104, 0 }, { 166, 189, 215 },
        { 193, 0, 32 }, { 206, 162, 98 },
        { 129, 112, 102 }
    };
    cv::Mat regions_img = cv::Mat::zeros(m_image->rows, m_image->cols, CV_8UC3);


    for (int i = 0; i < regions_img.rows; i++) {
        for (int j = 0; j < regions_img.cols; j++) {
            int val = m_image->at<uchar>(i, j);
            if (val < (int)distinct_colors.size()) {
                regions_img.at<cv::Vec3b>(i, j)[0] = distinct_colors[val][0];
                regions_img.at<cv::Vec3b>(i, j)[1] = distinct_colors[val][1];
                regions_img.at<cv::Vec3b>(i, j)[2] = distinct_colors[val][2];
            }
            else {
                int rgb[3];
                randomRGBColor(rgb);
                distinct_colors.push_back({rgb[0], rgb[1], rgb[2]});
                regions_img.at<cv::Vec3b>(i, j)[0] = rgb[0];
                regions_img.at<cv::Vec3b>(i, j)[1] = rgb[1];
                regions_img.at<cv::Vec3b>(i, j)[2] = rgb[2];
            }
        }
    }

    cv::imshow("Segmentation image", regions_img);
}