#include "regionGrowing.hpp"

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
