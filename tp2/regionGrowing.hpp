#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"

typedef cv::Mat_<unsigned char> OpenCVGrayscaleMat;

class RegionGrowing {
public:
    struct Seed {
        Seed(unsigned int x, unsigned int y, unsigned int parent) : position_x(x), position_y(y), parent_value(parent) {}

        unsigned int position_x;
        unsigned int position_y;

        //Valeur du germe duquel est apparu le germe actuel.
        //Si le germe de départ avait pour valeur 38 dans l'image
        //et que en s'étendant, on est arrivé au germe actuel (le germe décrit
        //par cette strcture), alors la parent_value = 38 aussi
        unsigned int parent_value;
    };

    RegionGrowing(OpenCVGrayscaleMat* image);
    ~RegionGrowing();

    /**
     * Positionne les germes manuellement à partir de la liste des positions des germes
     * donnée
     * @param positionsList La liste des positions des germes à placer. 
     * Les positions doivent être de la forme (x, y)
     * La liste doit donc être de la forme [(x1, y1), (x2, y2), ...]
     */
    void placeSeedsManual(std::vector<std::pair<unsigned int, unsigned int>> positionsList);

    /**
     * Positionne les germes qui seront utilisés pour le region growing.
     * Cette méthode positionne les germes aléatoirement dans l'image de
     * façon à obtenir une bonne couverture de l'image
     * 
     * @param nb_seeds
     */
    void placeSeedsRandom(unsigned int nb_seeds_per_row);

    /**
     * Lance la segmentation de l'image (grossissement des germes).
     * Cette segmentation va utiliser une valeur seuil pour savoir si oui
     * ou non un germe doit grossir sur un pixel adjacent ou non.
     * Si la différence de valeur entre les deux pixels est inférieure ou égale
     * au treshold, le pixel sera pris en compte
     */
    void segmentationDifference(unsigned int treshold);
    void regionFusion();

    void printRegionMatrix();
    
    void showSegmentation();

private:
    bool m_seeds_placed;

    OpenCVGrayscaleMat* m_image;

    int** m_region_matrix;

    //Stocke les positions des germes initiaux
    std::vector<std::pair<unsigned int, unsigned int>> m_seeds_positions;

    //Stocke les valeurs des germes (permet de savoir à quel région tel ou tel)
    //germe appartient
    std::vector<unsigned int> seeds_values;
};
