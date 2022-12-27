#include <algorithm>
#include <unordered_set>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"

typedef cv::Mat_<unsigned char> OpenCVGrayscaleMat;

class RegionGrowing {
public:
    struct Seed {
        Seed(unsigned int x, unsigned int y, int val) : position_x(x), position_y(y), value(val) {}

        unsigned int position_x;
        unsigned int position_y;

        int value;
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
     * @param nb_seeds Le nombre de germe qui sera placé dans l'image
     */
    void placeSeedsRandom(const unsigned int nb_seeds_per_row);

    /**
     * Lance la segmentation de l'image (grossissement des germes).
     * Cette segmentation va utiliser une valeur seuil pour savoir si oui
     * ou non un germe doit grossir sur un pixel adjacent ou non.
     * Si la différence de valeur entre les deux pixels est inférieure ou égale
     * au treshold, le pixel sera pris en compte
     */
    void segmentationDifference(const unsigned int treshold);

    /**
     * S'assure que les listes d'adjacence de deux régions adjacentes se contiennent bien l'une
     * l'autre. Dû à la façon dont les adjacences sont construites pendant l'étalement des germes,
     * il se peut qu'une région A soit adjacente a une région B (en se fiant à m_regions_adjacency)
     * mais que B ne soit pas adjacente à A (en se fiant toujours à m_regions_adjacency).
     */
    void normalizeAdjacency();

    /**
     * Retourne true si la région de valeur 'regionAValue' est adjacente à la région de valeur 
     * 'regionBValue'
     */
    bool isRegionAdjacent(int regionAValue, int regionBValue);

    void regionFusion(const unsigned int treshold);

    void printRegionMatrix();
    void printRegionsAdjacency();

    void showSegmentation(std::string windowName, bool showInitialsSeeds);
    void showSeeds(cv::Mat* image);

private:
    bool m_seeds_placed;
    bool m_regions_placed;

    OpenCVGrayscaleMat* m_image;

    int** m_region_matrix;
    int m_nb_regions; //Nombre de régions / germes placés

    // Stocke les couleurs associées aux régions
    std::vector<std::vector<int>> distinct_colors = {
        { 255, 179, 0 }, { 128, 62, 117 },
        { 255, 104, 0 }, { 166, 189, 215 },
        { 193, 0, 32 }, { 206, 162, 98 },
        { 129, 112, 102 }
    };

    //Stocke les positions des germes initiaux
    std::vector<std::pair<unsigned int, unsigned int>> m_seeds_positions;

    //Une liste de N ensembles. N est le nombre de germes posé au départ de l'algorithme.
    //L'ensemble d'index 'i' contient les valeurs des germes des régions qui sont adjacents à la région 'i'
    // 
    //Par exemple, m_region_adjacency[0] contient les valeurs des régions adjacentes à la région
    //de valeur 0
    //Ainsi si m_regions_adjacency[0] = { 1, 2, 3 }, cela signifie que tous les germes de valeurs 1, 2 ou 3
    //dans la m_region_matrix forment des régions (les régions 1, 2 et 3) qui sont adjacentes à la région 0
    std::vector<std::unordered_set<int>> m_regions_adjacency;
};
