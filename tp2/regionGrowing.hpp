#include <algorithm>
#include <unordered_set>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"

#define PI 3.141592653589793238462643383279502884

typedef cv::Mat_<unsigned char> OpenCVGrayscaleMat;

class RegionGrowing {
public:
    struct Seed {
        Seed(unsigned int x, unsigned int y, int val, int region_number) : position_x(x), position_y(y), value(val), region(region_number) {}

        unsigned int position_x;
        unsigned int position_y;

        int value;//La valeur de la seed
        int region;//Le numero de la region à laquelle appartient la seed
    };

    struct SeedRGB {
        SeedRGB(unsigned int x, unsigned int y, cv::Vec3b val, int region_number) : position_x(x), position_y(y), value(val), region(region_number) {}

        unsigned int position_x;
        unsigned int position_y;

        cv::Vec3b value;//La valeur de la seed
        int region;//Le numero de la region à laquelle appartient la seed
    };

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
     * Floute l'image dans le but d'éliminer le bruit afin de rendre la segmentation plus efficace 
     * 
     * @param kernel_size La taille du noyau gaussien qui va être utilisé pour flouter l'image
     * @param sigma Le paramètre sigma du noyau gaussien
     */
    void blur(unsigned int kernel_size, double sigma);

    /*
     * Floute l'image en niveaux de gris. Voir doc de 'blur'
     */
    void blurGrayscale(double** kernel, int kernel_size);

    /*
     * Floute l'image en RGB. Voir doc de 'blur'
     */
    void blurRGB(double** kernel, int kernel_size);

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

    /**
     * Teste si un pixel donné est en bordure de sa région ou non (en connexité 4). 
     * L'image doit avoir préalalement été segmentée avant d'appeler cette fonction
     * @param pixel_value La valeur du pixel à tester
     * @param y_pixel La position y du pixel à tester
     * @param x_pixel La position x du pixel à tester
     */
    bool is_pixel_on_border(int pixel_value, unsigned int y_pixel, unsigned int x_pixel);

    void removeNoise(const unsigned int nbPixels);

    void printRegionMatrix();
    void printRegionMatrixToFile(const std::string filename);
    void printRegionsAdjacency();

    /**
    * Calcule et affiche une image qui montre les frontières des régions préalablement calculées.
    */
    void showRegionBorders(std::string window_name, bool show_initial_seeds);
    void showSegmentation(std::string window_name, bool show_initials_seeds);

    /**
     * Affiche l'image contenue dans m_image
     * 
     * @param window_name Nom de la fenêtre d'affichage
     * @param wait_for_key Si oui ou non attendre pour l'appui d'une touche avant de continuer l'exécution
     */
    void show_img(std::string window_name, bool wait_for_key = false);

    /**
     * Rajoute du texte sur l'image donnée qui montre là où étaient les seeds
     * qui ont segmenté l'image. Le texte sera de la couleur donnée
     */
    void showSeeds(cv::Mat* image, cv::Scalar color = cv::Scalar(0, 0, 0));

protected:
    void init_region_matrix(cv::Mat* image);

    RegionGrowing(OpenCVGrayscaleMat* image);
    RegionGrowing(cv::Mat* image);

    ~RegionGrowing();

    bool m_seeds_placed;
    bool m_regions_computed;

    OpenCVGrayscaleMat* m_image = nullptr;
    cv::Mat* m_image_rgb = nullptr;

    unsigned int m_rows, m_cols;

    int** m_region_matrix;
    size_t m_nb_regions; //Nombre de régions / germes placés

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

class RegionGrowingDifference : public RegionGrowing {
public:
    /*
     * Grayscale constructor
     */
    RegionGrowingDifference(OpenCVGrayscaleMat* image);

    /*
     * RGB constructor
     */
    RegionGrowingDifference(cv::Mat* image);

    /**
     * Lance la segmentation de l'image en niveaux de gris. Voir doc de 'segmentation'
     */
    void segmentationGrayscale(const unsigned int treshold);

    /**
     * Lance la segmentation de l'image en RGB. Voir doc de 'segmentation'
     */
    void segmentationRGB(const unsigned int treshold);

    /**
     * Lance la segmentation de l'image (grossissement des germes).
     * Cette segmentation va utiliser une valeur seuil pour savoir si oui
     * ou non un germe doit grossir sur un pixel adjacent ou non.
     * Si la différence de valeur entre les deux pixels est inférieure ou égale
     * au treshold, le pixel sera pris en compte
     */
    void segmentation(const unsigned int treshold);

    /**
     * Fusione les régions similaires et adjacentes selon le treshold donné
     */
    void regionFusion(const unsigned int treshold);
};

class RegionGrowingAverage : public RegionGrowing {
public:
    /*
     * Grayscale constructor
     */
    RegionGrowingAverage(OpenCVGrayscaleMat* image);

    /*
     * RGB constructor
     */
    RegionGrowingAverage(cv::Mat* image);

    /**
     * Segmentation d'une image en niveau de gris
     */
    void segmentationGrayscale(const float treshold);

    /**
     * Segmentation d'une image en RGB
     */
    void segmentationRGB(const float treshold);

    /**
     * Lacement la segmentation de l'image
     * Cette segmentation va admettre un nouveau pixel dans la région
     * si la valeur du pixel est similaire (au treshold près) à la valeur
     * moyenne des pixels de la région
     *
     * @param treshold Valeur seuil pour l'admission d'un nouveau pixel lors de la
     * comparaison de sa valeur avec la valeur moyenne de la région
     */
    void segmentation(const float treshold);

    /**
     * Fusione les régions similaires et adjacentes selon le treshold donné
     */
    void regionFusion(const float average_threshold);

    /** 
     * Redéfinition de la fonction ici afin de pouvoir utiliser les raccourcis de calculs mis en place
     * dans la fonction 'regionFusion'
     */
    void showSegmentation(const std::string& window_name, const bool show_initials_seeds);

protected:
    /**
     * Va stocker la valeur moyenne des pixels des régions calculées pendant la segmentation de l'image
     * grayscale
     */
    std::vector<float> m_regions_averages;

    /*
     * Idem pour le RGB
     */
    std::vector<cv::Vec3f> m_regions_averages_rgb;

    /**
     * Nouveaux indices des régions après fusion: certaines régions vont prendre le même indice de région
     * qu'une autre région puisqu'elles ont été fusionnées
     * 
     * Par exemple, le nouvel indice de la région "d'indice de base" 2 se trouve en m_new_regions_indexes.at(2).
     */
    std::vector<unsigned int> m_new_regions_indexes;
};
