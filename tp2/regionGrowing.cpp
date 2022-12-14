/*
* Pour la segmentation en couleurs, une fois qu'on a segmenté les 3 canaux, on peut:
    - Choisir le canal qui est le plus parlant comme étant notre segmentation finale
    - Faire un vote parmis les différents canaux: si 2 canaux disent oui alors ça 
        sera plutot oui que non
*/

#include "regionGrowing.hpp"
#include <vector>

#include <algorithm>
#include <cmath>
#include <ctime>
#include <deque>
#include <iostream>
#include <random> //std::default_random_engine

typedef RegionGrowing::Seed Seed;

RegionGrowing::RegionGrowing(OpenCVGrayscaleMat* image) {
    std::cout << "Image [" << image->cols << ", " << image->rows << "]\n";

    m_image = image;

    //Création de la matrice de la même taille que l'image pour accueillir les données
    //du région growing

    //TODO (Tom) Comparer les perfomrances entre ça et un tableau contigu
    m_region_matrix = new int*[image->rows];
    for (int row = 0; row < image->rows; row++) {
        m_region_matrix[row] = new int[image->cols];

        //Initializing toutes les valeurs à -1
        for(int i = 0; i < image->cols; i++) {
            m_region_matrix[row][i] = -1;
        }
    }
}

RegionGrowing::~RegionGrowing() {
    for (int row = 0; row < m_image->rows; row++)
        delete [] m_region_matrix[row];

    delete [] m_region_matrix;
}

void RegionGrowing::placeSeedsManual(std::vector<std::pair<unsigned int, unsigned int>> positionsList) {
    m_seeds_placed = true;
    m_seeds_positions.reserve(positionsList.size());
    m_regions_adjacency.resize(positionsList.size());
    m_nb_regions = positionsList.size();

    unsigned int index = 0;

    for (const std::pair<unsigned int, unsigned int>& position : positionsList) {
        m_region_matrix[position.second][position.first] = index++;

        m_seeds_positions.push_back(position);
    }
}

void RegionGrowing::placeSeedsRandom(const unsigned int nb_seeds) {
    m_seeds_placed = true;
    m_seeds_positions.reserve(nb_seeds);
    m_regions_adjacency.resize(nb_seeds);
    m_nb_regions = nb_seeds;

    //Génération d'une seed pour les prng qui vont être utilisés après
    unsigned int randomSeed = std::chrono::system_clock::now().time_since_epoch().count();
    std::srand(randomSeed);

    //On va diviser l'image en nbCellsPerRow * nbCellsPerRow 'regions' qui
    //vont accueillir les germes
    unsigned int nbCellsPerRow = (unsigned int)std::ceil(std::sqrt(nb_seeds));

    //Ces tableaux vont contenir les largeurs et les hauteurs des cellules disponibles
    //On a besoin de tableaux comme ça et non pas juste deux variables cellWidth et cellHeight
    //car les cellules peuvent ne psa être toutes de la même taille dans le cas ou
    //la taille de l'image n'est pas divisible par nbCellsPerRow
    unsigned int* cellWidths = new unsigned int[nbCellsPerRow];
    unsigned int* cellHeights = new unsigned int[nbCellsPerRow];

    //On commence par remplir grossièrement le tableau et on ajustera après
    //pour tenir compte des tailles de cellules différentes
    unsigned int baseCellHeight = m_image->rows / nbCellsPerRow;
    unsigned int baseCellWidth = m_image->cols / nbCellsPerRow;
    for (int i = 0; i < nbCellsPerRow; i++) {
        cellWidths[i] = baseCellWidth;
        cellHeights[i] = baseCellHeight;
    }

    //On va élargir de 1 pixel toutes les cellules qui en ont besoin pour arriver à la taille de l'image
    //si jamais on avait jusque là des pixels qui n'étaient pas pris en compte
    //(les pixels tout à droite de l'image si on parle de la largeur)
    const unsigned int widthRemainder = m_image->cols % cellWidths[0];
    for (int i = 0; i < widthRemainder; i++) {
        cellWidths[i]++;
    }

    //Même chose pour la hauteur
    const unsigned int heightRemainder = m_image->rows % cellHeights[0];
    for (int i = 0; i < heightRemainder; i++) {
        cellHeights[i]++;
    }

    std::vector<unsigned int> randomCellIndexes;
    randomCellIndexes.reserve(nbCellsPerRow * nbCellsPerRow);
    for(unsigned int index = 0; index < nbCellsPerRow * nbCellsPerRow; index++)
        randomCellIndexes.push_back(index);

    std::shuffle(randomCellIndexes.begin(), randomCellIndexes.end(), std::default_random_engine(randomSeed));

    for (unsigned int i = 0; i < nb_seeds; i++) {
        unsigned int randomCellIndex = randomCellIndexes.at(i);

        //TODO (Tom) remplacer le rand par un générateur plus rapide car std::rand = slow
        unsigned int cellWidth = cellWidths[randomCellIndex % nbCellsPerRow];
        unsigned int cellHeight = cellHeights[randomCellIndex / nbCellsPerRow];

        unsigned int randomX = std::rand() % cellWidth + (randomCellIndex % nbCellsPerRow) * cellWidth;
        unsigned int randomY = std::rand() % cellHeight + (randomCellIndex / nbCellsPerRow) * cellHeight;

        //On va positioner le germe à la position aléatoire calculée.
        //Les germes vont avoir pour valeur 0, 1, i-1. Cela correspondra
        //aux valeurs des pixels des régions plus tard
        m_region_matrix[randomY][randomX] = i;

        m_seeds_positions.push_back(std::pair<unsigned int, unsigned int>(randomX, randomY));
    }

    delete[] cellWidths;
    delete[] cellHeights;
}

void RegionGrowing::segmentationDifference(const unsigned int treshold) {
    //On vérifie bien que les seeds sont placées avant de commencer
    if(!m_seeds_placed) {
        return;
    }

    std::deque<Seed> active_seeds;

    unsigned int index = 0;
    for (const std::pair<unsigned int, unsigned int>& initial_seed_position : m_seeds_positions) {
        unsigned int x = initial_seed_position.first;
        unsigned int y = initial_seed_position.second;

        Seed seed(x, y, index);

        active_seeds.push_back(seed);
    }

    //On va faire grandir les régions tant qu'il y a des 
    while (!active_seeds.empty()) {
        Seed seed = active_seeds.front();
        active_seeds.pop_front();

        unsigned int x = seed.position_x;
        unsigned int y = seed.position_y;

        //Valeur du pixel de référence
        unsigned int value = (*m_image)(seed.position_y, seed.position_x);

        //Valeurs des pixels au dessus, à gauche, à droite ou en dessous du germe
        int neighborPixels[4] = {
            (y >= 1) ? (*m_image)(y - 1, x) : -1,
            (x >= 1) ? (*m_image)(y, x - 1) : -1,
            (y < m_image->rows - 1) ? (*m_image)(y + 1, x) : -1,
            (x < m_image->cols - 1) ? (*m_image)(y, x + 1) : -1
        };

        //Offsets utilisés pour factoriser le code
        //Ces offsets correspondent aux décalages que l'on doit appliquer
        //au pixel de référence pour obtenir, dans l'ordre:
        //top, left, bottom, right
        int xOffsets[4] = { 0, -1, 0, 1 };
        int yOffsets[4] = { -1, 0, 1, 0 };

        for (int offset = 0; offset < 4; offset++) {
            int xOffset = xOffsets[offset];
            int yOffset = yOffsets[offset];

            int xNeighbor = x + xOffset;
            int yNeighbor = y + yOffset;

            int neighborPixelValue = neighborPixels[offset];
            int neighborSeedValue = neighborPixelValue == -1 ? -1 : m_region_matrix[yNeighbor][xNeighbor];

            if (neighborPixelValue != -1 && //On a bien un pixel au dessus
                std::abs((int)(neighborPixelValue - value)) <= treshold) {//Le pixel satisfait le critère de resssemblance

                if (neighborSeedValue == -1) { //Le pixel voisin n'a pas encore été visité par un germe
                    Seed new_seed(xNeighbor, yNeighbor, m_region_matrix[y][x]);

                    active_seeds.push_back(new_seed);
                    m_region_matrix[yNeighbor][xNeighbor] = m_region_matrix[y][x];
                }
            }

            if (neighborSeedValue != seed.value && neighborSeedValue != -1) { //Le pixel voisin est déjà occupé par un germe et ce n'est pas un germe de notre propre région
                //Le pixel a déjà été visité, on va ajouter la valeur du germe voisin
                //à la liste des régions adjacentes de la région actuelle
                m_regions_adjacency[seed.value].insert(neighborSeedValue);
            }
        }
    }

    normalizeAdjacency();
}

void RegionGrowing::normalizeAdjacency() {
    for (int regionIndex = 0; regionIndex < m_regions_adjacency.size(); regionIndex++) {
        for (int neighborRegion : m_regions_adjacency[regionIndex]) {
            m_regions_adjacency[neighborRegion].insert(regionIndex);
        }
    }
}

bool RegionGrowing::isRegionAdjacent(int regionAValue, int regionBValue) {
    return m_regions_adjacency[regionAValue].find(regionBValue) != m_regions_adjacency[regionAValue].end();
}

void RegionGrowing::regionFusion() {

}

void randomRGBColor(int rgb[])
{
    for(int i = 0; i < 3; i++) {
        rgb[i] = rand() % 256;
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

    printRegionsAdjacency();

    for (int i = 0; i < regions_img.rows; i++) {
        for (int j = 0; j < regions_img.cols; j++) {
            int val = m_region_matrix[i][j];
            if (val == -1) {//Partie de l'image qui n'a pas été "capturée" par les germes 
                //Couleur noire
                regions_img.at<cv::Vec3b>(i, j)[0] = 0;
                regions_img.at<cv::Vec3b>(i, j)[1] = 0;
                regions_img.at<cv::Vec3b>(i, j)[2] = 0;
            } else if (val < (int)distinct_colors.size()) {
                regions_img.at<cv::Vec3b>(i, j)[0] = distinct_colors[val][2];
                regions_img.at<cv::Vec3b>(i, j)[1] = distinct_colors[val][1];
                regions_img.at<cv::Vec3b>(i, j)[2] = distinct_colors[val][0];
            }  else {
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
    cv::waitKey(0);
}

void RegionGrowing::printRegionMatrix() {
    for(int i = 0; i < m_image->rows; i++) {
        std::cout << "[";
        for(int j = 0; j < m_image->cols; j++) {
            std::cout << m_region_matrix[i][j] << ", ";
        }
        std::cout << "]\n";
    }
}

void RegionGrowing::printRegionsAdjacency() {
    int index = 0;
    for (auto start = m_regions_adjacency.begin(), end = m_regions_adjacency.end(); start != end; start++) {
        std::cout << "Voisins de la region " << index++ << ": ";
        for (auto startNeighbors = (*start).begin(), endNeighbors = (*start).end(); startNeighbors != endNeighbors; startNeighbors++) {
            std::cout << *startNeighbors << ", ";
        }

        std::cout << std::endl;
    }
}
