/*
* Pour la segmentation en couleurs, une fois qu'on a segmenté les 3 canaux, on peut:
    - Choisir le canal qui est le plus parlant comme étant notre segmentation finale
    - Faire un vote parmis les différents canaux: si 2 canaux disent oui alors ça 
        sera plutot oui que non
*/

#include "regionGrowing.hpp"

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

    unsigned int index = 0;

    for (const std::pair<unsigned int, unsigned int>& position : positionsList) {
        m_region_matrix[position.second][position.first] = index++;

        m_seeds_positions.push_back(position);
    }

    printRegionMatrix();
}

void RegionGrowing::placeSeedsRandom(unsigned int nb_seeds) {
    m_seeds_placed = true;
    m_seeds_positions.reserve(nb_seeds);

    //Génération d'une seed pour les prng qui vont être utilisés après
    unsigned int randomSeed = std::chrono::system_clock::now().time_since_epoch().count();
    std::srand(randomSeed);

    //On va diviser l'image en nbCellsPerRow * nbCellsPerRow 'regions' qui
    //vont accueillir les germes
    unsigned int nbCellsPerRow = (unsigned int)std::ceil(std::sqrt(nb_seeds));
    unsigned int cellHeight = m_image->rows / (double)nbCellsPerRow;
    unsigned int cellWidth = m_image->cols / (double)nbCellsPerRow;

    std::vector<unsigned int> randomCellIndexes;
    randomCellIndexes.reserve(nbCellsPerRow * nbCellsPerRow);
    for(unsigned int index = 0; index < nbCellsPerRow * nbCellsPerRow; index++)
        randomCellIndexes.push_back(index);

    std::shuffle(randomCellIndexes.begin(), randomCellIndexes.end(), std::default_random_engine(randomSeed));

    //Affichage du vecteur des index random
    //for (unsigned int i = 0; i < nbCellsPerRow * nbCellsPerRow; i++)
        //std::cout << randomCellIndexes.at(i) << "\n";

    for (unsigned int i = 0; i < nb_seeds; i++) {
        unsigned int randomCellIndex = randomCellIndexes.at(i);

        //TODO (Tom) remplacer le rand par un générateur plus rapide car std::rand = slow
        unsigned int randomX = std::rand() % cellWidth + (randomCellIndex % nbCellsPerRow) * cellWidth;
        unsigned int randomY = std::rand() % cellHeight + (randomCellIndex / nbCellsPerRow) * cellHeight;

        //On va positioner le germe à la position aléatoire calculée.
        //Les germes vont avoir pour valeur 0, 1, i-1. Cela correspondra
        //aux valeurs des pixels des régions plus tard
        m_region_matrix[randomY][randomX] = i;

        m_seeds_positions.push_back(std::pair<unsigned int, unsigned int>(randomX, randomY));
    }
}

void RegionGrowing::segmentationDifference(unsigned int treshold) {
    //On vérifie bien que les seeds sont placées avant de commencer
    if(!m_seeds_placed) {
        return;
    }

    std::deque<Seed> active_seeds;

    unsigned int index = 0;
    for (const std::pair<unsigned int, unsigned int>& initial_seed_position : m_seeds_positions) {
        Seed seed {
                    initial_seed_position.first,
                    initial_seed_position.second,
                    index++
                };

        active_seeds.push_back(seed);
    }

    for (const Seed& seed : active_seeds) {
        unsigned int x = seed.position_x;
        unsigned int y = seed.position_y;

        //Valeurs des pixels au dessus, à gauche, à droite ou en dessous du germe
        int top, left, bottom, right;

        top = (y >= 1) ? (*m_image)(y - 1, x) : -1;
        bottom = (y < m_image->rows - 1) ? (*m_image)(y + 1, x) : -1;
        left = (x >= 1) ? (*m_image)(y, x - 1) : -1;
        right = (x < m_image->cols - 1) ? (*m_image)(y, x + 1) : -1;

        std::cout << "seed position: " << x << ", " << y << std::endl;
        std::cout << "top, left, bottom, right: " << top << ", " << left << ", " << bottom << ", " << right << std::endl;
    }
}

void RegionGrowing::regionFusion() {

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
