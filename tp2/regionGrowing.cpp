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
#include <fstream>
#include <iostream>
#include <random> //std::default_random_engine

typedef RegionGrowing::Seed Seed;

void RegionGrowing::init_region_matrix(cv::Mat* image) {
    //Création de la matrice de la même taille que l'image pour accueillir les données
    //du région growing
    
    //TODO (Tom) Comparer les perfomrances entre ça et un tableau contigu
    m_region_matrix = new int* [image->rows];
    for (int row = 0; row < image->rows; row++) {
        m_region_matrix[row] = new int[image->cols];

        //Initializing toutes les valeurs à -1
        for (int i = 0; i < image->cols; i++) {
            m_region_matrix[row][i] = -1;
        }
    }
}

RegionGrowing::RegionGrowing(OpenCVGrayscaleMat* image) {
    m_image = image;

    init_region_matrix(m_image);
}

RegionGrowing::RegionGrowing(cv::Mat* image) {
    m_image_rgb = image;

    init_region_matrix(m_image_rgb);
}

RegionGrowing::~RegionGrowing() {
    for (int row = 0; row < m_image->rows; row++)
        delete [] m_region_matrix[row];

    delete [] m_region_matrix;
}

void RegionGrowing::placeSeedsManual(std::vector<std::pair<unsigned int, unsigned int>> positionsList) {
    m_seeds_placed = true;

    m_seeds_positions.clear();
    m_seeds_positions.reserve(positionsList.size());

    m_regions_adjacency.resize(positionsList.size());
    m_nb_regions = positionsList.size();

    for (const std::pair<unsigned int, unsigned int>& position : positionsList) {
        m_seeds_positions.push_back(position);
    }
}

void RegionGrowing::placeSeedsRandom(const unsigned int nb_seeds) {
    m_seeds_placed = true;

    m_seeds_positions.clear();
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
    unsigned int rows = m_image_rgb == nullptr ? m_image->rows : m_image_rgb->rows;
    unsigned int cols = m_image_rgb == nullptr ? m_image->cols : m_image_rgb->cols;

    unsigned int baseCellHeight = rows / nbCellsPerRow;
    unsigned int baseCellWidth = cols / nbCellsPerRow;
    for (int i = 0; i < nbCellsPerRow; i++) {
        cellWidths[i] = baseCellWidth;
        cellHeights[i] = baseCellHeight;
    }

    //On va élargir de 1 pixel toutes les cellules qui en ont besoin pour arriver à la taille de l'image
    //si jamais on avait jusque là des pixels qui n'étaient pas pris en compte
    //(les pixels tout à droite de l'image si on parle de la largeur)
    const unsigned int widthRemainder = cols % cellWidths[0];
    for (int i = 0; i < widthRemainder; i++) {
        cellWidths[i]++;
    }

    //Même chose pour la hauteur
    const unsigned int heightRemainder = rows % cellHeights[0];
    for (int i = 0; i < heightRemainder; i++) {
        cellHeights[i]++;
    }

    std::vector<unsigned int> randomCellIndexes;
    randomCellIndexes.reserve(nbCellsPerRow * nbCellsPerRow);
    for(unsigned int index = 0; index < nbCellsPerRow * nbCellsPerRow; index++)
        randomCellIndexes.push_back(index);

    std::shuffle(randomCellIndexes.begin(), randomCellIndexes.end(), std::default_random_engine(randomSeed));

    for (int i = 0; i < nb_seeds; i++) {
        unsigned int randomCellIndex = randomCellIndexes.at(i);

        //TODO (Tom) remplacer le rand par un générateur plus rapide car std::rand = slow
        unsigned int cellWidth = cellWidths[randomCellIndex % nbCellsPerRow];
        unsigned int cellHeight = cellHeights[randomCellIndex / nbCellsPerRow];

        unsigned int randomX = std::rand() % cellWidth + (randomCellIndex % nbCellsPerRow) * cellWidth;
        unsigned int randomY = std::rand() % cellHeight + (randomCellIndex / nbCellsPerRow) * cellHeight;

        m_seeds_positions.push_back(std::pair<unsigned int, unsigned int>(randomX, randomY));
    }

    //Affichage des positions des seeds
    //for (int i = 0; i < m_seeds_positions.size(); i++) {
    //    std::cout << "positionsSimpleImageGrayscale.push_back(std::pair<unsigned int, unsigned int>(" << m_seeds_positions.at(i).first << ", " << m_seeds_positions.at(i).second << "));" << std::endl;
    //    //std::cout << "seed " << i << ": " << m_seeds_positions.at(i).first << ", " << m_seeds_positions.at(i).second << std::endl;
    //}

    delete[] cellWidths;
    delete[] cellHeights;
}

void compute_gaussian_kernel(double** kernel, unsigned int kernel_size, double sigma) {
    unsigned int half_size = kernel_size / 2;
    double kernel_sum = 0;

    for (int y = 0; y < kernel_size; y++) {
        for (int x = 0; x < kernel_size; x++) {
            int shift_x = x - half_size;
            int shift_y = y - half_size;

            kernel[y][x] = 1.0 / (2 * PI * sigma * sigma) * exp(-((shift_x * shift_x + shift_y * shift_y) / (2 * sigma * sigma)));

            kernel_sum += kernel[y][x];
        }
    }

    //Pour être sûr que la somme des valeurs du noyau = 1
    for (int y = 0; y < kernel_size; y++) {
        for (int x = 0; x < kernel_size; x++) {
            kernel[y][x] /= kernel_sum;
        }
    }
}

void RegionGrowing::blurGrayscale(double** kernel, int kernel_size) {
    int half_kernel_size = kernel_size / 2;

    for (int y_img = 0; y_img < m_image->rows; y_img++) {
        for (int x_img = 0; x_img < m_image->cols; x_img++) {
            unsigned char current_pixel_value = (*m_image)(y_img, x_img);

            double new_pixel_value = 0;
            for (int y_kernel = 0; y_kernel < kernel_size; y_kernel++) {
                for (int x_kernel = 0; x_kernel < kernel_size; x_kernel++) {
                    int y_kernel_shift = y_kernel - half_kernel_size;
                    int x_kernel_shift = x_kernel - half_kernel_size;

                    unsigned int y_pos = y_img + y_kernel_shift;
                    unsigned int x_pos = x_img + x_kernel_shift;

                    unsigned char pixel_value;

                    //Si on est en train de dépasser des bords de l'image
                    if (y_pos < 0 || y_pos >= m_image->rows || x_pos < 0 || x_pos >= m_image->cols) {
                        //On va considérer que la valeur du pixel est la même que celle du pixel courant
                        pixel_value = current_pixel_value;
                    }
                    else {
                        pixel_value = (*m_image)(y_pos, x_pos);
                    }

                    new_pixel_value += pixel_value * kernel[y_kernel][x_kernel];
                }
            }

            (*m_image)(y_img, x_img) = (unsigned char)new_pixel_value;
        }
    }
}

void RegionGrowing::blurRGB(double** kernel, int kernel_size) {
    int half_kernel_size = kernel_size / 2;

    for (int y_img = 0; y_img < m_image_rgb->rows; y_img++) {
        for (int x_img = 0; x_img < m_image_rgb->cols; x_img++) {
            cv::Vec3b current_pixel_value = m_image_rgb->at<cv::Vec3b>(y_img, x_img);

            cv::Vec3f new_pixel_value = 0;
            for (int y_kernel = 0; y_kernel < kernel_size; y_kernel++) {
                for (int x_kernel = 0; x_kernel < kernel_size; x_kernel++) {
                    int y_kernel_shift = y_kernel - half_kernel_size;
                    int x_kernel_shift = x_kernel - half_kernel_size;

                    unsigned int y_pos = y_img + y_kernel_shift;
                    unsigned int x_pos = x_img + x_kernel_shift;

                    cv::Vec3b pixel_value;

                    //Si on est en train de dépasser des bords de l'image
                    if (y_pos < 0 || y_pos >= m_image_rgb->rows || x_pos < 0 || x_pos >= m_image_rgb->cols) {
                        //On va considérer que la valeur du pixel est la même que celle du pixel courant
                        pixel_value = current_pixel_value;
                    }
                    else {
                        pixel_value = m_image_rgb->at<cv::Vec3b>(y_pos, x_pos);
                    }

                    new_pixel_value += pixel_value * kernel[y_kernel][x_kernel];
                }
            }

            m_image_rgb->at<cv::Vec3b>(y_img, x_img) = (cv::Vec3b)new_pixel_value;
        }
    }
}

void RegionGrowing::blur(unsigned int kernel_size, double sigma) {
    double** kernel = (double**)malloc(sizeof(double*) * kernel_size);
    if (kernel == NULL) {
        return;
    }

    for (int i = 0; i < kernel_size; i++) {
        kernel[i] = (double*)malloc(sizeof(double) * kernel_size);

        if (kernel[i] == NULL) {
            return;
        }

    }

    compute_gaussian_kernel(kernel, kernel_size, sigma);

    if (m_image_rgb == nullptr)
        blurGrayscale(kernel, kernel_size);
    else
        blurRGB(kernel, kernel_size);

    for (int i = 0; i < kernel_size; i++) {
        free(kernel[i]);
    }

    free(kernel);
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

bool RegionGrowing::is_pixel_on_border(int pixel_value, unsigned int y_pixel, unsigned int x_pixel) {
    //Si le pixel est en bordure de l'image elle même
    unsigned int rows = m_image_rgb == nullptr ? m_image->rows : m_image_rgb->rows;
    unsigned int cols = m_image_rgb == nullptr ? m_image->cols : m_image_rgb->cols;

    if (y_pixel == 0 || y_pixel == rows - 1 || x_pixel == 0 || x_pixel == cols - 1) {
        return true;
    }

    //Pixel au dessus
    if (y_pixel > 0 && m_region_matrix[y_pixel - 1][x_pixel] != pixel_value) {
        return true;
    } else if (y_pixel < rows - 1 && m_region_matrix[y_pixel + 1][x_pixel] != pixel_value) {//Pixel en dessous
        return true;
    } else if (x_pixel > 0 && m_region_matrix[y_pixel][x_pixel - 1] != pixel_value) {//Pixel à gauche
        return true;
    } else if (x_pixel < cols - 1 && m_region_matrix[y_pixel][x_pixel + 1] != pixel_value) {//Pixel à droite
        return true;
    }

    return false;
}

void RegionGrowing::removeNoise(const unsigned int nbPixels) {
    // On vérifie bien que les regions sont placées avant de commencer
    if (!m_regions_computed) {
        return;
    }

    // On compte les pixels de chaque zones de la matrice en parcourant la matrice des régions
    std::vector<unsigned int> regions_pixels_count(m_regions_adjacency.size(), 0);
    
    unsigned int rows = m_image_rgb == nullptr ? m_image->rows : m_image_rgb->rows;
    unsigned int cols = m_image_rgb == nullptr ? m_image->cols : m_image_rgb->cols;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            int regionIdx = m_region_matrix[y][x];
            if (regionIdx != -1)
                regions_pixels_count[regionIdx]++;
        }
    }

    // Print regions_pixels_count
    //for (int i = 0; i < regions_pixels_count.size(); i++) {
    //    std::cout << "Region " << i << " : " << regions_pixels_count[i] << " pixels" << std::endl;
    //}

    // On remplace la région par un de ses voisins dans la matrice des régions
    for (int regionIdx = 0; regionIdx < regions_pixels_count.size(); regionIdx++) {
        if (regions_pixels_count[regionIdx] < nbPixels) {
            // On remplace la région par un de ses voisins dans la matrice des régions
            int newRegion = *(m_regions_adjacency[regionIdx].begin());

            for (int y = 0; y < rows; y++) {
                for (int x = 0; x < cols; x++) {
                    if (m_region_matrix[y][x] == regionIdx) {
                        m_region_matrix[y][x] = newRegion;
                    }
                }
            }

            // On supprime la région dans la liste d'adjacence
            m_regions_adjacency[regionIdx].clear();
            m_regions_adjacency[regionIdx].insert(-1);
        }
    }
}

void randomRGBColor(int rgb[])
{
    for(int i = 0; i < 3; i++) {
        rgb[i] = rand() % 256;
    }
}

float rgb_distance_L1(const cv::Vec3f& rgb1, const cv::Vec3f& rgb2) {
    return abs(rgb1[0] - rgb2[0]) + abs(rgb1[1] - rgb2[1]) + abs(rgb1[2] - rgb2[2]);
}

void RegionGrowing::showSegmentation(std::string window_name, bool show_initials_seeds) {
    cv::Mat regions_img = cv::Mat::zeros(m_image->rows, m_image->cols, CV_8UC3);

    // Parcourt la matrice des régions et colorie l'image en concéquence
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
            } else {
                int rgb[3];
                randomRGBColor(rgb);
                distinct_colors.push_back({rgb[0], rgb[1], rgb[2]});
                regions_img.at<cv::Vec3b>(i, j)[0] = rgb[0];
                regions_img.at<cv::Vec3b>(i, j)[1] = rgb[1];
                regions_img.at<cv::Vec3b>(i, j)[2] = rgb[2];
            }
        }
    }

    // Affiche les seeds initiaux sous forme de cercle
    if (show_initials_seeds) {
        showSeeds(&regions_img);
    }

    //Resize dans le cas d'une image plus grande que l'écran
    //cv::Mat resized;
    //cv::resize(regions_img, resized, cv::Size(1920, 1080), cv::INTER_LINEAR);

    cv::imshow(window_name, regions_img);
}

void RegionGrowing::show_img(std::string window_name, bool wait_for_key) {
    cv::imshow(window_name, *m_image);

    if (wait_for_key) {
        cv::waitKey(0);
    }
}

void RegionGrowing::showSeeds(cv::Mat* image, cv::Scalar color) {
    if (!m_seeds_placed || !m_regions_computed) {
        return;
    }

    int index = 0;
    for (std::pair<unsigned int, unsigned int> m_seed_position : m_seeds_positions) {
        // Affiche le texte si la région n'a pas été supprimée (Fusion)
        if (m_regions_adjacency[index].find(-1) == m_regions_adjacency[index].end()) {
            cv::Point center(m_seed_position.first, m_seed_position.second);

            int radius = 5;
            int thicknessCircle = 2;

            cv::circle(*image, center, radius, color, -1);

            std::string text = std::to_string(index);
            int fontFace = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = 0.6;
            double thicknessText = 2;
            cv::putText(*image, text, center, fontFace, fontScale, color, thicknessText);
        }
        index++;
    }
}

void RegionGrowing::showRegionBorders(std::string window_name, bool show_initial_seeds) {
    unsigned int rows = m_image_rgb == nullptr ? m_image->rows : m_image_rgb->rows;
    unsigned int cols = m_image_rgb == nullptr ? m_image->cols : m_image_rgb->cols;

    cv::Mat borders_img = cv::Mat::zeros(rows, cols, CV_8UC3);

    //On va parcourir chaque pixel de l'image segmetnée et regarder les voisins de ce pixel
    //si un des voisins du pixel n'a pas la même valeur que lui, cela veut dire que le pixel
    //est en bordure de sa région et on va donc le colorer d'une certaine couleur dans l'image
    //qui montrera les bordures de région
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int pixel_value = m_region_matrix[i][j];

            if (pixel_value == -1) {//Partie de l'image qui n'a pas été "capturée" par les germes
                //on skip ce pixel parce qu'il ne correspond à aucune région et on ne va pas calculer
                //la bordure d'une région qui n'existe pas

                continue;
            } else {
                //On regarde les voisins du pixel pour savoir si on est en bordure de la région
                bool is_on_border = is_pixel_on_border(pixel_value, i, j);

                if (is_on_border) {
                    if (pixel_value < (int)distinct_colors.size()) {
                        borders_img.at<cv::Vec3b>(i, j)[0] = distinct_colors[pixel_value][2];
                        borders_img.at<cv::Vec3b>(i, j)[1] = distinct_colors[pixel_value][1];
                        borders_img.at<cv::Vec3b>(i, j)[2] = distinct_colors[pixel_value][0];
                    } else {
                        int rgb[3];
                        randomRGBColor(rgb);
                        distinct_colors.push_back({ rgb[0], rgb[1], rgb[2] });

                        borders_img.at<cv::Vec3b>(i, j)[0] = rgb[0];
                        borders_img.at<cv::Vec3b>(i, j)[1] = rgb[1];
                        borders_img.at<cv::Vec3b>(i, j)[2] = rgb[2];
                    }
                }
            }
        }
    }

    // Affiche les seeds initiaux sous forme de cercle
    if (show_initial_seeds) {
        showSeeds(&borders_img, cv::Scalar(255, 255, 255));
    }

    cv::imshow(window_name, borders_img);
}

void RegionGrowing::printRegionMatrix() {
    for(int i = 0; i < m_image->rows; i++) {
        std::cout << "[";
        for(int j = 0; j < m_image->cols; j++) {
            int val = m_region_matrix[i][j];
            if (val == -1)
                std::cout << "-, ";
            else
                std::cout << val << ", ";
        }
        std::cout << "]\n";
    }
}

void RegionGrowing::printRegionMatrixToFile(const std::string filename) {
    std::ofstream outputFile;
    outputFile.open(filename);

    for (int i = 0; i < m_image->rows; i++) {
        outputFile << "[";
        for (int j = 0; j < m_image->cols; j++) {
            int val = m_region_matrix[i][j];
            if (val == -1)
                outputFile << "-, ";
            else
                outputFile << val << ", ";
        }
        outputFile << "]\n";
    }

    outputFile.close();
}

void RegionGrowing::printRegionsAdjacency() {
    int index = 0;
    for (auto start = m_regions_adjacency.begin(), end = m_regions_adjacency.end(); start != end; start++) {
        std::cout << "Voisins de la region " << index++ << ": ";
        for (auto startNeighbors = (*start).begin(), endNeighbors = (*start).end(); startNeighbors != endNeighbors; startNeighbors++) {
            if (*startNeighbors == -1)
                std::cout << "This region was deleted";
            else
                std::cout << *startNeighbors << ", ";
        }

        std::cout << std::endl;
    }
}

RegionGrowingDifference::RegionGrowingDifference(OpenCVGrayscaleMat* image) : RegionGrowing(image) {}
RegionGrowingDifference::RegionGrowingDifference(cv::Mat* image) : RegionGrowing(image) {}

void RegionGrowingDifference::segmentation(const unsigned int treshold) {
    //On vérifie bien que les seeds sont placées avant de commencer
    if (!m_seeds_placed) {
        return;
    }

    std::deque<Seed> active_seeds;

    int index = 0;
    for (const std::pair<unsigned int, unsigned int>& initial_seed_position : m_seeds_positions) {
        unsigned int x = initial_seed_position.first;
        unsigned int y = initial_seed_position.second;

        Seed seed(x, y, (*m_image)(y, x), index++);

        active_seeds.push_back(seed);
        m_region_matrix[y][x] = seed.region;
    }

    //On va faire grandir les régions tant qu'il y a des seeds
    while (!active_seeds.empty()) {
        Seed seed = active_seeds.front();
        active_seeds.pop_front();

        unsigned int x = seed.position_x;
        unsigned int y = seed.position_y;

        //Valeur du pixel
        unsigned int value = seed.value;

        //Valeurs des pixels au dessus, à gauche, à droite ou en dessous du germe
        int neighborPixelsValues[4] = {
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

            int neighborPixelValue = neighborPixelsValues[offset];
            int neighborSeedRegion = neighborPixelValue == -1 ? -1 : m_region_matrix[yNeighbor][xNeighbor];

            if (neighborPixelValue != -1 && //On a bien un pixel voisin (on est pas en dehors de l'image)
                std::abs((int)(neighborPixelValue - value)) <= treshold) {//Le pixel satisfait le critère de ressemblance

                if (neighborSeedRegion == -1) { //Le pixel voisin n'a pas encore été visité par un germe
                    Seed new_seed(xNeighbor, yNeighbor, seed.value, seed.region);

                    active_seeds.push_back(new_seed);
                    m_region_matrix[yNeighbor][xNeighbor] = seed.region;
                }
            }

            if (neighborSeedRegion != seed.region && neighborSeedRegion != -1) { //Le pixel voisin est déjà occupé par un germe et ce n'est pas un germe de notre propre région
                //Le pixel a déjà été visité, on va ajouter la valeur du germe voisin
                //à la liste des régions adjacentes de la région actuelle
                m_regions_adjacency[seed.region].insert(neighborSeedRegion);
            }
        }
    }

    normalizeAdjacency();
    m_regions_computed = true;
}

void RegionGrowingDifference::regionFusion(const unsigned int treshold) {
    // On vérifie bien que les regions sont placées avant de commencer
    if (!m_regions_computed) {
        return;
    }

    std::deque<int> active_regions_idx;

    // On ajoute toutes les régions à la file
    for (int regionIndex = 0; regionIndex < m_regions_adjacency.size(); regionIndex++) {
        active_regions_idx.push_back(regionIndex);
    }

    // On va fusionner les régions tant qu'il y a des régions dans la file
    while (!active_regions_idx.empty()) {
        int regionIdx = active_regions_idx.front();
        active_regions_idx.pop_front();

        unsigned int x = m_seeds_positions[regionIdx].first;
        unsigned int y = m_seeds_positions[regionIdx].second;

        //Valeur du pixel de la région
        unsigned int value = (*m_image)(y, x);

        auto it = m_regions_adjacency[regionIdx].begin();
        bool hasBeenMerged = false;
        while (!hasBeenMerged && it != m_regions_adjacency[regionIdx].end()) {
            int neighborRegionIdx = *it;

            int neighborX = m_seeds_positions[neighborRegionIdx].first;
            int neighborY = m_seeds_positions[neighborRegionIdx].second;

            unsigned char neighborValue = (*m_image)(neighborY, neighborX);

            if (std::abs((int)(neighborValue - value)) <= treshold) {
                // On fusionne les régions dans la liste d'adjacence
                m_regions_adjacency[regionIdx].insert(m_regions_adjacency[neighborRegionIdx].begin(), m_regions_adjacency[neighborRegionIdx].end());
                m_regions_adjacency[regionIdx].erase(neighborRegionIdx);
                m_regions_adjacency[regionIdx].erase(regionIdx);

                // On fusionne les régions dans la matrice
                for (int y = 0; y < m_image->rows; y++) {
                    for (int x = 0; x < m_image->cols; x++) {
                        if (m_region_matrix[y][x] == neighborRegionIdx) {
                            m_region_matrix[y][x] = regionIdx;
                        }
                    }
                }

                // On "supprime" la région fusionnée dans la liste d'adjacence en changeant ses voisins par -1
                m_regions_adjacency[neighborRegionIdx].clear();
                m_regions_adjacency[neighborRegionIdx].insert(-1);

                // On ajoute la "nouvelle" région à la file
                active_regions_idx.push_back(regionIdx);

                // On supprime la région fusionnée de la file
                active_regions_idx.erase(
                    std::remove(active_regions_idx.begin(), active_regions_idx.end(), neighborRegionIdx),
                    active_regions_idx.end()
                );

                // On remplace la région fusionnée par la nouvelle région dans chaque liste d'adjacence
                for (int regionIndex = 0; regionIndex < m_regions_adjacency.size(); regionIndex++) {
                    // Si la région d'adjacence n'a pas été supprimé (-1)
                    // et que la région d'adjacence contient la région à supprimer
                    if (m_regions_adjacency[regionIndex].find(-1) == m_regions_adjacency[regionIndex].end()
                        && m_regions_adjacency[regionIndex].find(neighborRegionIdx) != m_regions_adjacency[regionIndex].end()) {
                        m_regions_adjacency[regionIndex].erase(neighborRegionIdx);
                        m_regions_adjacency[regionIndex].insert(regionIdx);
                    }
                }

                hasBeenMerged = true;
            }
            it++;
        }
    }
}

RegionGrowingAverage::RegionGrowingAverage(OpenCVGrayscaleMat* image) : RegionGrowing(image) {}
RegionGrowingAverage::RegionGrowingAverage(cv::Mat* image) : RegionGrowing(image) {}

void RegionGrowingAverage::segmentationGrayscale(const float treshold) {
    std::vector<unsigned int> regions_sums;//Va contenir la somme des valeurs des pixels des régions. Utile pour calculer la moyenne
    std::vector<unsigned int> regions_pixel_count;//Contient le nombre de pixels de chaque région pour calculer la moyenne

    regions_sums.resize(m_seeds_positions.size());
    regions_pixel_count.resize(m_seeds_positions.size());
    m_regions_averages.resize(m_seeds_positions.size());

    std::deque<Seed> active_seeds;

    int index = 0;
    for (const std::pair<unsigned int, unsigned int>& initial_seed_position : m_seeds_positions) {
        unsigned int x = initial_seed_position.first;
        unsigned int y = initial_seed_position.second;

        Seed seed(x, y, (*m_image)(y, x), index);

        active_seeds.push_back(seed);

        //Initialisation de la région matrix et des compteurs des régions
        m_region_matrix[y][x] = seed.region;
        regions_sums.at(index) = seed.value;
        regions_pixel_count.at(index) = 1;
        m_regions_averages.at(index) = seed.value;

        index++;
    }

    //On va faire grandir les régions tant qu'il y a des seeds
    while (!active_seeds.empty()) {
        Seed seed = active_seeds.front();
        active_seeds.pop_front();

        unsigned int x = seed.position_x;
        unsigned int y = seed.position_y;

        //Valeur du pixel courant
        unsigned int current_region = seed.region;

        //Valeurs des pixels au dessus, à gauche, à droite ou en dessous du germe
        int neighborPixelsValues[4] = {
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

            float currentRegionAverage = m_regions_averages.at(current_region);
            int neighborPixelValue = neighborPixelsValues[offset];
            int neighborSeedRegion = neighborPixelValue == -1 ? -1 : m_region_matrix[yNeighbor][xNeighbor];

            if (neighborPixelValue != -1 && //On a bien un pixel voisin (on est pas en dehors de l'image)
                std::abs(neighborPixelValue - currentRegionAverage) <= treshold) {//Le pixel satisfait le critère de ressemblance

                if (neighborSeedRegion == -1) { //Le pixel voisin n'a pas encore été visité par un germe
                    Seed new_seed(xNeighbor, yNeighbor, neighborPixelValue, seed.region);

                    active_seeds.push_back(new_seed);
                    m_region_matrix[yNeighbor][xNeighbor] = seed.region;

                    //Mise à jour de la moyenne
                    regions_pixel_count.at(current_region)++;
                    regions_sums.at(current_region) += neighborPixelValue;
                    m_regions_averages.at(current_region) = (float)regions_sums.at(current_region) / (float)regions_pixel_count.at(current_region);
                }
            }

            if (neighborSeedRegion != seed.region && neighborSeedRegion != -1) { //Le pixel voisin est déjà occupé par un germe et ce n'est pas un germe de notre propre région
                //Le pixel a déjà été visité, on va ajouter la valeur du germe voisin
                //à la liste des régions adjacentes de la région actuelle
                m_regions_adjacency[seed.region].insert(neighborSeedRegion);
            }
        }
    }

    normalizeAdjacency();

    //Initialisation des regions indexes
    m_new_regions_indexes.clear();
    m_new_regions_indexes.reserve(m_regions_adjacency.size());
    for (int i = 0; i < m_regions_adjacency.size(); i++) {
        m_new_regions_indexes.push_back(i);
    }
    m_regions_computed = true;
}

void RegionGrowingAverage::segmentationRGB(const float treshold) {
    std::vector<cv::Vec3i> regions_sums;//Va contenir la somme des valeurs des pixels des régions. Utile pour calculer la moyenne
    std::vector<unsigned int> regions_pixel_count;//Contient le nombre de pixels de chaque région pour calculer la moyenne

    regions_sums.resize(m_seeds_positions.size());
    regions_pixel_count.resize(m_seeds_positions.size());
    m_regions_averages_rgb.resize(m_seeds_positions.size());

    const cv::Vec3i NO_NEIGHBOR(-1, -1, -1);
    std::deque<SeedRGB> active_seeds;

    int index = 0;
    for (const std::pair<unsigned int, unsigned int>& initial_seed_position : m_seeds_positions) {
        unsigned int x = initial_seed_position.first;
        unsigned int y = initial_seed_position.second;

        SeedRGB seed(x, y, m_image_rgb->at<cv::Vec3b>(y, x), index);

        active_seeds.push_back(seed);

        //Initialisation de la région matrix et des compteurs des régions
        m_region_matrix[y][x] = seed.region;
        regions_sums.at(index) = seed.value;
        regions_pixel_count.at(index) = 1;
        m_regions_averages_rgb.at(index) = seed.value;

        index++;
    }

    //On va faire grandir les régions tant qu'il y a des seeds
    while (!active_seeds.empty()) {
        SeedRGB seed = active_seeds.front();
        active_seeds.pop_front();

        unsigned int x = seed.position_x;
        unsigned int y = seed.position_y;

        //Valeur du pixel courant
        unsigned int current_region = seed.region;

        //Valeurs des pixels au dessus, à gauche, à droite ou en dessous du germe
        cv::Vec3i neighborPixelsValues[4] = {
            (y >= 1) ? (cv::Vec3i)m_image_rgb->at<cv::Vec3b>(y - 1, x) : NO_NEIGHBOR,
            (x >= 1) ? (cv::Vec3i)m_image_rgb->at<cv::Vec3b>(y, x - 1) : NO_NEIGHBOR,
            (y < m_image_rgb->rows - 1) ? (cv::Vec3i)m_image_rgb->at<cv::Vec3b>(y + 1, x) : NO_NEIGHBOR,
            (x < m_image_rgb->cols - 1) ? (cv::Vec3i)m_image_rgb->at<cv::Vec3b>(y, x + 1) : NO_NEIGHBOR,
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

            cv::Vec3f currentRegionAverage = m_regions_averages_rgb.at(current_region);
            cv::Vec3i neighborPixelValue = neighborPixelsValues[offset];
            int neighborSeedRegion = neighborPixelValue == NO_NEIGHBOR ? -1 : m_region_matrix[yNeighbor][xNeighbor];

            if (neighborPixelValue != NO_NEIGHBOR && //On a bien un pixel voisin (on est pas en dehors de l'image)
                rgb_distance_L1(neighborPixelValue, currentRegionAverage) <= treshold) {//Le pixel satisfait le critère de ressemblance

                if (neighborSeedRegion == -1) { //Le pixel voisin n'a pas encore été visité par un germe
                    SeedRGB new_seed(xNeighbor, yNeighbor, neighborPixelValue, seed.region);

                    active_seeds.push_back(new_seed);
                    m_region_matrix[yNeighbor][xNeighbor] = seed.region;

                    //Mise à jour de la moyenne
                    regions_pixel_count.at(current_region)++;
                    regions_sums.at(current_region) += neighborPixelValue;
                    m_regions_averages_rgb.at(current_region) = (cv::Vec3f)regions_sums.at(current_region) / (float)regions_pixel_count.at(current_region);

                    //std::cout << "average: " << (cv::Vec3f)regions_sums.at(current_region) << "/" << (float)regions_pixel_count.at(current_region) << "=" << m_regions_averages_rgb.at(current_region) << std::endl; 
                }
            }

            if (neighborSeedRegion != seed.region && neighborSeedRegion != -1) { //Le pixel voisin est déjà occupé par un germe et ce n'est pas un germe de notre propre région
                //Le pixel a déjà été visité, on va ajouter la valeur du germe voisin
                //à la liste des régions adjacentes de la région actuelle
                m_regions_adjacency[seed.region].insert(neighborSeedRegion);
            }
        }
    }

    normalizeAdjacency();

    //Initialisation des regions indexes
    m_new_regions_indexes.clear();
    m_new_regions_indexes.reserve(m_regions_adjacency.size());
    for (int i = 0; i < m_regions_adjacency.size(); i++) {
        m_new_regions_indexes.push_back(i);
    }
    m_regions_computed = true;
}

void RegionGrowingAverage::segmentation(const float treshold) {
    //On vérifie bien que les seeds sont placées avant de commencer
    if (!m_seeds_placed) {
        return;
    }

    if (m_image != nullptr) {//Grayscale
        segmentationGrayscale(treshold);
    }
    else {
        segmentationRGB(treshold);
    }
}

bool compareRegionsGrayscale(const float average_region1, const float average_region2, const float treshold) {
    return abs(average_region1 - average_region2) <= treshold;
}

bool compareRegionsRGB(const cv::Vec3f& average_region1, const cv::Vec3f& average_region2, const float treshold) {
    return rgb_distance_L1(average_region1, average_region2) <= treshold;
}

void RegionGrowingAverage::regionFusion(const float treshold) {
    // On vérifie bien que les regions sont placées avant de commencer
    if (!m_regions_computed) {
        return;
    }

    m_new_regions_indexes.clear();
    m_new_regions_indexes.reserve(m_regions_adjacency.size());
    for (int i = 0; i < m_regions_adjacency.size(); i++) {
        m_new_regions_indexes.push_back(i);
    }

    //Pour chaque région
    for (int region_index = 0; region_index < m_regions_adjacency.size(); region_index++) {
        //On parcourt les régions adjacentes
        for (int region_adja_index : m_regions_adjacency.at(region_index)) {
            if (region_adja_index < region_index) {
                //Pour ne pas re-merger deux qui auraient déjà été mergées entres elles
                continue;
            }

            unsigned int adja_new_index = m_new_regions_indexes.at(region_adja_index);
            unsigned int current_region_new_index = m_new_regions_indexes.at(region_index);

            bool similar_regions = false;
            if (m_image_rgb == nullptr) {
                similar_regions = compareRegionsGrayscale(m_regions_averages.at(current_region_new_index), m_regions_averages.at(adja_new_index), treshold);
            }
            else {
                similar_regions = compareRegionsRGB(m_regions_averages_rgb.at(current_region_new_index), m_regions_averages_rgb.at(adja_new_index), treshold);
            }

            //Si la région voisine satisfait le critère de ressemblance avec la région actuelle
            if (similar_regions) {
                //Si on essaye de merge avec une région qui a déjà été mergée
                if (adja_new_index == current_region_new_index) {
                    //Ces deux régions sont déjà mergées ensemble
                    continue;
                }
                else if (adja_new_index != region_adja_index) {
                    unsigned int new_index = current_region_new_index;

                    for (int idx = 0; idx < m_new_regions_indexes.size(); idx++) {
                        if (m_new_regions_indexes.at(idx) == adja_new_index) {
                            m_new_regions_indexes.at(idx) = new_index;
                        }
                    }
                }

                //On renumérote la région voisine pour qu'elle soit considérée comme la même région que la région actuelle
                m_new_regions_indexes.at(region_adja_index) = current_region_new_index;
            }
        }
    }
}

void RegionGrowingAverage::showSegmentation(const std::string& window_name, const bool show_initials_seeds) {
    cv::Mat regions_img;

    if (m_image_rgb == nullptr) {
        regions_img = cv::Mat::zeros(m_image->rows, m_image->cols, CV_8UC3);
    }
    else {
        regions_img = cv::Mat::zeros(m_image_rgb->rows, m_image_rgb->cols, CV_8UC3);
    }

    // Parcourt la matrice des régions et colorie l'image en concéquence
    for (int i = 0; i < regions_img.rows; i++) {
        for (int j = 0; j < regions_img.cols; j++) {
            int region_matrix_val = m_region_matrix[i][j];
            if (region_matrix_val == -1) {//Partie de l'image qui n'a pas été "capturée" par les germes 
                //Couleur noire
                regions_img.at<cv::Vec3b>(i, j)[0] = 0;
                regions_img.at<cv::Vec3b>(i, j)[1] = 0;
                regions_img.at<cv::Vec3b>(i, j)[2] = 0;

                continue;
            }

            //std::cout << "region matrix: " << m_region_matrix[i][j] << " " << m_new_regions_indexes.size() << std::endl;
            int val = m_new_regions_indexes.at(m_region_matrix[i][j]);
            if (val < (int)distinct_colors.size()) {
                regions_img.at<cv::Vec3b>(i, j)[0] = distinct_colors[val][2];
                regions_img.at<cv::Vec3b>(i, j)[1] = distinct_colors[val][1];
                regions_img.at<cv::Vec3b>(i, j)[2] = distinct_colors[val][0];
            }
            else {
                int rgb[3];
                randomRGBColor(rgb);
                distinct_colors.push_back({ rgb[0], rgb[1], rgb[2] });
                regions_img.at<cv::Vec3b>(i, j)[0] = rgb[0];
                regions_img.at<cv::Vec3b>(i, j)[1] = rgb[1];
                regions_img.at<cv::Vec3b>(i, j)[2] = rgb[2];
            }
        }
    }

    // Affiche les seeds initiaux sous forme de cercle
    if (show_initials_seeds) {
        showSeeds(&regions_img);
    }

    //Resize dans le cas d'une image plus grande que l'écran
    //cv::Mat resized;
    //cv::resize(regions_img, resized, cv::Size(1920, 1080), cv::INTER_LINEAR);

    cv::imshow(window_name, regions_img);
}
