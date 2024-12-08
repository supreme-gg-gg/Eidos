#ifndef GENERIC_DATA_LOADER_H
#define GENERIC_DATA_LOADER_H

#include "data_loader.h"
#include <string>
#include <vector>
#include <Eigen/Dense>

class GenericDataLoader : public DataLoader {
public:
    /**
     * @brief Loads data from a specified file.
     * 
     * This function loads data from the given file path and stores the data in the provided vectors for features and labels.
     * 
     * @param filePath The path to the file from which to load data.
     * @param features A vector to store the loaded feature matrices.
     * @param labels A vector to store the corresponding labels.
     */
    void load_data(const std::string& filePath, std::vector<Eigen::MatrixXf>& features, std::vector<std::string>& labels) override;
};

#endif // GENERIC_DATA_LOADER_H