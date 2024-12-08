#include "../include/generic_data_loader.h"
#include "../include/console.hpp"
#include "../include/csvparser.h"
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cctype>

void ImageLoader::load_data(const std::string& filePath, std::vector<Eigen::MatrixXf>& features, std::vector<std::string>& labels) {
    CSVParser parser = CSVParser(',');
    std::vector<std::vector<std::string>> data = parser.parse(filePath);
    
    // Extract headers
    std::vector<std::string> headers = data[0];
    int labelIndex = -1;
    // Find the index of the "label" column
    for (size_t i = 0; i < headers.size(); ++i) {
        std::string fieldName = headers[i];
        std::transform(fieldName.begin(), fieldName.end(), fieldName.begin(),[](unsigned char c){ return std::tolower(c); });
        if (fieldName == "label" || fieldName == "labels") {
            labelIndex = i;
            break;
        }
    }

    if (labelIndex == -1) {
        throw std::runtime_error("Label column not found in CSV file.");
    }

    // Extract labels and features
    labels.clear();
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i].size() != headers.size()) {
            throw std::runtime_error("Invalid data entry in CSV file.");
        }
        labels.push_back(data[i][labelIndex]);

        Eigen::MatrixXf featureVector(1, headers.size() - 1);
        int featureIndex = 0;
        for (size_t j = 0; j < data[i].size(); ++j) {
            if (j != labelIndex) {
                featureVector(0, featureIndex++) = std::stof(data[i][j]);
            }
        }
        features.push_back(featureVector);
    }
}

void ImageLoader::preprocess_data(std::vector<Eigen::MatrixXf>& features, std::vector<std::string>& labels) {
    // Implement preprocessing steps here
}