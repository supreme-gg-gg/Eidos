#include "../include/generic_data_loader.h"
#include "../include/console.hpp"
#include "../include/csvparser.h"
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <random>
#include <numeric>
#include <map>

void GenericDataLoader::load_data(const std::string& filePath, std::vector<Eigen::MatrixXf>& features, std::vector<std::string>& labels) {
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

void GenericDataLoader::split_data(const std::vector<Eigen::MatrixXf>& features, const std::vector<std::string>& labels, std::vector<Eigen::MatrixXf>& train_features, std::vector<std::string>& train_labels, std::vector<Eigen::MatrixXf>& test_features, std::vector<std::string>& test_labels, float trainToTestSplitRatio) {
    if (features.size() != labels.size()) {
        throw std::runtime_error("Number of features and labels do not match.");
    }

    size_t numTrainSamples = static_cast<size_t>(features.size() * trainToTestSplitRatio);
    size_t numTestSamples = features.size() - numTrainSamples;

    train_features.clear();
    train_labels.clear();
    test_features.clear();
    test_labels.clear();

    std::vector<size_t> indices(features.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    for (size_t i = 0; i < features.size(); ++i) {
        if (i < numTrainSamples) {
            train_features.push_back(features[indices[i]]);
            train_labels.push_back(labels[indices[i]]);
        } else {
            test_features.push_back(features[indices[i]]);
            test_labels.push_back(labels[indices[i]]);
        }
    }
}

void GenericDataLoader::convert_to_one_hot(const std::vector<std::string>& labels, std::vector<Eigen::MatrixXf>& one_hot_labels, const std::map<std::string, int>& mapping) {
    one_hot_labels.clear();
    for (const std::string& label : labels) {
        if (mapping.find(label) == mapping.end()) {
            throw std::runtime_error("Label not found in mapping.");
        }
        Eigen::MatrixXf oneHotVector = Eigen::MatrixXf::Zero(1, mapping.size());
        oneHotVector(0, mapping.at(label)) = 1.0f;
        one_hot_labels.push_back(oneHotVector);
    }
}