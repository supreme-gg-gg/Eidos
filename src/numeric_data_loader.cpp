#include "../include/preprocessors/numeric_data_loader.h"
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

NumericDataLoader::NumericDataLoader(const std::string& filePath, 
        const std::string labelsHeaderName, 
        bool shuffle)
{
    CSVParser parser = CSVParser(',');
    std::vector<std::vector<std::string>> data = parser.parse(filePath);
    
    // Extract headers
    std::vector<std::string> headers = data[0];
    int labelIndex = -1;
    // Find the index of the "label" column
    for (size_t i = 0; i < headers.size(); ++i) {
        std::string fieldName = headers[i];
        std::transform(fieldName.begin(), fieldName.end(), fieldName.begin(),[](unsigned char c){ return std::tolower(c); });
        if (fieldName == labelsHeaderName) {
            labelIndex = i;
            break;
        }
    }

    if (labelIndex == -1) {
        throw std::runtime_error("Label column not found in CSV file.");
    }

    // Extract labels and features
    features_.resize(data.size() - 1, headers.size() - 1);
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i].size() != headers.size()) {
            throw std::runtime_error("Invalid data entry in CSV file.");
        }
        if (oneHotMapping_.find(data[i][labelIndex]) == oneHotMapping_.end()) {
            oneHotMapping_[data[i][labelIndex]] = oneHotMapping_.size();
        }
        
        Eigen::VectorXf featureVector(headers.size() - 1);
        int featureIndex = 0;
        for (size_t j = 0; j < data[i].size(); ++j) {
            if (j != labelIndex) {
                featureVector(featureIndex++) = std::stof(data[i][j]);
            }
        }
        features_.row(i-1) = featureVector;
    }
    labels_.resize(data.size() - 1, oneHotMapping_.size());
    for (int i = 1; i < data.size(); ++i) {
        if (oneHotMapping_.find(data[i][labelIndex]) == oneHotMapping_.end()) {
            throw std::runtime_error("Label not found in mapping.");
        }
        Eigen::RowVectorXf oneHotVector = Eigen::RowVectorXf::Zero(oneHotMapping_.size());
        oneHotVector(oneHotMapping_.at(data[i][labelIndex])) = 1.0f;
        labels_.row(i-1) = oneHotVector;
    }
    if (shuffle) {
        this->shuffle();
    }
}

NumericDataLoader::NumericDataLoader(const Eigen::MatrixXf& features, 
        const Eigen::MatrixXf& labels)
    : features_(features), labels_(labels) {}

InputData NumericDataLoader::train_test_split(float trainToTestSplitRatio) {
    if (trainToTestSplitRatio < 0.0f || trainToTestSplitRatio > 1.0f) {
        throw std::invalid_argument("Invalid train-test split ratio. (Expected: 0.0-1.0)");
    }
    size_t numTrainSamples = static_cast<size_t>(features_.rows() * trainToTestSplitRatio);
    size_t numTestSamples = features_.rows() - numTrainSamples;
    
    InputData result;
    result.train_features = features_.topRows(numTrainSamples);
    result.train_labels = labels_.topRows(numTrainSamples);
    result.test_features = features_.bottomRows(numTestSamples);
    result.test_labels = labels_.bottomRows(numTestSamples);
    return result;
}

NumericDataLoader& NumericDataLoader::shuffle() {
    std::vector<size_t> indices(features_.rows());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    Eigen::MatrixXf shuffled_features(features_.rows(), features_.cols());
    Eigen::MatrixXf shuffled_labels(labels_.rows(), labels_.cols());
    for (size_t i = 0; i < indices.size(); ++i) {
        shuffled_features.row(i) = features_.row(indices[i]);
        shuffled_labels.row(i) = labels_.row(indices[i]);
    }
    features_ = shuffled_features;
    labels_ = shuffled_labels;

    return *this;
}

NumericDataLoader& NumericDataLoader::center() {
    features_ = features_.rowwise() - features_.colwise().mean();
    return *this;
}

NumericDataLoader& NumericDataLoader::linear_transform(float a, float b) {
    features_ = features_.array() * a + b;
    return *this;
}

NumericDataLoader& NumericDataLoader::min_max_scale(int min_val, int max_val) {
    Eigen::VectorXf min_vals = features_.colwise().minCoeff();
    Eigen::VectorXf max_vals = features_.colwise().maxCoeff();
    features_ = (features_.rowwise() - min_vals.transpose()).array().rowwise() / (max_vals - min_vals).transpose().array();
    features_ = features_.array() * (max_val - min_val) + min_val;
    return *this;
}

NumericDataLoader& NumericDataLoader::remove_outliers(float z_threshold) {
    // Calculate mean and standard deviation for each feature
    Eigen::VectorXf mean = features_.colwise().mean();
    Eigen::MatrixXf centered = features_.rowwise() - mean.transpose();
    Eigen::VectorXf std_dev = (centered.array().square().colwise().sum() / (features_.rows() - 1)).sqrt();

    // Identify outliers
    Eigen::MatrixXf z_scores = centered.array().rowwise() / std_dev.transpose().array();
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> outliers = z_scores.array().abs() > z_threshold;

    // Handle outliers (e.g., remove rows with outliers)
    std::vector<int> rows_to_keep;
    for (int i = 0; i < outliers.rows(); ++i) {
        if (!outliers.row(i).any()) {
            rows_to_keep.push_back(i);
        }
    }

    // Create new matrices without outliers
    Eigen::MatrixXf filtered_features(rows_to_keep.size(), features_.cols());
    Eigen::MatrixXf filtered_labels(rows_to_keep.size(), labels_.cols());
    for (size_t i = 0; i < rows_to_keep.size(); ++i) {
        filtered_features.row(i) = features_.row(rows_to_keep[i]);
        filtered_labels.row(i) = labels_.row(rows_to_keep[i]);
    }
    features_ = filtered_features;
    labels_ = filtered_labels;

    return *this;
}

NumericDataLoader& NumericDataLoader::z_score_normalize() {
    Eigen::VectorXf mean = features_.colwise().mean();
    Eigen::VectorXf stddev = ((features_.rowwise() - mean.transpose()).array().square().colwise().sum() / (features_.rows() - 1)).sqrt();

    features_ = (features_.rowwise() - mean.transpose()).array().rowwise() / stddev.transpose().array();

    return *this;
}

NumericDataLoader& NumericDataLoader::pca(int target_dim) {
    // Center the data by subtracting the mean
    center();
    
    // Compute the covariance matrix
    Eigen::MatrixXf cov = (features_.adjoint() * features_) / static_cast<float>(features_.rows() - 1);

    // Compute the eigenvalues and eigenvectors
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigen_solver(cov);
    Eigen::MatrixXf eigenvectors = eigen_solver.eigenvectors().rightCols(target_dim);

    // Transform the data to the new basis
    features_ *= eigenvectors;

    return *this;
}

size_t NumericDataLoader::num_samples() const {
    return features_.rows();
}

size_t NumericDataLoader::num_features() const {
    return features_.cols();
}

size_t NumericDataLoader::num_classes() const {
    return oneHotMapping_.size();
}

std::pair<int, int> NumericDataLoader::shape() const {
    return std::make_pair(features_.rows(), features_.cols());
}

void NumericDataLoader::print_preview(int num_samples) const {
    if (num_samples > features_.rows()) {
        num_samples = features_.rows();
    }

    std::string output = "";
    output += "Features:\n";
    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < features_.cols(); ++j) {
            output += std::to_string(features_(i, j))+" ";
        }
        output += "\n";
    }
    
    output += "\nLabels:\n";
    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < labels_.cols(); ++j) {
            output += std::to_string(labels_(i, j))+" ";
        }
        output += "\n";
    }
    output += "\n";
    output += "Number of samples: " + std::to_string(features_.rows()) + "\n";
    output += "Number of features: " + std::to_string(features_.cols()) + "\n";
    output += "Number of categories: " + std::to_string(oneHotMapping_.size()) + "\n";
    Console::log(output);
}