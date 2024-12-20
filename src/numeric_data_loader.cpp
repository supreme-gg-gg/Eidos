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
#include <unordered_map>
#include <unordered_set>

NumericDataLoader::NumericDataLoader(const std::string& filePath, 
        const std::string labelsHeaderName, 
        bool shuffle)
{
    CSVParser parser = CSVParser(',');
    std::vector<std::vector<std::string>> data = parser.parse(filePath);
    if (data.empty() || data[0].empty()) {
        throw std::runtime_error("Empty CSV file: " + filePath);
    };
    
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
    
    const size_t num_samples = data.size() - 1;  // Excluding header row
    const size_t num_features = data[0].size() - 1; // Excluding target column
    
    // Analyze each feature to determine if it's numeric or categorical
    std::vector<bool> is_numeric(num_features+1, true);
    std::vector<std::unordered_set<std::string>> unique_categories(num_features+1);
    
    // For each feature
    for (size_t feat = 0; feat < num_features+1; ++feat) {
        // Check all samples for this feature
        if (feat == labelIndex) {
            continue;
        }
        for (size_t sample = 1; sample < data.size(); ++sample) {
            const std::string& value = data[sample][feat];
            
            // Try to convert to float
            try {
                std::stof(value);
            } catch (...) {
                is_numeric[feat] = false;
            }
            
            // Store unique categories
            if (!is_numeric[feat]) {
                unique_categories[feat].insert(value);
            }
        }
    }
    
    // Calculate total number of features after one-hot encoding
    size_t total_features = 0;
    for (size_t feat = 0; feat < num_features+1; ++feat) {
        if (feat == labelIndex) {
            continue;
        }
        if (is_numeric[feat]) {
            total_features += 1;
        } else {
            total_features += unique_categories[feat].size();
        }
    }
    
    // Prepare the result vector
    features_.reserve(num_samples);
    
    // Process each sample
    for (size_t sample = 1; sample <= num_samples; ++sample) {
        Eigen::MatrixXf sample_vector(1, total_features);
        size_t current_pos = 0;
        
        // Process each feature
        for (size_t feat = 0; feat < num_features+1; ++feat) {
            if (feat == labelIndex) {
                continue;
            }
            if (is_numeric[feat]) {
                // Convert numeric feature
                sample_vector(0, current_pos) = std::stof(data[sample][feat]);
                current_pos += 1;
            } else {
                // One-hot encode categorical feature
                const std::string& category = data[sample][feat];
                size_t category_size = unique_categories[feat].size();
                Eigen::MatrixXf one_hot = Eigen::MatrixXf::Zero(1, category_size);
                
                // Find position of current category
                size_t category_pos = 0;
                for (const auto& cat : unique_categories[feat]) {
                    if (cat == category) {
                        one_hot(0, category_pos) = 1.0f;
                        break;
                    }
                    category_pos++;
                }
                
                sample_vector.block(0, current_pos, 1, category_size) = one_hot;
                current_pos += category_size;
            }
        }
        features_.push_back(sample_vector);
    }
    
    // Preprocess the labels
    for (size_t sample = 1; sample < num_samples; ++sample) {
        if (oneHotMapping_.find(data[sample][labelIndex]) == oneHotMapping_.end()) {
            oneHotMapping_[data[sample][labelIndex]] = oneHotMapping_.size();
        }
    }
    labels_.resize(num_samples);
    for (int sample = 1; sample < data.size(); ++sample) {
        if (oneHotMapping_.find(data[sample][labelIndex]) == oneHotMapping_.end()) {
            throw std::runtime_error("Label not found in mapping.");
        }
        Eigen::RowVectorXf oneHotVector = Eigen::RowVectorXf::Zero(oneHotMapping_.size());
        oneHotVector(oneHotMapping_.at(data[sample][labelIndex])) = 1.0f;
        labels_[sample-1] = oneHotVector;
    }
    if (shuffle) {
        this->shuffle();
    }
}

NumericDataLoader::NumericDataLoader(const std::vector<Eigen::MatrixXf>& features, 
        const std::vector<Eigen::MatrixXf>& labels)
    : features_(features), labels_(labels) {}

IndividualInputData<Eigen::MatrixXf, Eigen::MatrixXf> NumericDataLoader::get_individual_data(float trainToTestSplitRatio) {
    if (trainToTestSplitRatio < 0.0f || trainToTestSplitRatio > 1.0f) {
        throw std::invalid_argument("Invalid train-test split ratio. (Expected: 0.0-1.0)");
    }
    size_t numTrainSamples = static_cast<size_t>(features_.size() * trainToTestSplitRatio);
    size_t numTestSamples = features_.size() - numTrainSamples;
    
    IndividualInputData<Eigen::MatrixXf, Eigen::MatrixXf> result;
    result.training.samples.reserve(numTrainSamples);
    for (size_t i = 0; i < numTrainSamples; ++i) {
        result.training.samples.push_back(IndividualDataSample<Eigen::MatrixXf, Eigen::MatrixXf>(features_[i], labels_[i]));
    }
    for (size_t i = numTrainSamples; i < features_.size(); ++i) {
        result.testing.samples.push_back(IndividualDataSample<Eigen::MatrixXf, Eigen::MatrixXf>(features_[i], labels_[i]));
    }
    
    return result;
}

BatchInputData<Eigen::MatrixXf, Eigen::MatrixXf> NumericDataLoader::get_batch_data(float trainToTestSplitRatio, int batch_size) {
    if (trainToTestSplitRatio < 0.0f || trainToTestSplitRatio > 1.0f) {
        throw std::invalid_argument("Invalid train-test split ratio. (Expected: 0.0-1.0)");
    }
    if (batch_size <= 0 || batch_size > features_.size()) {
        Console::log("Invalid batch size. Defaulting to 1.", Console::WARNING);
        batch_size = 1;
    }
    size_t numTrainSamples = static_cast<size_t>(features_.size() * trainToTestSplitRatio);
    size_t numTestSamples = features_.size() - numTrainSamples;
    if (numTrainSamples < batch_size || numTestSamples < batch_size) {
        Console::log("Batch size exceeds number of samples. Defaulting to 1.", Console::WARNING);
        batch_size = 1;
    }
    
    BatchInputData<Eigen::MatrixXf, Eigen::MatrixXf> result;
    result.training.set_batch_size(batch_size);
    result.testing.set_batch_size(batch_size);
    
    size_t sample = 0;
    while (sample < features_.size()) {
        // Construct a MatrixXf batch data
        Eigen::MatrixXf batch_features(batch_size, features_[0].cols());
        Eigen::MatrixXf batch_labels(batch_size, labels_[0].cols());
        if (sample + batch_size > features_.size()) {
            batch_features.resize(features_.size() - sample, Eigen::NoChange);
            batch_labels.resize(features_.size() - sample, Eigen::NoChange);
        }
        for (int i = 0; i < batch_size; ++i) {
            batch_features.row(i) = features_[sample];
            batch_labels.row(i) = labels_[sample];
            sample++;
        }
        result.training.samples.push_back(BatchDataSample<Eigen::MatrixXf, Eigen::MatrixXf>(batch_features, batch_labels));
        sample++;
    }
    for (size_t i = 0; i < numTrainSamples; ++i) {
        // Construct a MatrixXf batch data
        Eigen::MatrixXf batch_features(batch_size, features_[0].cols());
        Eigen::MatrixXf batch_labels(batch_size, labels_[0].cols());
        for (int j = 0; j < batch_size; ++j) {
            // Push whatever is left if the batch size is not a multiple of the number of samples
            if (i*batch_size+j >= features_.size()) {
                batch_features.resize(j, Eigen::NoChange);
                batch_labels.resize(j, Eigen::NoChange);
                break;
            }
            batch_features.row(j) = features_[i*batch_size+j];
            batch_labels.row(j) = labels_[i*batch_size+j];
        }
        result.training.samples.push_back(BatchDataSample<Eigen::MatrixXf, Eigen::MatrixXf>(batch_features, batch_labels));
    }
    for (size_t i = numTrainSamples; i < features_.size(); ++i) {
        // Construct a MatrixXf batch data
        Eigen::MatrixXf batch_features(batch_size, features_[0].cols());
        Eigen::MatrixXf batch_labels(batch_size, labels_[0].cols());
        for (int j = 0; j < batch_size; ++j) {
            if (i*batch_size+j >= features_.size()) {
                batch_features.resize(j, Eigen::NoChange);
                batch_labels.resize(j, Eigen::NoChange);
                break;
            }
            batch_features.row(j) = features_[i*batch_size+j];
            batch_labels.row(j) = labels_[i*batch_size+j];
        }
        result.testing.samples.push_back(BatchDataSample<Eigen::MatrixXf, Eigen::MatrixXf>(batch_features, batch_labels));
    }
    
    return result;
}

NumericDataLoader& NumericDataLoader::shuffle() {
    std::vector<size_t> indices(features_.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    std::vector<Eigen::MatrixXf> shuffled_features(features_.size(), Eigen::MatrixXf(1, features_[0].cols()));
    std::vector<Eigen::MatrixXf> shuffled_labels(labels_.size(), Eigen::MatrixXf(1, labels_[0].cols()));
    for (size_t i = 0; i < indices.size(); ++i) {
        shuffled_features[i] = features_[indices[i]];
        shuffled_labels[i] = labels_[indices[i]];
    }
    features_ = shuffled_features;
    labels_ = shuffled_labels;

    return *this;
}

NumericDataLoader& NumericDataLoader::center(float center_val) {
    // Calculate the mean of each column
    Eigen::VectorXf mean = Eigen::VectorXf::Zero(features_[0].cols());
    for (const auto& feature : features_) {
        mean += feature;
    }
    mean /= features_.size();
    mean = mean.array() + center_val;
    // Subtract the mean from each feature
    for (auto& feature : features_) {
        feature.rowwise() -= mean.transpose();
    }
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