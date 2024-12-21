#include "../include/preprocessors/numeric_data_loader.h"
#include "../include/tensor.hpp"
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
        const std::string labelsHeaderName)
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
    features_.resize(num_samples, total_features);
    
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
        features_.row(sample-1) = sample_vector;
    }
    
    // Preprocess the labels
    for (size_t sample = 1; sample < num_samples; ++sample) {
        if (oneHotMapping_.find(data[sample][labelIndex]) == oneHotMapping_.end()) {
            oneHotMapping_[data[sample][labelIndex]] = oneHotMapping_.size();
        }
    }
    labels_.resize(num_samples, oneHotMapping_.size());
    for (int sample = 1; sample < data.size(); ++sample) {
        if (oneHotMapping_.find(data[sample][labelIndex]) == oneHotMapping_.end()) {
            throw std::runtime_error("Label not found in mapping.");
        }
        Eigen::RowVectorXf oneHotVector = Eigen::RowVectorXf::Zero(oneHotMapping_.size());
        oneHotVector(oneHotMapping_.at(data[sample][labelIndex])) = 1.0f;
        labels_.row(sample-1) = oneHotVector;
    }
    Console::log("Data loaded successfully.", Console::DEBUG);
}

NumericDataLoader::NumericDataLoader(const Eigen::MatrixXf& features, 
        const Eigen::MatrixXf& labels)
    : features_(features), labels_(labels) {}

InputData NumericDataLoader::train_test_split(float trainToTestSplitRatio, int batch_size) {
    if (trainToTestSplitRatio < 0.0f || trainToTestSplitRatio > 1.0f) {
        throw std::invalid_argument("Invalid train-test split ratio. (Expected: 0.0-1.0)");
    }
    size_t numTrainSamples = static_cast<size_t>(features_.rows() * trainToTestSplitRatio);
    size_t numTestSamples = features_.rows() - numTrainSamples;
    size_t numTrainBatches = numTrainSamples / batch_size;
    size_t numTestBatches = numTestSamples / batch_size;
    // Check if there are any samples left out
    size_t numLeftOut = features_.rows() - numTrainBatches*batch_size - numTestBatches*batch_size;
    if (numLeftOut > 0) {
        Console::log(std::to_string(100.0f*numLeftOut / features_.rows())+"% of samples will be left out due to divisibility by batch size.", Console::WARNING);
    }
    if (numTrainBatches == 0 || numTestBatches == 0) {
        throw std::invalid_argument("Batch size too large for the given split ratio.");
    }
    
    InputData result(features_.cols(), oneHotMapping_.size());
    for (size_t i = 0; i < numTrainBatches; ++i) {
        Eigen::MatrixXf train_features(batch_size, features_.cols());
        Eigen::MatrixXf train_labels(batch_size, labels_.cols());
        for (size_t j = 0; j < batch_size; ++j) {
            train_features.row(j) = features_.row(i*batch_size + j);
            train_labels.row(j) = labels_.row(i*batch_size + j);
        }
        result.training.inputs.push_back(train_features);
        result.training.targets.push_back(train_labels);
    }
    for (size_t i = numTrainBatches; i < numTrainBatches + numTestBatches; ++i) {
        Eigen::MatrixXf test_features(batch_size, features_.cols());
        Eigen::MatrixXf test_labels(batch_size, labels_.cols());
        for (size_t j = 0; j < batch_size; ++j) {
            test_features.row(j) = features_.row(i*batch_size + j);
            test_labels.row(j) = labels_.row(i*batch_size + j);
        }
        result.testing.inputs.push_back(test_features);
        result.testing.targets.push_back(test_labels);
    }

    // Debug print the content of result
    /*
    for (size_t i = 0; i < result.training.inputs.depth(); ++i) {
        std::stringstream ss;
        ss << "Training batch " << i << " features:\n" << result.training.inputs[i];
        Console::log(ss.str(), Console::DEBUG);
        ss.str("");
        ss << "Training batch " << i << " labels:\n" << result.training.targets[i];
        Console::log(ss.str(), Console::DEBUG);
    }
    for (size_t i = 0; i < result.testing.inputs.depth(); ++i) {
        std::stringstream ss;
        ss << "Testing batch " << i << " features:\n" << result.testing.inputs[i];
        Console::log(ss.str(), Console::DEBUG);
        ss.str("");
        ss << "Testing batch " << i << " labels:\n" << result.testing.targets[i];
        Console::log(ss.str(), Console::DEBUG);
    }
    */
    
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

NumericDataLoader& NumericDataLoader::center(float center_val) {
    Eigen::VectorXf mean = features_.colwise().mean();
    features_ = features_.rowwise() - mean.transpose();
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