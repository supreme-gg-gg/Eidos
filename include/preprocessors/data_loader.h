#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <map>

struct Dataset {
    Tensor inputs;
    Tensor targets;
    Dataset(const Tensor& inputs, const Tensor& targets) : inputs(inputs), targets(targets) {}
    Dataset() = default;
};

struct InputData {
    Dataset training;
    Dataset testing;
    size_t num_features() const { return num_features_; }
    size_t num_classes() const { return num_classes_; }
    InputData(size_t num_features, size_t num_classes) : training(), testing(), num_features_(num_features), num_classes_(num_classes) {}
    InputData(const Dataset& training, const Dataset& testing, const size_t num_features, const size_t num_classes) : training(training), testing(testing), num_features_(num_features), num_classes_(num_classes) {}
    InputData() = default;
private:
    size_t num_features_;
    size_t num_classes_;
};

template <typename T>
class DataLoader {
public:
    virtual ~DataLoader() = default;

    /**
     * @brief Pure virtual function to load data from a specified file.
     * 
     * This function must be implemented by derived classes to load data from the given file path.
     * The loaded data should be stored in the provided vectors for features and labels.
     * 
     * @param filePath The path to the file from which to load data.
     * @param features A vector to store the loaded feature matrices.
     * @param labels A vector to store the corresponding labels.
     */
    virtual void load_data(const std::string& filePath, std::vector<T>& features, 
        std::vector<std::string>& labels) = 0;

    // Overload for loading data in place
    virtual void load_data(const std::string& filePath) = 0;
    
    /**
     * @brief Splits the dataset into training and testing sets.
     * 
     * @param features A vector of Eigen::MatrixXf containing the feature matrices.
     * @param labels A vector of strings containing the corresponding labels.
     * @param train_features A vector of Eigen::MatrixXf to store the training feature matrices.
     * @param train_labels A vector of strings to store the training labels.
     * @param test_features A vector of Eigen::MatrixXf to store the testing feature matrices.
     * @param test_labels A vector of strings to store the testing labels.
     */
    virtual void split_data(const std::vector<T>& features, 
        const std::vector<std::string>& labels, 
        std::vector<T>& train_features, 
        std::vector<std::string>& train_labels, 
        std::vector<T>& test_features, 
        std::vector<std::string>& test_labels, 
        float trainToTestSplitRatio) = 0;

    virtual void split_data(float trainToTestSplitRatio) = 0;

    /**
     * @brief Converts string labels to one-hot encoded labels.
     * 
     * This function takes a vector of string labels and converts them into one-hot encoded labels.
     * The conversion is done using a std::map.
     * 
     * @tparam T2 The type of the mapping object used for conversion.
     * @param labels A vector of string labels to be converted.
     * @param one_hot_labels A vector to store the resulting one-hot encoded labels.
     * @param mapping A std::map object used to map string labels to their corresponding one-hot encoded values.
     */
    virtual void convert_to_one_hot(const std::vector<std::string>& labels, 
        std::vector<Eigen::MatrixXf>& one_hot_labels, const std::map<std::string, int>& mapping) = 0;

    virtual void convert_to_one_hot(const std::map<std::string, int>& mapping) = 0;

    // These getters should be overridden in derived classes if supported
    virtual std::vector<T>& get_features() {
        throw std::logic_error("Getter not supported in this DataLoader variant.");
    }

    virtual std::vector<std::string>& get_labels() {
        throw std::logic_error("Getter not supported in this DataLoader variant.");
    }
};

#endif // DATA_LOADER_H