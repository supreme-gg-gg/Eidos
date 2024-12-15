#ifndef GENERIC_DATA_LOADER_H
#define GENERIC_DATA_LOADER_H

#include "../include/data_loader.h"
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <iterator>

class GenericDataLoader : public DataLoader<Eigen::MatrixXf> {
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
    void load_data(const std::string& filePath, 
        std::vector<Eigen::MatrixXf>& features, 
        std::vector<std::string>& labels) override ;

    void load_data(const std::string& filePath) override {};

    /**
     * @brief Splits the dataset into training and testing sets based on the given ratio.
     *
     * @param features A vector of Eigen::MatrixXf containing the feature matrices.
     * @param labels A vector of strings containing the corresponding labels.
     * @param train_features A vector of Eigen::MatrixXf to store the training feature matrices.
     * @param train_labels A vector of strings to store the training labels.
     * @param test_features A vector of Eigen::MatrixXf to store the testing feature matrices.
     * @param test_labels A vector of strings to store the testing labels.
     * @param trainToTestSplitRatio A float representing the ratio of training data to testing data.
     */
    void split_data(const std::vector<Eigen::MatrixXf>& features, 
        const std::vector<std::string>& labels, 
        std::vector<Eigen::MatrixXf>& train_features, 
        std::vector<std::string>& train_labels, 
        std::vector<Eigen::MatrixXf>& test_features, 
        std::vector<std::string>& test_labels, 
        float trainToTestSplitRatio) override;

    void split_data(float trainToTestSplitRatio) override {};

    /**
     * @brief Converts string labels to one-hot encoded labels.
     *
     * This function takes a vector of string labels and converts them into one-hot encoded labels.
     * The conversion is done using a mapping object of type T2, which should be a key-value object
     * where values are accessible using T2[key] (map, unordered_map, etc.).
     *
     * @tparam T2 The type of the mapping object used for conversion.
     * @param labels A vector of string labels to be converted.
     * @param one_hot_labels A vector to store the resulting one-hot encoded labels.
     * @param mapping A key-value object used to map string labels to their corresponding one-hot encoded values.
     */
    void convert_to_one_hot(const std::vector<std::string>& labels, 
        std::vector<Eigen::MatrixXf>& one_hot_labels, 
        const std::map<std::string, int>& mapping) override ;

    void convert_to_one_hot(const std::map<std::string, int>& mapping) override {};
};

class BatchDataLoader : public DataLoader<Eigen::MatrixXf> {
public:
    // Constructor
    BatchDataLoader(const std::string& file_path, int batch_size);

    // Loads data from a file (e.g., CSV)
    void load_data(const std::string& file_path) override;

    // Splits the dataset into training and testing sets based on the given ratio
    void split_data(float trainToTestSplitRatio) override;

    // Converts string labels to one-hot encoded labels
    void convert_to_one_hot(const std::map<std::string, int>& mapping) override;

    // Returns the current batch of data as a tuple (features, labels)
    std::tuple<Eigen::MatrixXf, Eigen::MatrixXf> get_batch() const;

    // Sets the batch size for the DataLoader
    void set_batch_size(int batch_size);

    // Returns the number of batches available for iteration
    int num_batches() const;

    // Iterator class to enable iteration over batches
    class Iterator {
    public:
        Iterator(BatchDataLoader& data_loader, bool is_end = false);

        // Dereferencing operator to get the current batch
        std::tuple<Eigen::MatrixXf, Eigen::MatrixXf>& operator*();

        // Increment operator to move to the next batch
        Iterator& operator++();

        // Equality comparison for iterators (used by range-based for)
        bool operator==(const Iterator& other) const;
        bool operator!=(const Iterator& other) const;

    private:
        BatchDataLoader& data_loader_;
        int batch_idx_;
    };

    // Returns the beginning iterator for the DataLoader
    Iterator begin();

    // Returns the end iterator for the DataLoader
    Iterator end();

private:
    // Stores the data and labels
    std::vector<Eigen::MatrixXf> features_;
    std::vector<Eigen::MatrixXf> labels_;

    // Batch size for batching during training
    int batch_size_;

    // Current index for batch fetching
    int current_batch_idx_;

    // Number of batches
    int num_batches_;
};

#endif // GENERIC_DATA_LOADER_H
