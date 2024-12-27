/*
TODO: the image loader class is currently incomplete. The class is supposed to load images from a directory and preprocess them for use in a neural network. The class should be able to load images in different formats, resize them, convert them to grayscale, and split the data into training and testing sets. The class should also be able to shuffle the data and transform the images linearly.
*/

#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include "data_loader.h"
#include "../tensor.hpp"
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <map>


class ImageLoader : public DataLoader<std::vector<Eigen::MatrixXf>> {
public:
    ImageLoader(const std::string& filePath, 
                      const std::string labelsHeaderName = "labels");
    /**
     * @brief Splits the data into training and testing sets.
     * @param trainToTestSplitRatio Ratio of training to testing data.
     * @return InputData structure containing split data.
     */
    InputData train_test_split(float trainToTestSplitRatio = 0.8f);

    /**
     * @brief Shuffles the data.
     * @return Reference to the current object fluently.
     */
    ImageLoader& shuffle();

    ImageLoader& linear_transform(float a, float b);
    
    ImageLoader& resize(int height, int width);

    ImageLoader& to_grayscale();

    /**
     * @brief Gets the number of samples in the data.
     * @return Number of samples.
     */
    size_t num_samples() const;

    /**
     * @brief Gets the number of features in the data.
     * @return Number of features.
     */
    size_t num_features() const;

    /**
     * @brief Gets the number of classes in the labels.
     * @return Number of classes.
     */
    size_t num_classes() const;

    /**
     * @brief Gets the shape of the data.
     * @return Pair of integers representing the shape (rows, columns).
     */
    std::pair<int, int> shape() const;

    /**
     * @brief Prints a preview of the data.
     * @param num_columns Number of columns to preview.
     */
    void print_preview(int num_columns = 5) const;

private:
    std::vector<Tensor> features_; ///< Matrix of features num_samples * (channels, height, width)
    Eigen::MatrixXf labels_;   ///< Matrix of labels
    std::map<std::string, int> oneHotMapping_; ///< Mapping for one-hot encoding
};

#endif // IMAGE_LOADER_H