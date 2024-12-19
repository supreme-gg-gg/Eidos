#ifndef TENSOR_H
#define TENSOR_H

#include <Eigen/Dense>
#include <vector>
#include <iostream>

class Tensor {
public:
    // Constructor for an empty tensor
    Tensor() = default;

    // Constructor to initialize a tensor with a specific number of matrices
    Tensor(size_t depth, size_t rows, size_t cols)
        : data_(depth, Eigen::MatrixXf::Zero(rows, cols)) {}

    // Constructor to initialize from a single matrix
    explicit Tensor(const Eigen::MatrixXf& matrix) {
        data_.push_back(matrix);
    }

    // Constructor to initialize from a vector of matrices
    explicit Tensor(const std::vector<Eigen::MatrixXf>& matrices)
        : data_(matrices) {}

    // Access element by depth and indices (const and non-const)
    Eigen::MatrixXf& operator[](size_t index) {
        return data_[index];
    }

    const Eigen::MatrixXf& operator[](size_t index) const {
        return data_[index];
    }

    // Get the depth (number of matrices)
    size_t depth() const {
        return data_.size();
    }

    // Get the shape (returns {depth, rows, cols})
    std::tuple<size_t, size_t, size_t> shape() const {
        if (data_.empty()) return {0, 0, 0};
        return {data_.size(), data_[0].rows(), data_[0].cols()};
    }

    // Utility for resizing the tensor
    void resize(size_t depth, size_t rows, size_t cols) {
        data_.resize(depth, Eigen::MatrixXf::Zero(rows, cols));
    }

    // Scalar operations
    Tensor& operator*=(float scalar) {
        for (auto& matrix : data_) {
            matrix.array() *= scalar;
        }
        return *this;
    }

    Tensor operator*(float scalar) const {
        Tensor result = *this;
        result *= scalar;
        return result;
    }

    Tensor& operator-=(float scalar) {
        for (auto& matrix : data_) {
            matrix.array() -= scalar;
        }
        return *this;
    }

    Tensor operator-(float scalar) const {
        Tensor result = *this;
        result -= scalar;
        return result;
    }

    Tensor& operator/=(float scalar) {
        if (scalar == 0.0f) {
            throw std::invalid_argument("Division by zero is not allowed.");
        }
        for (auto& matrix : data_) {
            matrix.array() /= scalar;
        }
        return *this;
    }

    Tensor operator/(float scalar) const {
        Tensor result = *this;
        result /= scalar;
        return result;
    }

    // Tensor element-wise addition
    Tensor& operator+=(const Tensor& other) {
        if (data_.size() != other.data_.size()) {
            throw std::invalid_argument("Tensors must have the same size for addition.");
        }
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i].array() += other.data_[i].array();
        }
        return *this;
    }

    Tensor operator+(const Tensor& other) const {
        Tensor result = *this;
        result += other;
        return result;
    }

    // Flatten tensor to a single matrix
    Eigen::MatrixXf flatten() const {
        if (data_.empty()) return Eigen::MatrixXf();
        size_t rows = data_[0].rows();
        size_t cols = data_[0].cols();
        Eigen::MatrixXf flat(rows * data_.size(), cols);
        for (size_t i = 0; i < data_.size(); ++i) {
            flat.middleRows(i * rows, rows) = data_[i];
        }
        return flat;
    }

    // Slicing function: Get a slice of the tensor (e.g., extracting (1, batch_size, features) from (num_batch, batch_size, features))
    Tensor slice(int batch_idx) const {
        int num_batch = data_.size();
        int batch_size = data_[0].rows();
        
        // Ensure that the batch_idx is within the valid range
        if (batch_idx < 0 || batch_idx >= num_batch) {
            std::cerr << "Error: Batch index out of range!" << std::endl;
            return Tensor(); // Return an empty tensor in case of invalid index
        }

        // Reshape the slice to (1, batch_size, features)
        Tensor slice_data(1, batch_size, data_[0].cols());
        
        // Extract the data for the slice (1, batch_size, features)
        slice_data.push_back(data_[batch_idx]);

        return slice_data;
    }

    // Add a new matrix to the tensor
    void push_back(const Eigen::MatrixXf& matrix) {
        data_.push_back(matrix);
    }

    // Remove the last matrix from the tensor
    void pop_back() {
        if (data_.empty()) {
            throw std::runtime_error("Cannot pop from an empty tensor.");
        }
        data_.pop_back();
    }

    // Check if the tensor is a single matrix
    bool isSingleMatrix() const {
        return data_.size() == 1;
    }

    const Eigen::MatrixXf& getSingleMatrix() const {
        if (!isSingleMatrix()) {
            throw std::runtime_error("Tensor does not contain a single matrix.");
        }
        return data_[0];
    }

    // Iterators for easy traversal
    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto begin() const { return data_.cbegin(); }
    auto end() const { return data_.cend(); }

private:
    std::vector<Eigen::MatrixXf> data_;
};

#endif // TENSOR_H