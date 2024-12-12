#ifndef TENSOR_H
#define TENSOR_H

#include <Eigen/Dense>
#include <vector>
#include <iostream>

template <typename T = Eigen::MatrixXf>
class Tensor {
public:
    // Constructor for an empty tensor
    Tensor() = default;

    // Constructor to initialize a tensor with a specific number of matrices
    Tensor(size_t depth, size_t rows, size_t cols) 
        : data_(depth, T::Zero(rows, cols)) {}

    // Access element by depth and indices (const and non-const)
    T& operator[](size_t index) {
        return data_[index];
    }

    const T& operator[](size_t index) const {
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
        data_.resize(depth, T::Zero(rows, cols));
    }

    // Scalar multiplication: Tensor * float
    Tensor<T>& operator*=(float scalar) {
        // Iterate over each matrix in the vector and perform element-wise multiplication
        for (auto& matrix : data_) {
            matrix.array() *= scalar;  // Element-wise multiplication using array()
        }
        return *this;
    }

    // Scalar subtraction: Tensor -= float
    Tensor<T>& operator-=(float scalar) {
        // Iterate over each matrix in the vector and perform element-wise subtraction
        for (auto& matrix : data_) {
            matrix.array() -= scalar;  // Element-wise subtraction using array()
        }
        return *this;
    }

    // Scalar multiplication with a tensor: float * Tensor
    Tensor<T> operator*(float scalar) const {
        Tensor<T> result = *this;
        result *= scalar;  // Reuse the *= operator for scalar multiplication
        return result;
    }

    // Scalar subtraction with a tensor: Tensor - float
    Tensor<T> operator-(float scalar) const {
        Tensor<T> result = *this;
        result -= scalar;  // Reuse the -= operator for scalar subtraction
        return result;
    }

    // Element-wise division: Tensor /= scalar
    Tensor<T>& operator/=(float scalar) {
        if (scalar == 0.0f) {
            throw std::invalid_argument("Division by zero is not allowed");
        }
        for (auto& matrix : data_) {
            matrix.array() /= scalar;  // Element-wise division using array()
        }
        return *this;
    }

    // Element-wise division: Tensor / scalar
    Tensor<T> operator/(float scalar) const {
        if (scalar == 0.0f) {
            throw std::invalid_argument("Division by zero is not allowed");
        }
        Tensor<T> result = *this;
        result /= scalar;  // Reuse the /= operator for division
        return result;
    }

    // Element-wise addition: Tensor += Tensor
    Tensor<T>& operator+=(const Tensor<T>& other) {
        if (data_.size() != other.data_.size()) {
            throw std::invalid_argument("Tensors must have the same size for addition");
        }
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i].array() += other.data_[i].array();  // Element-wise addition using array()
        }
        return *this;
    }

    // Element-wise addition: Tensor + Tensor
    Tensor<T> operator+(const Tensor<T>& other) const {
        Tensor<T> result = *this;
        result += other;  // Reuse the += operator for addition
        return result;
    }

    // Adding eleemnt
    void push_back(const T& matrix) {
        data_.push_back(matrix);
    }

private:
    std::vector<T> data_; // Vector of matrices
};

#endif // TENSOR_H