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

    // Constructor to initialize a tensor with a vector of dimensions
    Tensor(const std::vector<int>& dimensions) {
        if (dimensions.size() != 3) {
            throw std::invalid_argument("Dimensions vector must have exactly 3 elements.");
        }
        size_t depth = dimensions[0];
        size_t rows = dimensions[1];
        size_t cols = dimensions[2];
        data_.resize(depth, Eigen::MatrixXf::Zero(rows, cols));
    }

    // Constructor to initialize from a single matrix
    explicit Tensor(const Eigen::MatrixXf& matrix) {
        data_.push_back(matrix);
    }

    // Constructor to initialize from a vector of matrices
    explicit Tensor(const std::vector<Eigen::MatrixXf>& matrices)
        : data_(matrices) {}

    // Access element by depth and indices (const and non-const)
    Eigen::MatrixXf& operator[](size_t index) {
        if (index < 0 || index >= data_.size()) {
            throw std::out_of_range("Index out of range.");
        }
        return data_[index];
    }

    const Eigen::MatrixXf& operator[](size_t index) const {
        if (index < 0 || index >= data_.size()) {
            throw std::out_of_range("Index out of range.");
        }
        return data_[index];
    }

    // Access element by depth, row, and column (const and non-const)
    float& operator()(size_t depth, size_t row, size_t col) {
        return data_[depth](row, col);
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

    void print_shape() const {
        auto [depth, rows, cols] = shape();
        std::cout << "Depth: " << depth << ", Rows: " << rows << ", Cols: " << cols << std::endl;
    }

    void set_random() {
        for (auto& matrix : data_) {
            matrix = Eigen::MatrixXf::Random(matrix.rows(), matrix.cols());
        }
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

        return Tensor({data_[batch_idx]});
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

    // Overload the << operator for printing
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        for (size_t c = 0; c < tensor.depth(); ++c) {
            os << "Channel " << c << ":\n";
            os << tensor.data_[c] << "\n";
        }
        return os;
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