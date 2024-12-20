#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <map>

template <typename T_inputs, typename T_targets>
struct DataSample {
    T_inputs inputs; ///< Features matrix
    T_targets targets; ///< Labels matrix

    DataSample() = default;
    DataSample(const T_inputs& inputs, const T_targets& targets)
        : inputs(inputs), targets(targets) {}

    virtual bool is_batch() const { return false; }
    virtual ~DataSample() = default;
};
template <typename T_inputs, typename T_targets>
struct IndividualDataSample : DataSample<T_inputs, T_targets> {
    using DataSample<T_inputs, T_targets>::DataSample;

    bool is_batch() const override { return false; }
};
template <typename T_inputs, typename T_targets>
struct BatchDataSample : DataSample<T_inputs, T_targets> {
    using DataSample<T_inputs, T_targets>::DataSample;

    bool is_batch() const override { return true; }
};

template <typename T_samples>
struct Dataset {
public:
    T_sample get_current();
    size_t num_samples() const ;
    // Iterator class to enable iteration over batches
    class Iterator {
    public:
        Iterator(DataSet<T_samples>& data, bool is_end = false);

        // Dereferencing operator to get the current batch
        T_sample& operator*();

        // Increment operator to move to the next batch
        Iterator& operator++();

        // Equality comparison for iterators (used by range-based for)
        bool operator==(const Iterator& other) const;
        bool operator!=(const Iterator& other) const;

    private:
        Dataset& data_;
        int current_idx_;
    };

    Iterator begin();
    Iterator end();

    std::vector<T_samples> samples;
};

template <typename T_inputs, typename T_targets>
struct IndividualDataset : public Dataset<IndividualDataSample<T_inputs, T_targets>> {
public:
    IndividualDataSample<T_inputs, T_targets> get_current();
    // Iterator class to enable iteration over batches
};

template <typename T_inputs, typename T_targets>
class BatchDataset : public Dataset<BatchDataSample<T_inputs, T_targets>> {
public:
    // Returns the current batch of data
    BatchDataSample<T_inputs, T_targets> get_current() const;
    // Sets the batch size for the DataLoader
    void set_batch_size(int batch_size);
    // Returns the number of batches available for iteration
    int num_batches() const;

private:
    size_t batch_size_;
    size_t current_batch_idx_;
    size_t num_batches_;
};

template <typename T_dataset>
struct InputData {
    T_dataset training; ///< Training data
    T_dataset testing;  ///< Testing data
};

template <typename T_inputs, typename T_targets>
struct IndividualInputData {
    IndividualDataset<T_inputs, T_targets> training; ///< Training data
    IndividualDataset<T_inputs, T_targets> testing;  ///< Testing data
};

template <typename T_inputs, typename T_targets>
struct BatchInputData {
    BatchDataset<T_inputs, T_targets> training; ///< Training data
    BatchDataset<T_inputs, T_targets> testing;  ///< Testing data
};

#endif // DATA_LOADER_H