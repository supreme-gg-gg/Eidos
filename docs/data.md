# Data Preprocessing

Data preprocessing is an important step in machine learning. It involves transforming raw data into a format that is suitable for training a model. This can include tasks such as normalizing data, encoding categorical variables, and splitting data into training and testing sets.

Eidoes support structured data in the form of CSV files and images.

> We are working towards natively supporting time series in the near future.

## Numeric Data

The `NumericDataLoader` class is used to load and preprocess data. This class provides methods for loading data from a CSV file, encoding categorical variables, normalizing data, and splitting data into training and testing sets.

It provides the following methods:

- `load_data`: Load data from a CSV file.
- `shuffle`: Shuffle the data.
- `linear_transform`: Apply a linear transformation to the data.
- `min_max_scale`: Apply min-max scaling to the data.
- `z_score_normalize`: Apply z-score normalization to the data.
- `center`: Center the data around a specified value.
- `remove_outliers`: Remove outliers from the data.
- `pca`: Apply principal component analysis to the data.

And utility functions such as `print_preview`.

Here is an example of loading and preprocessing data using the `NumericDataLoader` class:

```cpp
NumericDataLoader data_loader("data.csv", "label", label_map);

data_loader.shuffle().min_max_scale(). linear_transform(1f/255f, 0.0f);

data_loader.print_preview();

auto data = data_loader.train_test_split(0.8, 32); // 80% training, 32 batch size
```

Here, `data` is of type `InputData` which contains the training and testing data of `Dataset` type. The `Dataset` type contains two tensors, `inputs` and `targets`.

You can implement your own data type and loader by understanding modules in `preprocessing/`.

## Image Data

The `ImageDataLoader` class is used to load and preprocess image data. This class provides methods for loading images from a directory, resizing images, normalizing pixel values, and splitting data into training and testing sets.

## Conclusion

You should now be familiar with the data preprocessing capabilities of Eidos. Please move on to the [next tutorial](./layers_plus.md) to learn about other powerful layers provided by Eidos.
