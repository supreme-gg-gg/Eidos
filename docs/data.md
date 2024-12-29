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

Here, `data` is of type `InputData` (default return type of `train_test_split`) which contains the training and testing data of `Dataset` type. The `Dataset` type contains two tensors, `inputs` and `targets`.

You can implement your own data type and loader by understanding modules in `preprocessing/`.

### Converting Numeric Data to Images

In examples like MNIST, the data is provided as a CSV file with pixel values. You can convert this data to images by specifying the height and width of the images. This can be done easily by first loading the dataset with `NumericDataLoader`, then using the `train_test_split_images` method to convert the data to images.

```cpp
// reshape into 28x28 images with 80% training data
auto data = NumericDataLoader("mnist_train.csv", "label", label_map)
    .train_test_split_images(28, 28, 0.8);
```

Here `data` is of type `ImageInputData` described in the next section.

## Image Data

The `ImageDataLoader` class is used to load and preprocess image data. This class provides methods for loading images from a directory, resizing images, normalizing pixel values, and splitting data into training and testing sets.

It provides the following methods:

- `shuffle`: Shuffle the images.
- `linear_transform`: Apply a linear transformation to the pixel values.
- `resize`: Resize the images to a specified height and width.
- `to_grayscale`: Convert the images to grayscale.

> Note that image loader does NOT support batching because each image itself is a 3D tensor. We recommend using `ImageDataLoader` for small datasets.

Here is an example of loading and preprocessing image data using the `ImageDataLoader` class:

```cpp
ImageDataLoader image_loader("images", "labels");
image_loader.shuffle().resize(28, 28).to_grayscale();
auto data = image_loader.train_test_split(0.8);
```

Here, `data` is of type `ImageInputData` which contains the training and testing data of `ImageDataset` type. The `ImageDataset` type contains two tensors, `images` and `labels`.

## Conclusion

You should now be familiar with the data preprocessing capabilities of Eidos. Please move on to the [next tutorial](./layers_plus.md) to learn about other powerful layers provided by Eidos.
