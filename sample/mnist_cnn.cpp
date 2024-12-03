#include <mlpack.hpp>

using namespace mlpack;

using namespace arma;
using namespace std;

using namespace ens;

Row<size_t> getLabels(const mat& predOut)
{
  Row<size_t> predLabels(predOut.n_cols);
  for(uword i = 0; i < predOut.n_cols; ++i)
  {
    predLabels(i) = predOut.col(i).index_max();
  }
  return predLabels;
}

int main()
{

    constexpr double RATIO = 0.2; // split into valid and train with 0.1 ratio
    const int EPOCHS = 5; // allow 60 passes over training unless early stop
    const int BATCH_SIZE = 64; // num of data ponits in each iter of SGD
    const double STEP_SIZE = 1.2e-3; // step size

    mat dataset;
    data::Load("train.csv", dataset, true);
    mat train, valid;
    data::Split(dataset, train, valid, RATIO); // split the dataset into train and valid dataset

    // train and valid dataset contains both labels and features
    const mat trainX = train.submat(1, 0, train.n_rows - 1, train.n_cols - 1) / 256.0;
    const mat validX = valid.submat(1, 0, valid.n_rows - 1, valid.n_cols - 1) / 256.0;

    // labels are in the first row
    const mat trainY = train.row(0);
    const mat validY = valid.row(0);

    // NegativeLogLikelihood is the output layer used for the classification problem 
    FFN<NegativeLogLikelihood, RandomInitialization> model;

    // In this example, the CNN architecture is chosen similar to LeNet-5.
    // The architecture follows a Conv-ReLU-Pool-Conv-ReLU-Pool-Dense schema. We
    // have used leaky ReLU activation instead of vanilla ReLU. Standard
    // max-pooling has been used for pooling. The first convolution uses 6 filters
    // of size 5x5 (and a stride of 1). The second convolution uses 16 filters of
    // size 5x5 (stride = 1). The final dense layer is connected to a softmax to
    // ensure that we get a valid probability distribution over the output classes

    // Layers schema.
    // 28x28x1 --- conv (6 filters of size 5x5. stride = 1) ---> 24x24x6
    // 24x24x6 --------------- Leaky ReLU ---------------------> 24x24x6
    // 24x24x6 --- max pooling (over 2x2 fields. stride = 2) --> 12x12x6
    // 12x12x6 --- conv (16 filters of size 5x5. stride = 1) --> 8x8x16
    // 8x8x16  --------------- Leaky ReLU ---------------------> 8x8x16
    // 8x8x16  --- max pooling (over 2x2 fields. stride = 2) --> 4x4x16
    // 4x4x16  ------------------- Dense ----------------------> 10

    model.Add<Convolution>(6, 5, 5, 1, 1, 0, 0); // input act maps, output act maps, filter width, filter height, stride along width, stride along height, padding width, padding height
    model.Add<LeakyReLU>();
    model.Add<MaxPooling>(2, 2, 2, 2, true); // filter width, filter height, stride along width, stride along height, by default it uses the maximum value in the pooling region
    model.Add<BatchNorm>();
    model.Add<Convolution>(16, 5, 5, 1, 1, 0, 0);
    model.Add<LeakyReLU>();
    model.Add<MaxPooling>(2, 2, 2, 2, true);
    model.Add<BatchNorm>();
    model.Add<Linear>(10);
    model.Add<LogSoftMax>();

    model.InputDimensions() = vector<size_t>({28, 28});

    ens::Adam optimizer(
        STEP_SIZE,
        BATCH_SIZE,
        0.9, // exponential decay rates for the moment estimates
        0.999, // exponential decay rate for the weighted infinity norm estimates
        1e-8, // value used to initialise the mean squared gradient parameter
        EPOCHS * trainX.n_cols,
        1e-8, // tolerance
        true
    );

    model.Train(trainX,
                trainY,
                optimizer,
                ens::PrintLoss(),
                ens::ProgressBar(),
                ens::EarlyStopAtMinLoss( // stop training if the validation loss does not decrease for 5 consecutive epochs
                    [&](const arma::mat&)
                    {
                        double validationLoss = model.Evaluate(validX, validY);
                        std::cout << "Validation loss: " << validationLoss << std::endl;
                        return validationLoss;
                    }
                )
    );

    mat predOut; // matrix to store the predictions on train and valid dataset
    model.Predict(trainX, predOut);
    Row<size_t> predLabels = getLabels(predOut);
    double trainAccuracy = accu(predLabels == trainY) / (double) trainY.n_elem * 100;

    model.Predict(validX, predOut);
    predLabels = getLabels(predOut);
    double validAccuracy = accu(predLabels == validY) / (double) validY.n_elem * 100;
    
    std::cout << "Accuracy: train = " << trainAccuracy << "%,"<< "\t valid = " << validAccuracy << "%" << std::endl;

    mlpack::data::Save("model.bin", "model", model, false);

    data::Load("test.csv", dataset, true);
    const mat testX = dataset.submat(1, 0, dataset.n_rows - 1, dataset.n_cols - 1) / 256.0;
    const mat testY = dataset.row(0);

    mat testPredOut;
    model.Predict(testX, testPredOut);
    arma::Row<size_t> testPred = getLabels(testPredOut);
    double testAccuracy = accu(testPred == testY) / (double) testY.n_elem * 100;
    std::cout << "Test accuracy: " << testAccuracy << "%" << std::endl;

    cout << "Saving predicted labels to \"results.csv.\"..." << endl;
    predLabels.save("results.csv", arma::csv_ascii);

    cout << "Neural network model is saved to \"model.bin\"" << endl;
    cout << "Finished" << endl;

}