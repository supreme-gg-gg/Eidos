#include "../include/Eidos/preprocessors/image_loader.h"
#include "../include/Eidos/console.hpp"
#include "../include/Eidos/csvparser.h"
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <random>

namespace fs = std::filesystem;

/*
今有化圖為矩陣之術。欲施。必先得圖之徑矣，先設路徑以入之。
既入。夫矩陣三則。畜赤青靛三色。其矩陣者，取指針以授之於術。
繩圖之尺寸而使其大小均齊。其參差者。取左上而去其餘。
既出。充矩陣以圖之彩。
若功成則歸零。禍者則歸乎一。書之。乃作罷。
可用圖式有四。PNG。JPG。JPEG。BMP云云。
慎之哉。欲施此術。必選無保護之徑。蓋其將生臨時之檔。若逢障礙則難成也。
*/
int loadImage(fs::path imagePath, Eigen::MatrixXf* redMatrix, Eigen::MatrixXf* greenMatrix, Eigen::MatrixXf* blueMatrix, fs::path workingDir) {
    Console::log("George is engaging in the art of image matrix transformation...", Console::WORSHIP);
    fs::path imgPath = imagePath;
    if (!fs::exists(imgPath) || !fs::is_regular_file(imgPath)) {
        Console::log("\""+imgPath.generic_string()+"\" is not a valid file.", Console::ERROR);
        return 1;
    }
    
    imgPath = fs::absolute(imgPath);
    
    std::string imgExtension = imgPath.extension().generic_string();
    std::string command[3];
    if (imgExtension == ".jpeg" || imgExtension == ".jpg" || imgPath == ".JPG" || imgPath == ".JPEG") {
        command[0] = "jpegtopnm";
    }
    else if (imgExtension == ".png" || imgExtension == ".PNG") {
        command[0] = "pngtopnm";
    }
    else if (imgExtension == ".bmp" || imgExtension == ".BMP") {
        command[0] = "bmptopnm";
    }
    else {
        Console::log("Unknown image format \""+imgExtension+"\".", Console::ERROR);
        return 1;
    }

    Console::log("George1: "+workingDir.string(), Console::DEBUG);
    fs::current_path(workingDir.string());
    Console::log("George1-1", Console::DEBUG);
    fs::path currentDir = fs::current_path();
    fs::create_directory("temp");
    Console::log(currentDir.string(), Console::DEBUG);
    fs::current_path(currentDir / "temp");
    command[0] += " -plain "+imgPath.generic_string()+" > "+imgPath.stem().generic_string()+"-0.ppm 2> nul";
    command[1] = "pamscale -xyfill 256 256 "+imgPath.stem().generic_string()+"-0.ppm > "+imgPath.stem().generic_string()+"-1.ppm";
    command[2] = "pamcut -width 256 -height 256 "+imgPath.stem().generic_string()+"-1.ppm > "+imgPath.stem().generic_string()+".ppm";
    for (int i = 0; i < 3; i++) {
        if (system(command[i].c_str()) != 0) {
            Console::log("Failed to transform image into PNM for processing", Console::ERROR);
            return 1;
        };
    }
    Console::log("Image translated to PNM format for processing.", Console::DEBUG);
    
    int imgWidth, imgHeight, imgMaxColor;
    std::string imgFormat;
    std::ifstream ppmFile(imgPath.stem().generic_string()+".ppm");
    std::string message;
    if (ppmFile.is_open()) {
        std::string line;
        if (std::getline(ppmFile, line)) {
            std::istringstream iss(line);
            if (!(iss >> imgFormat)) {
                Console::log("Invalid PNM file!", Console::ERROR);
                return 1;
            }
        }
        line = "";
        if (std::getline(ppmFile, line)) {
            std::istringstream iss(line);
            if (!(iss >> imgWidth >> imgHeight)) {
                Console::log("Invalid PNM file!", Console::ERROR);
                return 1;
            }
        }
        if (std::getline(ppmFile, line)) {
            std::istringstream iss(line);
            if (!(iss >> imgMaxColor)) {
                Console::log("Invalid PNM file!", Console::ERROR);
                return 1;
            }
        }
        
        Console::log("Starting to read PNM file content...", Console::DEBUG);
        redMatrix->resize(imgHeight, imgWidth);
        greenMatrix->resize(imgHeight, imgWidth);
        blueMatrix->resize(imgHeight, imgWidth);
        int col = 0;
        int row = 0;
        while (std::getline(ppmFile, line)) {
            std::istringstream iss(line);
            int buffer[3];
            while (iss >> buffer[0] >> buffer[1] >> buffer[2]) {
                (*redMatrix)(row, col) = buffer[0] / 255.0;
                (*greenMatrix)(row, col) = buffer[1] / 255.0;
                (*blueMatrix)(row, col) = buffer[2] / 255.0;
                (*greenMatrix)(row, col) = buffer[1];
                (*blueMatrix)(row, col) = buffer[2];
                col++;
                if (col >= imgWidth) {
                    col = 0;
                    row++;
                }
            }
        }
        
        ppmFile.close();
        std::string ppm0Path = imgPath.stem().generic_string()+"-0.ppm";
        std::string ppm1Path = imgPath.stem().generic_string()+"-1.ppm";
        std::string ppmPath = imgPath.stem().generic_string()+".ppm";
        remove(ppm0Path.c_str());
        remove(ppm1Path.c_str());
        remove(ppmPath.c_str());
        fs::current_path(currentDir);
        
        Console::log("Image transformed into matrices successfully!", Console::DEBUG);
        return 0;
    } else {
        Console::log("Unable to open PNM file.", Console::ERROR);
        return 1;
    }
}

ImageLoader::ImageLoader(const std::string& filePath, 
        const std::string labelsHeaderName)
{
    Console::log("George is engaged in the glorious labor of image data loading...", Console::WORSHIP);
    CSVParser parser = CSVParser(',');
    std::vector<std::vector<std::string>> data = parser.parse(filePath);
    
    // Extract headers
    std::vector<std::string> headers = data[0];
    int labelIndex = -1;
    // Find the index of the "label" column
    for (size_t i = 0; i < headers.size(); ++i) {
        std::string fieldName = headers[i];
        std::transform(fieldName.begin(), fieldName.end(), fieldName.begin(),[](unsigned char c){ return std::tolower(c); });
        if (fieldName == labelsHeaderName) {
            labelIndex = i;
            break;
        }
    }

    if (labelIndex == -1) {
        throw std::runtime_error("Label column not found in CSV file.");
    }
    size_t num_samples = data.size() - 1;
    size_t num_valid_samples = 0;
    
    // Extract labels and features
    for (size_t sample = 1; sample < num_samples; ++sample) {
        if (oneHotMapping_.find(data[sample][labelIndex]) == oneHotMapping_.end()) {
            oneHotMapping_[data[sample][labelIndex]] = oneHotMapping_.size();
        }
    }
    features_.clear();
    features_.reserve(num_samples);
    labels_.resize(num_samples, oneHotMapping_.size());
    fs::path currentDir = fs::current_path();
    Console::log(fs::absolute(filePath).parent_path().string(), Console::DEBUG);
    fs::current_path(fs::absolute(filePath).parent_path());
    fs::path workingDir = fs::current_path();
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i].size() != 2) {
            throw std::runtime_error("Invalid data entry in CSV file.");
        }
        
        std::vector<Eigen::MatrixXf> imgData(3);
        fs::path imgPath = data[i][1-labelIndex];
        if (fs::is_regular_file(imgPath)) {
            std::string extension = imgPath.extension().generic_string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp") {
                //fs::current_path(fs::absolute(filePath).parent_path());
                if (loadImage(fs::absolute(data[i][1-labelIndex]), &imgData[0], &imgData[1], &imgData[2], workingDir) == 0) {
                    features_.push_back(Tensor(imgData));
                    num_valid_samples++;
                    if (oneHotMapping_.find(data[i][labelIndex]) == oneHotMapping_.end()) {
                        throw std::runtime_error("Label not found in mapping.");
                    }
                    Eigen::RowVectorXf oneHotVector = Eigen::RowVectorXf::Zero(oneHotMapping_.size());
                    oneHotVector(oneHotMapping_.at(data[i][labelIndex])) = 1.0f;
                    labels_.row(num_valid_samples-1) = oneHotVector;
                }
                else {
                    imgData.clear();
                    Console::log("A mysterious error occurred while loading image: "+data[i][1-labelIndex], Console::ERROR);
                    continue;
                };
            }
            else {
                Console::log("Bad entry in data labels: In entry "+std::to_string(i)+": Unsupported image format \""+extension+"\". Skipped.", Console::WARNING);
                imgData.clear();
                continue;
            }
        }
        else {
            Console::log("Bad entry in data labels: In entry "+std::to_string(i)+": \""+imgPath.generic_string()+"\" is not a valid file. Skipped.", Console::WARNING);
            imgData.clear();
            continue;
        }
    }
    fs::current_path(currentDir);
    labels_ = labels_.topRows(num_valid_samples);
    Console::log("Image data loaded successfully.", Console::DEBUG);
}

// /*
// InputData NumericDataLoader::train_test_split(float trainToTestSplitRatio) {
//     if (trainToTestSplitRatio < 0.0f || trainToTestSplitRatio > 1.0f) {
//         throw std::invalid_argument("Invalid train-test split ratio. (Expected: 0.0-1.0)");
//     }
//     size_t numTrainSamples = static_cast<size_t>(features_.rows() * trainToTestSplitRatio);
//     size_t numTestSamples = features_.rows() - numTrainSamples;
//     size_t numTrainBatches = numTrainSamples / batch_size;
//     size_t numTestBatches = numTestSamples / batch_size;
//     // Check if there are any samples left out
//     size_t numLeftOut = features_.rows() - numTrainBatches*batch_size - numTestBatches*batch_size;
//     if (numLeftOut > 0) {
//         Console::log(std::to_string(100.0f*numLeftOut / features_.rows())+"% of samples will be left out due to divisibility by batch size.", Console::WARNING);
//     }
//     if (numTrainBatches == 0 || numTestBatches == 0) {
//         throw std::invalid_argument("Batch size too large for the given split ratio.");
//     }
    
//     InputData result(features_.cols(), oneHotMapping_.size());
//     for (size_t i = 0; i < numTrainBatches; ++i) {
//         Eigen::MatrixXf train_features(batch_size, features_.cols());
//         Eigen::MatrixXf train_labels(batch_size, labels_.cols());
//         for (size_t j = 0; j < batch_size; ++j) {
//             train_features.row(j) = features_.row(i*batch_size + j);
//             train_labels.row(j) = labels_.row(i*batch_size + j);
//         }
//         result.training.inputs.push_back(train_features);
//         result.training.targets.push_back(train_labels);
//     }
//     for (size_t i = numTrainBatches; i < numTrainBatches + numTestBatches; ++i) {
//         Eigen::MatrixXf test_features(batch_size, features_.cols());
//         Eigen::MatrixXf test_labels(batch_size, labels_.cols());
//         for (size_t j = 0; j < batch_size; ++j) {
//             test_features.row(j) = features_.row(i*batch_size + j);
//             test_labels.row(j) = labels_.row(i*batch_size + j);
//         }
//         result.testing.inputs.push_back(test_features);
//         result.testing.targets.push_back(test_labels);
//     }

//     // Debug print the content of result
//     /*
//     for (size_t i = 0; i < result.training.inputs.depth(); ++i) {
//         std::stringstream ss;
//         ss << "Training batch " << i << " features:\n" << result.training.inputs[i];
//         Console::log(ss.str(), Console::DEBUG);
//         ss.str("");
//         ss << "Training batch " << i << " labels:\n" << result.training.targets[i];
//         Console::log(ss.str(), Console::DEBUG);
//     }
//     for (size_t i = 0; i < result.testing.inputs.depth(); ++i) {
//         std::stringstream ss;
//         ss << "Testing batch " << i << " features:\n" << result.testing.inputs[i];
//         Console::log(ss.str(), Console::DEBUG);
//         ss.str("");
//         ss << "Testing batch " << i << " labels:\n" << result.testing.targets[i];
//         Console::log(ss.str(), Console::DEBUG);
//     }
//     */
    
//     return result;
// }


// NumericDataLoader& NumericDataLoader::shuffle() {
//     std::vector<size_t> indices(features_.rows());
//     std::iota(indices.begin(), indices.end(), 0);
//     std::random_device rd;
//     std::mt19937 g(rd());
//     std::shuffle(indices.begin(), indices.end(), g);

//     Eigen::MatrixXf shuffled_features(features_.rows(), features_.cols());
//     Eigen::MatrixXf shuffled_labels(labels_.rows(), labels_.cols());
//     for (size_t i = 0; i < indices.size(); ++i) {
//         shuffled_features.row(i) = features_.row(indices[i]);
//         shuffled_labels.row(i) = labels_.row(indices[i]);
//     }
//     features_ = shuffled_features;
//     labels_ = shuffled_labels;

//     return *this;
// }
// */