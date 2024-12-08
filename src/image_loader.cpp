#include "../include/image_loader.h"
#include "../include/console.hpp"
#include "../include/csvparser.h"
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

void ImageLoader::load_data(const std::string& filePath, std::vector<std::vector<Eigen::MatrixXf>>& features, std::vector<std::string>& labels) {
    Console::log("George0", Console::DEBUG);
    CSVParser parser = CSVParser(',');
    std::vector<std::vector<std::string>> data = parser.parse(filePath);
    
    // Extract headers
    std::vector<std::string> headers = data[0];
    int labelIndex = -1;
    // Find the index of the "label" column
    for (size_t i = 0; i < headers.size(); ++i) {
        std::string fieldName = headers[i];
        std::transform(fieldName.begin(), fieldName.end(), fieldName.begin(),[](unsigned char c){ return std::tolower(c); });
        if (fieldName == "label" || fieldName == "labels") {
            labelIndex = i;
            break;
        }
    }

    if (labelIndex == -1) {
        throw std::runtime_error("Label column not found in CSV file.");
    }

    // Extract labels and features
    fs::path currentDir = fs::current_path();
    Console::log(fs::absolute(filePath).parent_path().string(), Console::DEBUG);
    fs::current_path(fs::absolute(filePath).parent_path());
    fs::path workingDir = fs::current_path();
    Console::log("George69", Console::DEBUG);
    labels.clear();
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i].size() != 2) {
            throw std::runtime_error("Invalid data entry in CSV file.");
        }
        labels.push_back(data[i][labelIndex]);
        
        std::vector<Eigen::MatrixXf> imgData(3);
        fs::path imgPath = data[i][1-labelIndex];
        if (fs::is_regular_file(imgPath)) {
            std::string extension = imgPath.extension().generic_string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp") {
                //fs::current_path(fs::absolute(filePath).parent_path());
                if (loadImage(fs::absolute(data[i][1-labelIndex]), &imgData[0], &imgData[1], &imgData[2], workingDir) != 0) {
                    imgData.clear();
                };
            }
            else {
                Console::log("Bad entry in data labels: In entry "+std::to_string(i)+": Unsupported image format \""+extension+"\". Skipped.", Console::WARNING);
                imgData.clear();
            }
        }
        else {
            Console::log("Bad entry in data labels: In entry "+std::to_string(i)+": \""+imgPath.generic_string()+"\" is not a valid file. Skipped.", Console::WARNING);
            imgData.clear();
        }
        features.push_back(imgData);
    }
    fs::current_path(currentDir);
}

void ImageLoader::split_data(const std::vector<std::vector<Eigen::MatrixXf>>& features, const std::vector<std::string>& labels, std::vector<std::vector<Eigen::MatrixXf>>& train_features, std::vector<std::string>& train_labels, std::vector<std::vector<Eigen::MatrixXf>>& test_features, std::vector<std::string>& test_labels, float trainToTestSplitRatio) {
    std::vector<int> indices(features.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    train_features.clear();
    train_labels.clear();
    test_features.clear();
    test_labels.clear();
    size_t i = 0;
    for (; i < indices.size() * trainToTestSplitRatio; ++i) {
        train_features.push_back(features[indices[i]]);
        train_labels.push_back(labels[indices[i]]);
    }
    for (; i < indices.size(); ++i) {
        test_features.push_back(features[indices[i]]);
        test_labels.push_back(labels[indices[i]]);
    }
}

void ImageLoader::convert_to_one_hot(const std::vector<std::string>& labels, std::vector<Eigen::MatrixXf>& one_hot_labels, const std::map<std::string, int>& mapping) {
    one_hot_labels.clear();
    for (const std::string& label : labels) {
        if (mapping.find(label) == mapping.end()) {
            throw std::runtime_error("Label not found in mapping.");
        }
        Eigen::MatrixXf oneHotVector = Eigen::MatrixXf::Zero(1, mapping.size());
        oneHotVector(0, mapping.at(label)) = 1.0f;
        one_hot_labels.push_back(oneHotVector);
    }
}